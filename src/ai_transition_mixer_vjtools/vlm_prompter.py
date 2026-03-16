"""
VLM Auto-Prompter — Local companion for AI Transition Mixer
=============================================================

Runs locally alongside Scope. Captures the Spout feed, splits it into
Deck A and Deck B, sends each to Qwen 2.5 VL via Ollama for scene
description, and pushes the generated prompts to Scope's REST API.

The plugin on the cloud side reads prompt_a / prompt_b from its config
and blends them based on the crossfader position.

Requirements:
  - Ollama running locally with qwen2.5-vl pulled
  - Scope running (local or cloud mode)

Usage:
  python -m ai_transition_mixer_vjtools.vlm_prompter \\
      --scope-url http://localhost:8000 \\
      --model qwen2.5-vl:3b \\
      --interval 3.0 \\
      --split side_by_side
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import time
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Ollama VLM Client ─────────────────────────────────────────────────────

def _frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Encode a numpy frame as base64 JPEG for Ollama vision API."""
    from PIL import Image
    img = Image.fromarray(frame)
    # Resize to save bandwidth / speed up VLM inference
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def describe_frame(
    frame: np.ndarray,
    model: str = "qwen2.5-vl:3b",
    ollama_url: str = "http://localhost:11434",
    system_prompt: str = (
        "You are a VJ visual describer. Describe this video frame in 8-15 words "
        "for an AI video generator. Focus on colors, movement, mood, and key subjects. "
        "Be vivid and concise. No sentences, just descriptive phrases."
    ),
) -> str:
    """Send a frame to Ollama VLM and get a scene description."""
    import urllib.request

    b64 = _frame_to_base64(frame)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Describe this scene:",
                "images": [b64],
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 50,
        },
    }

    req = urllib.request.Request(
        f"{ollama_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning(f"Ollama request failed: {e}")
        return ""


# ─── Scope API Client ──────────────────────────────────────────────────────

def update_scope_params(
    scope_url: str,
    prompt_a: Optional[str] = None,
    prompt_b: Optional[str] = None,
) -> bool:
    """Push updated prompt_a / prompt_b to Scope's parameter API."""
    import urllib.request

    params = {}
    if prompt_a is not None:
        params["prompt_a"] = prompt_a
    if prompt_b is not None:
        params["prompt_b"] = prompt_b

    if not params:
        return True

    payload = json.dumps(params).encode("utf-8")

    req = urllib.request.Request(
        f"{scope_url}/api/v1/pipeline/update",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="PATCH",
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"Failed to update Scope params: {e}")
        return False


# ─── Frame Capture ──────────────────────────────────────────────────────────

def capture_spout_frame(source_name: str = "") -> Optional[np.ndarray]:
    """Try to capture a frame from Spout. Returns None if unavailable."""
    try:
        import SpoutGL
        receiver = SpoutGL.SpoutReceiver()
        receiver.setReceiverName(source_name)
        # Try to receive
        info = receiver.getReceiverInfo()
        if info and info.width > 0:
            frame = np.zeros((info.height, info.width, 4), dtype=np.uint8)
            receiver.receiveImage(frame)
            return frame[:, :, :3]  # Drop alpha
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Spout capture failed: {e}")
    return None


def split_frame(
    frame: np.ndarray, split: str = "side_by_side"
) -> tuple[np.ndarray, np.ndarray]:
    """Split a composite frame into Deck A and Deck B."""
    H, W = frame.shape[:2]
    if split == "top_bottom":
        mid = H // 2
        return frame[:mid], frame[mid:]
    else:
        mid = W // 2
        return frame[:, :mid], frame[:, mid:]


# ─── Main Loop ──────────────────────────────────────────────────────────────

class VlmPrompter:
    """Periodically describes each deck and pushes prompts to Scope."""

    def __init__(
        self,
        scope_url: str = "http://localhost:8000",
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5-vl:3b",
        interval: float = 3.0,
        split: str = "side_by_side",
        spout_source: str = "",
    ):
        self.scope_url = scope_url
        self.ollama_url = ollama_url
        self.model = model
        self.interval = interval
        self.split = split
        self.spout_source = spout_source

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_prompt_a = ""
        self._current_prompt_b = ""

        # If Spout isn't available, accept frames pushed externally
        self._external_frame: Optional[np.ndarray] = None

    def push_frame(self, frame: np.ndarray):
        """Push a frame externally (for testing or non-Spout setups)."""
        self._external_frame = frame.copy()

    def start(self):
        """Start the background prompter loop."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            f"[VlmPrompter] Started (model={self.model}, "
            f"interval={self.interval}s, scope={self.scope_url})"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _get_frame(self) -> Optional[np.ndarray]:
        """Get the latest composite frame."""
        # Try Spout first
        frame = capture_spout_frame(self.spout_source)
        if frame is not None:
            return frame
        # Fall back to externally pushed frame
        return self._external_frame

    def _loop(self):
        while self._running:
            try:
                frame = self._get_frame()
                if frame is not None:
                    deck_a, deck_b = split_frame(frame, self.split)

                    # Describe both decks (sequentially — could parallelize)
                    desc_a = describe_frame(deck_a, self.model, self.ollama_url)
                    desc_b = describe_frame(deck_b, self.model, self.ollama_url)

                    if desc_a and desc_a != self._current_prompt_a:
                        self._current_prompt_a = desc_a
                        logger.info(f"[VlmPrompter] Deck A: {desc_a}")

                    if desc_b and desc_b != self._current_prompt_b:
                        self._current_prompt_b = desc_b
                        logger.info(f"[VlmPrompter] Deck B: {desc_b}")

                    update_scope_params(
                        self.scope_url,
                        prompt_a=self._current_prompt_a,
                        prompt_b=self._current_prompt_b,
                    )

            except Exception as e:
                logger.error(f"[VlmPrompter] Error: {e}")

            time.sleep(self.interval)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM Auto-Prompter for AI Transition Mixer")
    parser.add_argument("--scope-url", default="http://localhost:8000", help="Scope API URL")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--model", default="qwen2.5-vl:3b", help="Ollama VLM model name")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between VLM calls")
    parser.add_argument("--split", choices=["side_by_side", "top_bottom"], default="side_by_side")
    parser.add_argument("--spout-source", default="", help="Spout source name (empty = any)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    prompter = VlmPrompter(
        scope_url=args.scope_url,
        ollama_url=args.ollama_url,
        model=args.model,
        interval=args.interval,
        split=args.split,
        spout_source=args.spout_source,
    )
    prompter.start()

    print(f"VLM Prompter running (Ctrl+C to stop)")
    print(f"  Model:    {args.model}")
    print(f"  Interval: {args.interval}s")
    print(f"  Scope:    {args.scope_url}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        prompter.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
