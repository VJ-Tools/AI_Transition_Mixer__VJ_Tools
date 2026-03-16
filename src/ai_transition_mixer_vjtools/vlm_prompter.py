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


TRANSITION_SYSTEM_PROMPT = """\
You are a prompt engineer for Wan2.1, a real-time AI video generation model.

You write sequential prompt arrays for smooth video transitions between two scenes. Each array has exactly 6 prompts — one per video chunk (~2 seconds each, ~12 seconds total).

RULES:
1. Output ONLY valid JSON: {"prompts": ["...", "...", "...", "...", "...", "..."]}
2. Every prompt must restate ALL key visual details (characters, clothing, setting, lighting, camera) to prevent drift between chunks.
3. Each subsequent prompt adds ONE incremental change that moves from Scene A toward Scene B.
4. The transition style describes HOW the scenes merge (e.g. "dissolve through smoke", "morph through water", "glitch transition").
5. Maintain consistent camera angle, aspect ratio, and style language across all 6 prompts.
6. Keep each prompt between 40-120 words. Be vivid and specific.
7. Do NOT include any explanation, commentary, or thinking. ONLY the JSON object.

/no_think"""


def generate_transition_prompts(
    prompt_a: str,
    prompt_b: str,
    crossfader: float,
    transition_style: str = "smooth morphing",
    num_prompts: int = 6,
    model: str = "qwen/qwen3.5-9b",
    lmstudio_url: str = "http://localhost:1234",
) -> list[str]:
    """
    Generate a sequential prompt array for a Wan2.1 video transition.

    Returns a list of prompts, one per chunk, that smoothly transitions
    from Scene A to Scene B using the specified transition style.
    The crossfader position determines WHERE in the transition we are,
    so the prompts are biased accordingly.
    """
    import urllib.request
    import re

    # At the extremes, just repeat the source prompt
    if crossfader <= 0.05:
        return [prompt_a] * num_prompts
    if crossfader >= 0.95:
        return [prompt_b] * num_prompts

    pct_b = int(crossfader * 100)
    pct_a = 100 - pct_b

    user_msg = (
        f"Generate a {num_prompts}-prompt transition array.\n\n"
        f"SCENE A (starting point):\n{prompt_a}\n\n"
        f"SCENE B (ending point):\n{prompt_b}\n\n"
        f"Transition style: {transition_style}\n"
        f"Current blend: {pct_a}% Scene A, {pct_b}% Scene B\n\n"
        f"The transition should be weighted toward the current blend point. "
        f"If blend is 70% A / 30% B, the first 4 prompts should be mostly A "
        f"with B elements gradually appearing, and only the last 2 should show "
        f"significant B presence.\n\n"
        f'Output ONLY: {{"prompts": ["...", "...", "...", "...", "...", "..."]}}'
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSITION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 2000,
        "temperature": 0.5,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "transition_prompts",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "prompts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": num_prompts,
                            "maxItems": num_prompts,
                        }
                    },
                    "required": ["prompts"],
                    "additionalProperties": False,
                },
            },
        },
    }

    req = urllib.request.Request(
        f"{lmstudio_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    content = ""
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()

            # Strip <think> tags (shouldn't appear with /no_think but just in case)
            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```" in content:
                content = re.sub(r"```json?\s*", "", content)
                content = re.sub(r"```", "", content)
                content = content.strip()

            parsed = json.loads(content)
            prompts = parsed.get("prompts", [])

            if isinstance(prompts, list) and len(prompts) >= 2:
                logger.info(f"[Prompter] Generated {len(prompts)} prompts successfully")
                return prompts[:num_prompts]
            else:
                logger.warning(f"[Prompter] Got {len(prompts)} prompts, expected {num_prompts}")
                return [prompt_a] * num_prompts

    except json.JSONDecodeError as e:
        logger.warning(f"[Prompter] Failed to parse JSON: {e}\nContent: {content[:200]}")
        return [prompt_a] * num_prompts
    except Exception as e:
        logger.warning(f"[Prompter] LM Studio request failed: {e}")
        return [prompt_a] * num_prompts


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

class TransitionPrompter:
    """
    Periodically generates creative transition prompts by blending
    deck descriptions via a local LLM (LM Studio).

    Deck descriptions are set manually or by an external VLM.
    This class handles the creative blending and pushes to Scope.
    """

    def __init__(
        self,
        scope_url: str = "http://localhost:8000",
        lmstudio_url: str = "http://localhost:1234",
        model: str = "qwen/qwen3.5-9b",
        interval: float = 2.0,
        transition_style: str = "smooth morphing",
    ):
        self.scope_url = scope_url
        self.lmstudio_url = lmstudio_url
        self.model = model
        self.interval = interval
        self.transition_style = transition_style

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Set these manually or via external VLM
        self.prompt_a: str = ""
        self.prompt_b: str = ""
        self.crossfader: float = 0.0  # Polled from Scope or set externally

        self._last_generated: list[str] = []
        self.num_prompts: int = 6

    def start(self):
        """Start the background prompter loop."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            f"[TransitionPrompter] Started (model={self.model}, "
            f"interval={self.interval}s, scope={self.scope_url})"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_crossfader(self):
        """Try to read the current crossfader value from Scope."""
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{self.scope_url}/api/v1/pipeline/parameters",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "crossfader" in data:
                    self.crossfader = float(data["crossfader"])
        except Exception:
            pass  # Use last known value

    def _loop(self):
        while self._running:
            try:
                if not self.prompt_a and not self.prompt_b:
                    time.sleep(self.interval)
                    continue

                self._poll_crossfader()

                # Only regenerate if both prompts are set and fader is in the blend zone
                if self.prompt_a and self.prompt_b and 0.05 < self.crossfader < 0.95:
                    prompts = generate_transition_prompts(
                        self.prompt_a,
                        self.prompt_b,
                        self.crossfader,
                        self.transition_style,
                        self.num_prompts,
                        self.model,
                        self.lmstudio_url,
                    )

                    if prompts and prompts != self._last_generated:
                        self._last_generated = prompts
                        for i, p in enumerate(prompts):
                            logger.info(f"[TransitionPrompter] [{i+1}/{len(prompts)}] {p[:80]}...")
                        # Push the prompt array to Scope
                        _push_prompts(self.scope_url, prompts)

            except Exception as e:
                logger.error(f"[TransitionPrompter] Error: {e}")

            time.sleep(self.interval)


def _push_prompts(scope_url: str, prompts: list[str]) -> bool:
    """Push a prompt array to Scope's pipeline (one prompt per chunk)."""
    import urllib.request
    payload = json.dumps({"prompts": prompts}).encode("utf-8")
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
        logger.warning(f"Failed to push prompts: {e}")
        return False


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transition Prompter for AI Transition Mixer")
    parser.add_argument("--scope-url", default="http://localhost:8000", help="Scope API URL")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234", help="LM Studio API URL")
    parser.add_argument("--model", default="qwen/qwen3.5-9b", help="LM Studio model name")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between LLM calls")
    parser.add_argument("--style", default="smooth morphing", help="Transition style description")
    parser.add_argument("--prompt-a", required=True, help="Description of Deck A content")
    parser.add_argument("--prompt-b", required=True, help="Description of Deck B content")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    prompter = TransitionPrompter(
        scope_url=args.scope_url,
        lmstudio_url=args.lmstudio_url,
        model=args.model,
        interval=args.interval,
        transition_style=args.style,
    )
    prompter.prompt_a = args.prompt_a
    prompter.prompt_b = args.prompt_b
    prompter.start()

    print(f"Transition Prompter running (Ctrl+C to stop)")
    print(f"  Model:    {args.model}")
    print(f"  Interval: {args.interval}s")
    print(f"  Scope:    {args.scope_url}")
    print(f"  Deck A:   {args.prompt_a}")
    print(f"  Deck B:   {args.prompt_b}")
    print(f"  Style:    {args.style}")
    print(f"\nPolling crossfader from Scope and generating blended prompts...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        prompter.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
