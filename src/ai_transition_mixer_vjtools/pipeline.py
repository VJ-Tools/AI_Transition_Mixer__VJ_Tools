"""
AI Transition Mixer — Scope Preprocessor (VACE)
================================================

Receives a side-by-side video frame from Resolume (Deck A | Deck B),
splits it, and packs VACE context frames weighted by the crossfader.

The VACE context tells the AI diffusion model what the output should
look like. By blending Deck A and Deck B frames in the VACE context,
the AI generates creative transitions between the two sources.

Setup:
  1. Resolume: Create a composition with Deck A on the left half
     and Deck B on the right half. Output via Spout.
  2. Scope: Select this preprocessor. Pick up the Spout source.
     Enable VACE on the main pipeline.
  3. Move the crossfader (MIDI-mappable) to transition between decks.

The crossfader controls the VACE context blend:
  0.0 = VACE context is 100% Deck A (AI stays close to A)
  0.5 = 50/50 blend (AI has maximum creative freedom)
  1.0 = VACE context is 100% Deck B (AI stays close to B)

VACE frames are passed as a list — Scope's PreprocessVideoBlock
auto-resizes and resamples frame count to match the pipeline chunk.
No masks needed = no frame count mismatch problem.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Split modes ────────────────────────────────────────────────────────────

class SplitMode(str, Enum):
    """How the two decks are arranged in the input frame."""
    SIDE_BY_SIDE = "side_by_side"  # A on left, B on right
    TOP_BOTTOM = "top_bottom"      # A on top, B on bottom


# ─── Scope SDK imports ──────────────────────────────────────────────────────

try:
    from pydantic import Field
    from scope.core.pipelines.base_schema import (
        BasePipelineConfig,
        ModeDefaults,
        ui_field_config,
    )
    from scope.core.pipelines.interface import Pipeline
    _HAS_SCOPE = True
except ImportError:
    _HAS_SCOPE = False

    class Pipeline:
        """Stub for testing outside Scope."""
        pass


# ─── Config ─────────────────────────────────────────────────────────────────

if _HAS_SCOPE:

    class AiTransitionMixerConfig(BasePipelineConfig):
        # ClassVar metadata — matches Scope's pipeline convention
        pipeline_id = "ai_transition_mixer__vj_tools"
        pipeline_name = "AI Transition Mixer"
        pipeline_description = "Split side-by-side Resolume decks and crossfade through AI via VACE"
        pipeline_version = "0.2.0"
        estimated_vram_gb = 0.5
        supports_vace = True
        modified = True

        # Declare inputs/outputs
        inputs = ["video", "vace_input_frames"]
        outputs = ["video"]

        # Mode support — video mode uses lower resolution for speed
        modes = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(),
        }

        # ── Performance Controls ────────────────────────────────────

        crossfader: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Blend between Deck A (0.0) and Deck B (1.0)",
            json_schema_extra=ui_field_config(
                order=0,
                label="Crossfader",
            ),
        )

        split_mode: SplitMode = Field(
            default=SplitMode.SIDE_BY_SIDE,
            description="How the two decks are arranged in the input frame",
            json_schema_extra=ui_field_config(
                order=1,
                label="Split Mode",
            ),
        )

        swap_decks: bool = Field(
            default=False,
            description="Swap Deck A and Deck B sides",
            json_schema_extra=ui_field_config(
                order=2,
                label="Swap A/B",
            ),
        )

        context_frames: int = Field(
            default=8,
            ge=1,
            le=24,
            description="Number of VACE context frames (more = smoother, heavier)",
            json_schema_extra=ui_field_config(
                order=3,
                label="Context Frames",
                is_load_param=True,
            ),
        )

        vace_context_scale: float = Field(
            default=0.8,
            ge=0.0,
            le=1.0,
            description="How strongly the AI follows the VACE context (0=ignore, 1=strict)",
            json_schema_extra=ui_field_config(
                order=4,
                label="Context Strength",
            ),
        )

        # ── Prompt Controls ─────────────────────────────────────────

        prompt_a: str = Field(
            default="",
            description="Text prompt for Deck A scene",
            json_schema_extra=ui_field_config(
                order=10,
                label="Deck A Prompt",
            ),
        )

        prompt_b: str = Field(
            default="",
            description="Text prompt for Deck B scene",
            json_schema_extra=ui_field_config(
                order=11,
                label="Deck B Prompt",
            ),
        )

        transition_style: str = Field(
            default="smooth morphing transition",
            description="Visual style for the AI transition between decks",
            json_schema_extra=ui_field_config(
                order=12,
                label="Transition Style",
            ),
        )

        auto_prompt: bool = Field(
            default=True,
            description="Auto-generate sequential prompt arrays from deck prompts",
            json_schema_extra=ui_field_config(
                order=13,
                label="Auto Prompt",
            ),
        )

else:

    class AiTransitionMixerConfig:
        """Standalone config for testing outside Scope."""
        def __init__(self, **kwargs):
            self.pipeline_id = kwargs.get("pipeline_id", "ai_transition_mixer__vj_tools")
            self.crossfader = kwargs.get("crossfader", 0.0)
            self.split_mode = kwargs.get("split_mode", "side_by_side")
            self.prompt_a = kwargs.get("prompt_a", "")
            self.prompt_b = kwargs.get("prompt_b", "")
            self.transition_style = kwargs.get("transition_style", "smooth morphing transition")
            self.auto_prompt = kwargs.get("auto_prompt", True)
            self.swap_decks = kwargs.get("swap_decks", False)
            self.context_frames = kwargs.get("context_frames", 8)
            self.vace_context_scale = kwargs.get("vace_context_scale", 0.8)


# ─── Preprocessor ───────────────────────────────────────────────────────────

class AiTransitionMixerPreprocessor(Pipeline):
    """
    Scope preprocessor: splits side-by-side input into two decks,
    generates VACE context frames blended by the crossfader, and
    passes them to the main V2V pipeline for AI-powered transitions.
    """

    @classmethod
    def get_config_class(cls):
        return AiTransitionMixerConfig

    def __init__(self, config=None, device=None, dtype=torch.float16, **kwargs):
        if config is None:
            config = AiTransitionMixerConfig()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if (device or self.device).type == "cuda" else torch.float32
        self._frame_count = 0

        # Accumulate recent frames from each deck for temporal context
        self._deck_a_history: list[np.ndarray] = []
        self._deck_b_history: list[np.ndarray] = []
        self._max_history = 24  # keep last N frames per deck

        logger.info("[TransitionMixer] Initialized (VACE mode)")

    def prepare(self, **kwargs):
        """Accept one frame at a time from Scope."""
        if _HAS_SCOPE:
            return Requirements(input_size=1)
        return None

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video", [])

        if isinstance(video, list) and len(video) == 0:
            return {"video": torch.zeros(1, 1, 1, 3)}

        # ── Read params ─────────────────────────────────────────────
        crossfader = float(kwargs.get("crossfader", getattr(self.config, "crossfader", 0.0)))
        crossfader = max(0.0, min(1.0, crossfader))

        split_str = str(kwargs.get("split_mode", getattr(self.config, "split_mode", "side_by_side")))
        try:
            split_mode = SplitMode(split_str)
        except ValueError:
            split_mode = SplitMode.SIDE_BY_SIDE

        swap = bool(kwargs.get("swap_decks", getattr(self.config, "swap_decks", False)))
        num_ctx = int(kwargs.get("context_frames", getattr(self.config, "context_frames", 8)))
        ctx_scale = float(kwargs.get("vace_context_scale", getattr(self.config, "vace_context_scale", 0.8)))

        prompt_a = str(kwargs.get("prompt_a", getattr(self.config, "prompt_a", "")))
        prompt_b = str(kwargs.get("prompt_b", getattr(self.config, "prompt_b", "")))
        transition_style = str(kwargs.get("transition_style", getattr(self.config, "transition_style", "smooth morphing transition")))
        auto_prompt = bool(kwargs.get("auto_prompt", getattr(self.config, "auto_prompt", True)))

        # ── Decode input tensor ─────────────────────────────────────
        if isinstance(video, list):
            frames = torch.cat(video, dim=0).float()
        else:
            frames = video.float() if video.dim() == 4 else video.unsqueeze(0).float()

        if frames.max() <= 1.0:
            frames = frames * 255.0

        frame_np = frames[-1].cpu().numpy().astype(np.uint8)
        H, W, C = frame_np.shape

        # ── Split into Deck A and Deck B ────────────────────────────
        if split_mode == SplitMode.SIDE_BY_SIDE:
            mid = W // 2
            deck_a = frame_np[:, :mid, :]
            deck_b = frame_np[:, mid:, :]
        else:  # TOP_BOTTOM
            mid = H // 2
            deck_a = frame_np[:mid, :, :]
            deck_b = frame_np[mid:, :, :]

        if swap:
            deck_a, deck_b = deck_b, deck_a

        # Store in history
        self._deck_a_history.append(deck_a)
        self._deck_b_history.append(deck_b)
        if len(self._deck_a_history) > self._max_history:
            self._deck_a_history.pop(0)
        if len(self._deck_b_history) > self._max_history:
            self._deck_b_history.pop(0)

        # ── Build VACE context frames ───────────────────────────────
        # Blend deck A and B histories based on crossfader position.
        # Each VACE context frame is a weighted mix of A and B.
        #
        # When crossfader = 0.0: all context frames are Deck A
        # When crossfader = 0.5: context frames are 50/50 blend
        # When crossfader = 1.0: all context frames are Deck B

        avail_a = len(self._deck_a_history)
        avail_b = len(self._deck_b_history)
        n_frames = min(num_ctx, avail_a, avail_b)
        if n_frames < 1:
            n_frames = 1

        vace_frames = []
        for i in range(n_frames):
            # Sample from the most recent history
            idx = max(0, avail_a - n_frames + i)
            a_frame = self._deck_a_history[min(idx, avail_a - 1)].astype(np.float32)

            idx_b = max(0, avail_b - n_frames + i)
            b_frame = self._deck_b_history[min(idx_b, avail_b - 1)].astype(np.float32)

            # Ensure same size
            if a_frame.shape != b_frame.shape:
                # Resize B to match A
                from PIL import Image
                img_b = Image.fromarray(b_frame.astype(np.uint8))
                img_b = img_b.resize((a_frame.shape[1], a_frame.shape[0]), Image.BILINEAR)
                b_frame = np.array(img_b).astype(np.float32)

            # Weighted blend
            blended = ((1.0 - crossfader) * a_frame + crossfader * b_frame).astype(np.uint8)

            # Convert to tensor [H, W, C] → [C, H, W] float 0-1
            t = torch.from_numpy(blended).float().permute(2, 0, 1) / 255.0
            vace_frames.append(t)

        # Stack into [B, C, F, H, W] — Scope's expected VACE format
        # Each frame is [C, H, W], stack on dim=1 after unsqueezing
        vace_tensor = torch.stack(vace_frames, dim=1).unsqueeze(0)  # [1, C, F, H, W]

        # ── Also pass the blended frame as the video output ─────────
        # The main pipeline sees this as its V2V input
        t_fader = crossfader
        latest_a = self._deck_a_history[-1].astype(np.float32)
        latest_b = self._deck_b_history[-1].astype(np.float32)
        if latest_a.shape != latest_b.shape:
            from PIL import Image
            img_b = Image.fromarray(latest_b.astype(np.uint8))
            img_b = img_b.resize((latest_a.shape[1], latest_a.shape[0]), Image.BILINEAR)
            latest_b = np.array(img_b).astype(np.float32)

        video_out = ((1.0 - t_fader) * latest_a + t_fader * latest_b).astype(np.uint8)
        video_tensor = torch.from_numpy(video_out).float().unsqueeze(0) / 255.0

        self._frame_count += 1
        if self._frame_count % 100 == 1:
            logger.info(
                f"[TransitionMixer] fader={crossfader:.2f} "
                f"vace_frames={n_frames} ctx_scale={ctx_scale:.2f} "
                f"deck_sizes=({deck_a.shape}, {deck_b.shape})"
            )

        # ── Build prompt array ─────────────────────────────────────────
        # Generate a 6-prompt sequential array for Wan2.1 chunk prompts.
        # This is the fallback when the external VLM prompter isn't running.
        # The external prompter generates much better prompts via Qwen 3.5.
        generated_prompts = None
        if auto_prompt and (prompt_a or prompt_b):
            if crossfader <= 0.05 and prompt_a:
                generated_prompts = [prompt_a]
            elif crossfader >= 0.95 and prompt_b:
                generated_prompts = [prompt_b]
            elif prompt_a and prompt_b:
                # Simple 6-step interpolation as fallback
                # (the external VLM prompter does this much better)
                steps = 6
                generated_prompts = []
                for i in range(steps):
                    t = i / (steps - 1)  # 0.0 to 1.0
                    # Bias by crossfader position
                    effective_t = crossfader * t + (1 - crossfader) * (t * 0.3)
                    if effective_t < 0.2:
                        generated_prompts.append(prompt_a)
                    elif effective_t > 0.8:
                        generated_prompts.append(prompt_b)
                    elif effective_t < 0.4:
                        generated_prompts.append(
                            f"{prompt_a}. {transition_style}, with subtle hints of: {prompt_b}"
                        )
                    elif effective_t > 0.6:
                        generated_prompts.append(
                            f"{prompt_b}. {transition_style}, with lingering traces of: {prompt_a}"
                        )
                    else:
                        generated_prompts.append(
                            f"{transition_style} between {prompt_a} and {prompt_b}"
                        )
            elif prompt_a:
                generated_prompts = [prompt_a]
            elif prompt_b:
                generated_prompts = [prompt_b]

        # ── Return VACE context + video + params ────────────────────
        # Pass vace_input_frames as a LIST so Scope's PreprocessVideoBlock
        # auto-resizes and resamples to the correct chunk frame count.
        # No masks = no frame count mismatch.
        result = {
            "video": video_tensor,
            "vace_input_frames": vace_frames,  # LIST triggers auto-resize
            "vace_enabled": True,
            "vace_context_scale": ctx_scale,
        }

        if generated_prompts:
            result["prompts"] = generated_prompts

        return result
