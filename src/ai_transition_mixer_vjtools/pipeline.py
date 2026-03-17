"""
AI Transition Mixer — Scope Graph Preprocessor Node
=====================================================

A pipeline node that sits in a Scope execution graph between two NDI
source nodes and a downstream diffusion pipeline (LongLive, Krea Realtime,
StreamDiffusionV2, etc.).

Graph topology:

    deck_a (NDI) ──video──→ ai_transition_mixer ──video──────────────→ longlive ──→ output
    deck_b (NDI) ──video──→ ai_transition_mixer ──vace_input_frames──→ longlive
                                                  ──vace_input_masks───→ longlive

The mixer:
  1. Receives Deck A frames on the "video" port.
  2. Receives Deck B frames on the "deck_b" port.
  3. Blends them by crossfader position → outputs on "video".
  4. Packs both decks as VACE context → outputs on "vace_input_frames".
  5. Creates appropriate masks → outputs on "vace_input_masks".

The downstream diffusion pipeline (LongLive, Krea, etc.) then uses the
blended video as its V2V input and the VACE context to guide the
creative transition.

This works with ANY VACE-capable pipeline — no coupling to a specific
diffusion backend.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import ClassVar

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Scope SDK imports ──────────────────────────────────────────────────────

try:
    from pydantic import Field
    from scope.core.pipelines.base_schema import (
        BasePipelineConfig,
        ModeDefaults,
        ui_field_config,
    )
    from scope.core.pipelines.interface import Pipeline, Requirements

    _HAS_SCOPE = True
except ImportError:
    _HAS_SCOPE = False

    class Pipeline:
        """Stub for testing outside Scope."""

        pass

    class Requirements:
        def __init__(self, input_size=1):
            self.input_size = input_size


# ─── Config ─────────────────────────────────────────────────────────────────

if _HAS_SCOPE:

    class AiTransitionMixerConfig(BasePipelineConfig):
        # ClassVar metadata
        pipeline_id: ClassVar[str] = "ai_transition_mixer__vj_tools"
        pipeline_name: ClassVar[str] = "AI Transition Mixer"
        pipeline_description: ClassVar[str] = (
            "Dual-input crossfader node: receives two video feeds, "
            "blends by crossfader, outputs video + VACE context for "
            "AI-powered transitions. Chain before LongLive or Krea Realtime."
        )
        pipeline_version: ClassVar[str] = "0.3.0"
        estimated_vram_gb: ClassVar[float] = 0.1  # pure CPU, no model
        supports_vace: ClassVar[bool] = False  # we OUTPUT vace, don't consume it
        modified: ClassVar[bool] = True

        # Graph I/O ports — validated by Scope's graph executor
        inputs: ClassVar[list[str]] = ["video", "deck_b"]
        outputs: ClassVar[list[str]] = ["video", "vace_input_frames", "vace_input_masks"]

        modes: ClassVar[dict] = {
            "video": ModeDefaults(default=True),
        }

        # ── Crossfader ─────────────────────────────────────────────

        crossfader: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Blend between Deck A (0.0) and Deck B (1.0)",
            json_schema_extra=ui_field_config(order=0, label="Crossfader"),
        )

        swap_decks: bool = Field(
            default=False,
            description="Swap Deck A and Deck B assignments",
            json_schema_extra=ui_field_config(order=1, label="Swap A/B"),
        )

        # ── VACE Context ───────────────────────────────────────────

        vace_context_mode: str = Field(
            default="both_decks",
            description=(
                "What to send as VACE context: "
                "'both_decks' = interleave A and B frames, "
                "'blend' = crossfader-weighted blend, "
                "'deck_a' = only Deck A, "
                "'deck_b' = only Deck B"
            ),
            json_schema_extra=ui_field_config(order=2, label="VACE Context Mode"),
        )

        # ── Prompts ────────────────────────────────────────────────

        prompt_a: str = Field(
            default="",
            description="Text prompt describing Deck A scene",
            json_schema_extra=ui_field_config(order=10, label="Deck A Prompt"),
        )

        prompt_b: str = Field(
            default="",
            description="Text prompt describing Deck B scene",
            json_schema_extra=ui_field_config(order=11, label="Deck B Prompt"),
        )

        transition_style: str = Field(
            default="smooth morphing transition",
            description="Visual style for the AI transition",
            json_schema_extra=ui_field_config(order=12, label="Transition Style"),
        )

else:

    class AiTransitionMixerConfig:
        """Standalone config for testing outside Scope."""

        pipeline_id = "ai_transition_mixer__vj_tools"
        inputs = ["video", "deck_b"]
        outputs = ["video", "vace_input_frames", "vace_input_masks"]

        def __init__(self, **kwargs):
            self.crossfader = kwargs.get("crossfader", 0.0)
            self.swap_decks = kwargs.get("swap_decks", False)
            self.vace_context_mode = kwargs.get("vace_context_mode", "both_decks")
            self.prompt_a = kwargs.get("prompt_a", "")
            self.prompt_b = kwargs.get("prompt_b", "")
            self.transition_style = kwargs.get(
                "transition_style", "smooth morphing transition"
            )


# ─── Pipeline Node ──────────────────────────────────────────────────────────


class AiTransitionMixerPipeline(Pipeline):
    """
    Scope graph preprocessor node: dual-input crossfader with VACE output.

    In the graph, this node receives frames on two input ports ("video" for
    Deck A, "deck_b" for Deck B) and outputs blended video + VACE context
    frames + masks for the downstream diffusion pipeline.
    """

    @classmethod
    def get_config_class(cls):
        return AiTransitionMixerConfig

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = AiTransitionMixerConfig()
        self.config = config
        self._frame_count = 0
        logger.info("[TransitionMixer] Initialized (graph preprocessor node v0.3)")

    def prepare(self, **kwargs):
        """Request 1 frame per chunk — we're a lightweight preprocessor."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """
        Process one chunk.

        Args (from graph edges):
            video: list[Tensor]   — Deck A frames (THWC uint8, 0-255)
            deck_b: list[Tensor]  — Deck B frames (THWC uint8, 0-255)

        Args (from session parameters):
            crossfader: float     — 0.0 = Deck A, 1.0 = Deck B
            swap_decks: bool
            vace_context_mode: str
            prompt_a, prompt_b, transition_style: str

        Returns:
            dict with "video", "vace_input_frames", "vace_input_masks" tensors
        """
        # ── Read parameters ──────────────────────────────────────────
        crossfader = float(
            kwargs.get("crossfader", getattr(self.config, "crossfader", 0.0))
        )
        crossfader = max(0.0, min(1.0, crossfader))

        swap = bool(
            kwargs.get("swap_decks", getattr(self.config, "swap_decks", False))
        )
        vace_mode = str(
            kwargs.get(
                "vace_context_mode",
                getattr(self.config, "vace_context_mode", "both_decks"),
            )
        )

        prompt_a = str(kwargs.get("prompt_a", getattr(self.config, "prompt_a", "")))
        prompt_b = str(kwargs.get("prompt_b", getattr(self.config, "prompt_b", "")))
        transition_style = str(
            kwargs.get(
                "transition_style",
                getattr(self.config, "transition_style", "smooth morphing transition"),
            )
        )

        # ── Extract input frames ─────────────────────────────────────
        deck_a_frames = kwargs.get("video", [])
        deck_b_frames = kwargs.get("deck_b", [])

        if not deck_a_frames and not deck_b_frames:
            # No input at all — return empty
            empty = torch.zeros(1, 1, 1, 3)
            return {"video": empty}

        # Convert frame lists to tensors (THWC, 0-255 uint8)
        deck_a = self._frames_to_tensor(deck_a_frames)
        deck_b = self._frames_to_tensor(deck_b_frames)

        if swap:
            deck_a, deck_b = deck_b, deck_a

        # Handle missing deck — use the available one for both
        if deck_a is None and deck_b is not None:
            deck_a = deck_b
        elif deck_b is None and deck_a is not None:
            deck_b = deck_a
        elif deck_a is None and deck_b is None:
            empty = torch.zeros(1, 1, 1, 3)
            return {"video": empty}

        # Ensure same spatial dimensions (resize B to match A)
        deck_b = self._match_size(deck_b, deck_a)

        # ── Blend for video output ───────────────────────────────────
        # Main V2V input: crossfader-weighted blend
        a_weight = 1.0 - crossfader
        b_weight = crossfader
        blended = (a_weight * deck_a.float() + b_weight * deck_b.float()).to(
            deck_a.dtype
        )

        # ── Build VACE context frames ────────────────────────────────
        # The VACE context tells the AI model about both source scenes.
        # Format: list of tensors, each [1, H, W, C] in THWC 0-255 uint8
        if vace_mode == "deck_a":
            vace_ctx = deck_a
        elif vace_mode == "deck_b":
            vace_ctx = deck_b
        elif vace_mode == "blend":
            vace_ctx = blended
        else:
            # "both_decks" — interleave A and B for maximum context
            # AI sees both the source and destination scenes
            n_a = deck_a.shape[0]
            n_b = deck_b.shape[0]
            n_total = n_a + n_b
            vace_ctx = torch.zeros(
                n_total,
                deck_a.shape[1],
                deck_a.shape[2],
                deck_a.shape[3],
                dtype=deck_a.dtype,
            )
            # Interleave: A, B, A, B, ...
            for i in range(max(n_a, n_b)):
                if i < n_a:
                    vace_ctx[i * 2] = deck_a[i]
                if i < n_b and (i * 2 + 1) < n_total:
                    vace_ctx[i * 2 + 1] = deck_b[i]

        # ── Build VACE masks ─────────────────────────────────────────
        # Full-frame masks (ones) = apply VACE guidance to entire frame
        H, W = blended.shape[1], blended.shape[2]
        n_vace = vace_ctx.shape[0]
        vace_masks = torch.ones(n_vace, H, W, 1, dtype=torch.uint8) * 255

        # Convert to frame lists for graph streaming (one tensor per frame)
        vace_frame_list = [vace_ctx[i : i + 1] for i in range(n_vace)]
        mask_frame_list = [vace_masks[i : i + 1] for i in range(n_vace)]
        video_frame_list = [blended[i : i + 1] for i in range(blended.shape[0])]

        # ── Build prompts ────────────────────────────────────────────
        result: dict = {
            "video": blended,  # THWC tensor for output port
            "vace_input_frames": vace_ctx,  # THWC tensor for VACE port
            "vace_input_masks": vace_masks,  # THWC tensor for mask port
        }

        # Generate transition-aware prompts if deck prompts are set
        prompts = self._build_prompts(
            crossfader, prompt_a, prompt_b, transition_style
        )
        if prompts:
            result["prompts"] = prompts

        self._frame_count += 1
        if self._frame_count % 100 == 1:
            logger.info(
                f"[TransitionMixer] fader={crossfader:.2f} "
                f"vace_mode={vace_mode} "
                f"deck_a={deck_a.shape} deck_b={deck_b.shape} "
                f"vace_ctx={vace_ctx.shape}"
            )

        return result

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _frames_to_tensor(frames) -> torch.Tensor | None:
        """Convert a list of frame tensors to a single THWC tensor."""
        if frames is None:
            return None
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 3:
                return frames.unsqueeze(0)
            return frames
        if isinstance(frames, list):
            if len(frames) == 0:
                return None
            return torch.cat(
                [f if f.dim() == 4 else f.unsqueeze(0) for f in frames], dim=0
            )
        return None

    @staticmethod
    def _match_size(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Resize src to match target's spatial dimensions (H, W)."""
        if src.shape[1:3] == target.shape[1:3]:
            return src
        # THWC → TCHW for interpolate, then back
        src_tchw = src.float().permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            src_tchw,
            size=(target.shape[1], target.shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1).to(src.dtype)

    @staticmethod
    def _build_prompts(
        crossfader: float,
        prompt_a: str,
        prompt_b: str,
        transition_style: str,
    ) -> list[str] | None:
        """Generate transition-aware prompt array."""
        if not prompt_a and not prompt_b:
            return None

        if crossfader <= 0.05 and prompt_a:
            return [prompt_a]
        if crossfader >= 0.95 and prompt_b:
            return [prompt_b]
        if not prompt_a:
            return [prompt_b]
        if not prompt_b:
            return [prompt_a]

        # Interpolated prompts for mid-transition
        if crossfader < 0.3:
            return [
                f"{prompt_a}. {transition_style}, with subtle hints of: {prompt_b}"
            ]
        elif crossfader > 0.7:
            return [
                f"{prompt_b}. {transition_style}, with lingering traces of: {prompt_a}"
            ]
        else:
            return [
                f"{transition_style} between {prompt_a} and {prompt_b}"
            ]
