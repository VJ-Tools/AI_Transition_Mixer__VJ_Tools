"""Quick smoke tests for AI Transition Mixer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [OK] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1

print("=== AI Transition Mixer Tests ===\n")

def test_split_side_by_side():
    from ai_transition_mixer_vjtools.pipeline import AiTransitionMixerPreprocessor, AiTransitionMixerConfig
    config = AiTransitionMixerConfig(crossfader=0.0, context_frames=4)
    mixer = AiTransitionMixerPreprocessor(config)

    # Create side-by-side frame: left=red, right=blue
    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    frame[:, :64, 0] = 255  # left = red
    frame[:, 64:, 2] = 255  # right = blue

    video = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    result = mixer(video=video)

    assert "video" in result
    assert "vace_input_frames" in result
    assert result["vace_enabled"] == True
    # At crossfader=0, output should be mostly red (deck A)
    out = (result["video"][0].numpy() * 255).astype(np.uint8)
    assert out[:, :, 0].mean() > 200, f"Expected red, got mean R={out[:,:,0].mean()}"

test("Split side-by-side (fader=0 -> Deck A)", test_split_side_by_side)


def test_crossfader_full_b():
    from ai_transition_mixer_vjtools.pipeline import AiTransitionMixerPreprocessor, AiTransitionMixerConfig
    config = AiTransitionMixerConfig(crossfader=1.0, context_frames=4)
    mixer = AiTransitionMixerPreprocessor(config)

    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    frame[:, :64, 0] = 255  # left = red
    frame[:, 64:, 2] = 255  # right = blue

    video = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    result = mixer(video=video)

    out = (result["video"][0].numpy() * 255).astype(np.uint8)
    assert out[:, :, 2].mean() > 200, f"Expected blue, got mean B={out[:,:,2].mean()}"

test("Crossfader=1.0 -> Deck B", test_crossfader_full_b)


def test_crossfader_midpoint():
    from ai_transition_mixer_vjtools.pipeline import AiTransitionMixerPreprocessor, AiTransitionMixerConfig
    config = AiTransitionMixerConfig(crossfader=0.5, context_frames=4)
    mixer = AiTransitionMixerPreprocessor(config)

    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    frame[:, :64, 0] = 255  # left = red
    frame[:, 64:, 2] = 255  # right = blue

    video = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    result = mixer(video=video)

    out = (result["video"][0].numpy() * 255).astype(np.uint8)
    # At midpoint, should be purple-ish (both R and B present)
    assert 100 < out[:, :, 0].mean() < 180, f"Expected ~128 red, got {out[:,:,0].mean()}"
    assert 100 < out[:, :, 2].mean() < 180, f"Expected ~128 blue, got {out[:,:,2].mean()}"

test("Crossfader=0.5 -> 50/50 blend", test_crossfader_midpoint)


def test_vace_frames_are_list():
    from ai_transition_mixer_vjtools.pipeline import AiTransitionMixerPreprocessor, AiTransitionMixerConfig
    config = AiTransitionMixerConfig(crossfader=0.3, context_frames=6)
    mixer = AiTransitionMixerPreprocessor(config)

    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    frame[:, :64, 0] = 255
    frame[:, 64:, 2] = 255

    # Feed multiple frames to build history
    for _ in range(8):
        video = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
        result = mixer(video=video)

    vace = result["vace_input_frames"]
    assert isinstance(vace, list), f"Expected list, got {type(vace)}"
    assert len(vace) == 6, f"Expected 6 context frames, got {len(vace)}"
    # Each should be [C, H, W] tensor
    assert vace[0].dim() == 3, f"Expected 3D tensor, got {vace[0].dim()}D"

test("VACE frames returned as list (triggers auto-resize)", test_vace_frames_are_list)


def test_top_bottom_split():
    from ai_transition_mixer_vjtools.pipeline import AiTransitionMixerPreprocessor, AiTransitionMixerConfig
    config = AiTransitionMixerConfig(crossfader=0.0, split_mode="top_bottom", context_frames=2)
    mixer = AiTransitionMixerPreprocessor(config)

    frame = np.zeros((128, 64, 3), dtype=np.uint8)
    frame[:64, :, 1] = 255  # top = green
    frame[64:, :, 0] = 255  # bottom = red

    video = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    result = mixer(video=video)

    out = (result["video"][0].numpy() * 255).astype(np.uint8)
    assert out[:, :, 1].mean() > 200, f"Expected green (top), got mean G={out[:,:,1].mean()}"

test("Top/bottom split mode", test_top_bottom_split)


def test_swap_decks():
    from ai_transition_mixer_vjtools.pipeline import AiTransitionMixerPreprocessor, AiTransitionMixerConfig
    config = AiTransitionMixerConfig(crossfader=0.0, swap_decks=True, context_frames=2)
    mixer = AiTransitionMixerPreprocessor(config)

    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    frame[:, :64, 0] = 255  # left = red (normally A)
    frame[:, 64:, 2] = 255  # right = blue (normally B)

    video = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
    result = mixer(video=video)

    # With swap, fader=0 should give blue (B is now on the left after swap)
    out = (result["video"][0].numpy() * 255).astype(np.uint8)
    assert out[:, :, 2].mean() > 200, f"Expected blue (swapped), got mean B={out[:,:,2].mean()}"

test("Swap decks", test_swap_decks)


print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
