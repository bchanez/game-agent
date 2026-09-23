"""Ego-motion compensation — a generic tool for screen-locked / scrolling games.

Problem (see FINDINGS): when the camera follows the avatar (Mario), the agent's own
motion shows up as GLOBAL background scroll, and the avatar is the *stable* region.
Any change-based signal then measures the scroll, not the controllable movers, so
action understanding goes blind (NOOP looks as active as every action).

Fix, borrowed from image stabilization: estimate the dominant global translation
between two frames (phase correlation, FFT-based — no game knowledge), then
  - keep the shift vector itself  -> locomotion ("right" makes the world slide), and
  - subtract it to get the independent-motion residual -> avatar/enemies moving
    *relative to the world*.

Together they decompose an action's effect into "how it moved my viewpoint" + "what
moved on its own" — the two things raw pixel-diff conflates on a scrolling game.
Generic and optional: the configurator enables it only when global motion is detected.
"""
import numpy as np


def _gray(f):
    a = np.asarray(f, dtype=np.float32)
    return a[..., 0] if a.ndim == 3 else a          # (H,W,1) -> (H,W)


def estimate_shift(prev, cur):
    """Dominant integer (dy, dx) global translation between two frames, by phase
    correlation. Sign convention is internally consistent (used only to align/compare),
    not asserted against any external frame."""
    a, b = _gray(prev), _gray(cur)
    if a.shape != b.shape:
        return 0, 0
    A, B = np.fft.rfft2(a), np.fft.rfft2(b)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-8                            # whiten -> a delta at the shift
    r = np.fft.irfft2(R, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dy), int(dx)


def shift_image(img, dy, dx):
    """Translate a frame by (dy, dx), zero-filling the exposed border."""
    a = _gray(img)
    out = np.zeros_like(a)
    h, w = a.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = a[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    return out


def compensate(prev, cur):
    """Independent-motion residual |cur - align(prev)| after removing the global shift,
    plus the shift vector. The residual isolates what moved relative to the world; the
    shift captures how the viewpoint moved (locomotion)."""
    dy, dx = estimate_shift(prev, cur)
    resid = np.abs(_gray(cur) - shift_image(prev, dy, dx))
    return resid, (dy, dx)
