"""Object-centric perception for grids — Phase 1 of the generic agent.

Turns a raw color grid into a *set of objects* (contiguous same-color regions) with
per-object features (color, size, position, shape). This is the transferable
"understanding" layer: an agent that sees entities + relations, not a bag of cells,
generalizes across unseen games far better than one reading raw cells (see ROADMAP /
FINDINGS — weight transfer is fragile, representation transfer is not).

Connected-components labeling is a GENERIC structural transform — zero game knowledge
(no sprite templates, no "this blob is an enemy"), exactly like grayscale/resize in
the pixel pipeline. That keeps the *no human indication* principle: objecthood is a
universal property of a grid, not a hand-coded per-game percept. The *meaning* of each
object is left for the net + RL to learn. (A learned slot-attention encoder is the
heavier upgrade; this is the cheap, interpretable first version.)
"""
import numpy as np
from scipy import ndimage

N_COLORS = 16
# per-object feature layout: 16 color one-hot + [size, cy, cx, h, w, fill]
FEAT_DIM = N_COLORS + 6


def _structure(connectivity):
    if connectivity == 8:
        return np.ones((3, 3), dtype=int)
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)   # 4-connectivity


def background_color(grid):
    """The most frequent color — treated as empty space, so we don't emit one giant
    'object' for the backdrop. Generic (majority vote), not a per-game constant."""
    return int(np.bincount(grid.ravel(), minlength=N_COLORS).argmax())


def extract_objects(grid, background=None, connectivity=4):
    """Every maximal contiguous same-color region (except the background color) as an
    object with generic geometric features. Returns a list of dicts sorted by size."""
    grid = np.asarray(grid)
    if background is None:
        background = background_color(grid)
    h, w = grid.shape
    struct = _structure(connectivity)
    objects = []
    for color in range(N_COLORS):
        if color == background:
            continue
        mask = grid == color
        if not mask.any():
            continue
        labels, n = ndimage.label(mask, structure=struct)
        for lab in range(1, n + 1):
            ys, xs = np.nonzero(labels == lab)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            bh, bw = (y1 - y0 + 1), (x1 - x0 + 1)
            size = len(ys)
            objects.append({
                "color": color, "size": size,
                "cy": float(ys.mean()), "cx": float(xs.mean()),
                "y0": int(y0), "x0": int(x0), "y1": int(y1), "x1": int(x1),
                "h": int(bh), "w": int(bw),
                "fill": size / float(bh * bw),
            })
    objects.sort(key=lambda o: -o["size"])
    return objects, background


def object_features(grid, max_objects=32, background=None):
    """Fixed-size (max_objects, FEAT_DIM) float array + a validity mask, for a
    permutation-invariant encoder. Objects beyond max_objects (the smallest) are
    dropped — `n_truncated` reports how many, so silent capping never hides coverage."""
    grid = np.asarray(grid)
    h, w = grid.shape
    objs, _ = extract_objects(grid, background)
    feats = np.zeros((max_objects, FEAT_DIM), dtype=np.float32)
    mask = np.zeros(max_objects, dtype=bool)
    for i, o in enumerate(objs[:max_objects]):
        feats[i, o["color"]] = 1.0                              # color one-hot
        feats[i, N_COLORS:] = [
            o["size"] / (h * w), o["cy"] / h, o["cx"] / w,
            o["h"] / h, o["w"] / w, o["fill"],
        ]
        mask[i] = True
    return feats, mask, max(0, len(objs) - max_objects)


# a fixed 16-color palette for *display only* (never fed to the agent)
_PALETTE = np.array([
    [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64], [255, 220, 0],
    [170, 170, 170], [240, 18, 190], [255, 133, 27], [127, 219, 255], [135, 12, 37],
    [1, 255, 112], [1, 68, 33], [133, 20, 75], [255, 255, 255], [85, 85, 85], [88, 44, 0],
], dtype=np.uint8)


def render_objects(grid, out_path, max_objects=32):
    """Save the grid (colorized) with each discovered object's bounding box + centroid
    — the interpretable proof that perception found entities with no labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    grid = np.asarray(grid)
    objs, bg = extract_objects(grid)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(_PALETTE[grid])
    for o in objs[:max_objects]:
        ax.add_patch(mpatches.Rectangle(
            (o["x0"] - 0.5, o["y0"] - 0.5), o["w"], o["h"],
            fill=False, edgecolor="white", linewidth=1.0))
        ax.plot(o["cx"], o["cy"], "w+", markersize=5)
    ax.set_title(f"{len(objs)} objects found (bg color={bg})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return len(objs), bg
