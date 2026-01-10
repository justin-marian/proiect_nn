import numpy as np

# change this import to your real module path
# from data.visualize.confusion_matrix import (
#     normalize_rows, plot_heatmap, format_acc_series, wrap_text,
#     plot_confusion_matrix, plot_confusion_matrices_side_by_side
# )

from data.visualize.confusion_matrix import (
    normalize_rows, format_acc_series, wrap_text,
    plot_confusion_matrix, plot_confusion_matrices_side_by_side
)


def _make_confusion_counts(rng: np.random.Generator, c: int, n: int = 5000) -> np.ndarray:
    """Create a synthetic confusion matrix with diagonal dominance (counts)."""
    true = rng.integers(0, c, size=n)
    pred = true.copy()
    flip = rng.random(n) < 0.20
    pred[flip] = rng.integers(0, c, size=flip.sum())
    cm = np.zeros((c, c), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


# ---------------------------
# Test 1: normalize_rows basics
# ---------------------------
def test_normalize_rows_sums_to_one():
    x = np.array([[1, 1, 2],
                  [0, 0, 0],
                  [5, 0, 5]], dtype=np.float32)
    y = normalize_rows(x)
    assert y.shape == x.shape
    assert np.allclose(y[0].sum(), 1.0)
    assert np.allclose(y[2].sum(), 1.0)
    # zero-row should remain all zeros (sum=0)
    assert np.allclose(y[1], 0.0)


# ---------------------------
# Test 2: format_acc_series (float)
# ---------------------------
def test_format_acc_series_float():
    s = format_acc_series(0.123456, decimals=4)
    assert s == "0.1235"


# ---------------------------
# Test 3: format_acc_series (sequence + truncation)
# ---------------------------
def test_format_acc_series_seq_trunc():
    s = format_acc_series([0.1, 0.2, 0.3, 0.4], decimals=2, max_items=2)
    assert s == "0.10, 0.20, ..."


# ---------------------------
# Test 4: wrap_text
# ---------------------------
def test_wrap_text_basic():
    s = "a" * 25
    w = wrap_text(s, 10)
    assert "\n" in w
    assert w.split("\n")[0] == "a" * 10


# ---------------------------
# Test 5: plot_confusion_matrix (counts, normalize=False)
# ---------------------------
def test_plot_confusion_matrix_counts():
    rng = np.random.default_rng(0)
    c = 6
    cm = _make_confusion_counts(rng, c, n=2000)
    class_names = ["car", "person", "bike", "bus", "truck", "other"]

    fig, ax = plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title="Counts CM",
        normalize=False,
        show=True
    )

    # sanity checks
    assert fig is not None and ax is not None
    assert ax.get_xlabel() == "Predicted"
    assert ax.get_ylabel() == "True"


# ---------------------------
# Test 6: plot_confusion_matrix (counts -> normalize=True)
# ---------------------------
def test_plot_confusion_matrix_normalized():
    rng = np.random.default_rng(1)
    c = 5
    cm = _make_confusion_counts(rng, c, n=1500)
    class_names = [f"cls_{i}" for i in range(c)]

    fig, ax = plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title="Normalized CM",
        normalize=True,
        show=True
    )

    assert fig is not None and ax is not None


# ---------------------------
# Test 7: side-by-side (shared_scale=True, with acc floats)
# ---------------------------
def test_plot_side_by_side_shared_scale():
    rng = np.random.default_rng(2)
    c = 7
    cm_a = _make_confusion_counts(rng, c, n=3000)
    cm_b = _make_confusion_counts(rng, c, n=3000)
    class_names = [f"cls_{i}" for i in range(c)]

    fig, axes = plot_confusion_matrices_side_by_side(
        cm_left=cm_a,
        cm_right=cm_b,
        class_names=class_names,
        left_title="Model A",
        right_title="Model B",
        left_acc=0.91,
        right_acc=0.93,
        shared_scale=True,
        show=True
    )

    assert fig is not None
    assert len(axes) == 2
    assert axes[0].get_xlabel() == "Predicted"
    assert axes[1].get_ylabel() == "True"


# ---------------------------
# Test 8: side-by-side (shared_scale=False, with acc sequence + truncation)
# ---------------------------
def test_plot_side_by_side_no_shared_scale_acc_seq():
    rng = np.random.default_rng(3)
    c = 8
    cm_a = _make_confusion_counts(rng, c, n=5000)
    cm_b = _make_confusion_counts(rng, c, n=1200)  # different counts scale
    class_names = [f"cls_{i}" for i in range(c)]

    per_class_acc_a = rng.random(c)
    per_class_acc_b = rng.random(c)

    fig, axes = plot_confusion_matrices_side_by_side(
        cm_left=cm_a,
        cm_right=cm_b,
        class_names=class_names,
        left_title="Model A (per-class acc)",
        right_title="Model B (per-class acc)",
        left_acc=per_class_acc_a,
        right_acc=per_class_acc_b,
        shared_scale=False,
        acc_max_items=3,
        show=True
    )

    assert fig is not None
    assert len(axes) == 2


# ---------------------------
# Test 9: zero-row robustness (a class with zero true samples)
# ---------------------------
def test_zero_row_confusion_matrix():
    c = 5
    cm = np.zeros((c, c), dtype=np.int64)
    cm[0, 0] = 10
    cm[1, 2] = 5
    # rows 2,3,4 are all zeros -> should not crash in normalize_rows
    class_names = [f"cls_{i}" for i in range(c)]

    fig, axes = plot_confusion_matrices_side_by_side(
        cm_left=cm,
        cm_right=cm,
        class_names=class_names,
        left_acc=None,
        right_acc=None,
        show=True
    )
    assert fig is not None
    assert len(axes) == 2


if __name__ == "__main__":
    # If running as a script (not pytest), run tests manually:
    test_normalize_rows_sums_to_one()
    test_format_acc_series_float()
    test_format_acc_series_seq_trunc()
    test_wrap_text_basic()
    test_plot_confusion_matrix_counts()
    test_plot_confusion_matrix_normalized()
    test_plot_side_by_side_shared_scale()
    test_plot_side_by_side_no_shared_scale_acc_seq()
    test_zero_row_confusion_matrix()
    print("All tests passed ✅")
