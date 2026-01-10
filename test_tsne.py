import os
import numpy as np
import matplotlib.pyplot as plt

# change this import to your real module path
from data.visualize.tsne import (
    tsne_embeddings,
    plot_tsne_labels,
    plot_tsne_transf,
)


def expect_raises(fn, exc_type, msg_contains: str):
    try:
        fn()
        raise AssertionError("Expected exception was NOT raised.")
    except exc_type as e:
        s = str(e)
        assert msg_contains in s, f"Expected '{msg_contains}' in error, got: {s}"


def make_features(rng, n, d, centers, std=1.0):
    """
    Build D-dim clustered features (not 2D).
    Returns: X (n,d), y (n,)
    """
    k = len(centers)
    y = rng.integers(0, k, size=n, endpoint=False)
    X = np.zeros((n, d), dtype=np.float32)
    for i in range(k):
        m = y == i
        if m.any():
            X[m] = rng.normal(loc=centers[i], scale=std, size=(m.sum(), d)).astype(np.float32)
    return X, y


def test_tsne_embeddings_shape_and_seed():
    rng = np.random.default_rng(0)
    n, d = 300, 64
    centers = rng.normal(size=(5, d)).astype(np.float32)

    X, _ = make_features(rng, n, d, centers, std=0.7)

    Z1 = tsne_embeddings(X, perplexity=30.0, max_iter=500, seed=123)
    Z2 = tsne_embeddings(X, perplexity=30.0, max_iter=500, seed=123)

    assert isinstance(Z1, np.ndarray)
    assert Z1.shape == (n, 2)

    # same seed should be deterministic (or extremely close)
    assert np.allclose(Z1, Z2, atol=1e-6), "Expected deterministic t-SNE for same seed."

    # sanity checks: finite + not collapsed
    assert np.isfinite(Z1).all(), "t-SNE output contains NaN/Inf."

    # ensure it didn't collapse to (almost) a point
    std = Z1.std(axis=0)
    assert (std > 1e-6).all(), f"t-SNE embedding collapsed (std={std})."



def test_tsne_embeddings_perplexity_error():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 16)).astype(np.float32)

    # your function enforces perplexity < N/3
    expect_raises(
        lambda: tsne_embeddings(X, perplexity=30.0, max_iter=250, seed=0),  # 30 >= 60/3 = 20
        ValueError,
        "perplexity < num_samples / 3",
    )


def test_plots_from_real_tsne(show_plots: bool):
    rng = np.random.default_rng(2)
    class_names = ["car", "person", "bike", "bus", "truck", "other"]
    num_classes = len(class_names)

    n_train, n_test, d = 500, 200, 64
    centers = rng.normal(size=(num_classes, d)).astype(np.float32)

    X_train, y_train = make_features(rng, n_train, d, centers, std=0.75)
    X_test, y_test = make_features(rng, n_test, d, centers, std=0.95)

    # IMPORTANT: run t-SNE once on concatenated data, then split
    X_all = np.concatenate([X_train, X_test], axis=0)
    Z_all = tsne_embeddings(X_all, perplexity=30.0, max_iter=800, seed=42)

    Z_train = Z_all[:n_train]
    Z_test = Z_all[n_train:]

    # 1) plot_tsne_labels
    fig1, ax1 = plot_tsne_labels(
        Z_train, Z_test, y_train, y_test,
        class_names, num_classes=None,
        show=show_plots,
        save_path="tsne_labels_current.png",
    )
    assert fig1 is not None and ax1 is not None
    assert os.path.exists("tsne_labels_current.png")
    os.remove("tsne_labels_current.png")
    if not show_plots:
        plt.close(fig1)

    # 2) plot_tsne_transf
    fig2, axes2 = plot_tsne_transf(
        Z_train, Z_test, y_train, y_test,
        class_names, num_classes=None,
        show=show_plots,
        save_path="tsne_transf_current.png",
    )
    assert fig2 is not None
    assert len(np.atleast_1d(axes2)) == 2
    assert os.path.exists("tsne_transf_current.png")
    os.remove("tsne_transf_current.png")
    if not show_plots:
        plt.close(fig2)


def test_plot_shape_errors():
    rng = np.random.default_rng(3)
    class_names = ["a", "b", "c"]

    # bad embedding shape (N,3)
    train_2d = rng.normal(size=(100, 3))
    test_2d = rng.normal(size=(50, 2))
    train_y = rng.integers(0, 3, size=100)
    test_y = rng.integers(0, 3, size=50)

    expect_raises(
        lambda: plot_tsne_labels(train_2d, test_2d, train_y, test_y, class_names, show=False),
        ValueError,
        "incompatible shapes",
    )

    # label length mismatch
    train_2d = rng.normal(size=(100, 2))
    train_y = rng.integers(0, 3, size=99)

    expect_raises(
        lambda: plot_tsne_transf(train_2d, test_2d, train_y, test_y, class_names, show=False),
        ValueError,
        "incompatible shapes",
    )


def run_all(show_plots: bool = True):
    print("Running current t-SNE tests...")

    test_tsne_embeddings_shape_and_seed()
    print("test_tsne_embeddings_shape_and_seed: OK")

    test_tsne_embeddings_perplexity_error()
    print("test_tsne_embeddings_perplexity_error: OK")

    test_plots_from_real_tsne(show_plots)
    print("test_plots_from_real_tsne: OK")

    test_plot_shape_errors()
    print("test_plot_shape_errors: OK")

    print("All tests passed ✅")


if __name__ == "__main__":
    # Set False for CI/headless
    run_all(show_plots=True)
