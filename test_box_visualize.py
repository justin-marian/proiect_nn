import numpy as np
import torch

from data.datasets.config import ClassInfo
from bbox.box_ops import BoxList

from data.visualize.draw_bbox import (
    to_numpy_boxes,
    get_info,
    legend_handles,
    draw_bbox,
    bbox_visualize,
)


def _dummy_image(h=240, w=320) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _make_boxlist(boxes_xyxy: np.ndarray) -> BoxList:
    boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
    labels_t = torch.zeros((boxes_t.shape[0],), dtype=torch.long)
    scores_t = torch.ones((boxes_t.shape[0],), dtype=torch.float32)
    return BoxList(boxes=boxes_t, labels=labels_t, scores=scores_t, image_size=(480, 640))


def _classes() -> dict[int, ClassInfo]:
    return {
        0: ClassInfo(name="car", color="#00FF00"),
        1: ClassInfo(name="person", color="#FF0000"),
        2: ClassInfo(name="bike", color="#0000FF"),
    }


def _expect_raises(fn, exc_type, msg_contains: str):
    try:
        fn()
        raise AssertionError("Expected exception was NOT raised.")
    except exc_type as e:
        s = str(e)
        assert msg_contains in s, f"Expected message to contain '{msg_contains}', got '{s}'"


def test_to_numpy_boxes_ok():
    boxes = np.array([[10, 15, 100, 120], [50, 60, 70, 80]], dtype=np.float32)
    bl = _make_boxlist(boxes)
    out = to_numpy_boxes(bl)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 4)
    assert out.dtype == np.float32
    assert np.allclose(out, boxes)


def test_to_numpy_boxes_empty():
    boxes = np.zeros((0, 4), dtype=np.float32)
    bl = _make_boxlist(boxes)
    out = to_numpy_boxes(bl)

    assert out.shape == (0, 4)
    assert out.dtype == np.float32


def test_get_info_default():
    info = get_info(_classes(), 999)
    assert info.name == "999"
    assert info.color == "#FFFFFF"


def test_legend_handles_used_ids():
    handles = legend_handles(_classes(), {2, 0})
    assert len(handles) == 2
    labels = [h.get_label() for h in handles]
    assert labels == ["car", "bike"]  # sorted: 0 then 2


def test_draw_bbox_label_mismatch_raises():
    img = _dummy_image()
    boxes = np.array([[10, 10, 50, 50], [20, 20, 80, 80]], dtype=np.float32)
    bl = _make_boxlist(boxes)
    labels = [0]  # mismatch

    _expect_raises(
        lambda: draw_bbox(img, bl, labels, None, _classes()),
        ValueError,
        "Labels length must match",
    )


def test_draw_bbox_score_mismatch_raises():
    img = _dummy_image()
    boxes = np.array([[10, 10, 50, 50], [20, 20, 80, 80]], dtype=np.float32)
    bl = _make_boxlist(boxes)
    labels = [0, 1]
    scores = [0.9]  # mismatch

    _expect_raises(
        lambda: draw_bbox(img, bl, labels, scores, _classes()),
        ValueError,
        "Scores length must match",
    )


def test_draw_bbox_conf_threshold_filters_boxes(show_plot: bool):
    img = _dummy_image()
    boxes = np.array(
        [
            [10, 10, 60, 60],     # keep
            [100, 80, 150, 130],  # filtered
            [30, 40, 90, 100],    # keep
        ],
        dtype=np.float32,
    )
    bl = _make_boxlist(boxes)
    labels = [0, 1, 2]
    scores = [0.9, 0.1, 0.8]

    fig, ax, used = draw_bbox(img, bl, labels, scores, _classes(), conf_thr=0.5)

    assert used == {0, 2}

    rects = [p for p in ax.patches if p.__class__.__name__ == "Rectangle"]
    assert len(rects) == 2

    if show_plot:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_bbox_visualize_multi_image(show_plot: bool):
    classes = _classes()

    imgs = [_dummy_image(200, 280), _dummy_image(200, 280), _dummy_image(200, 280)]
    boxes_list = [
        _make_boxlist(np.array([[10, 10, 60, 60]], dtype=np.float32)),
        _make_boxlist(np.array([[20, 30, 120, 140]], dtype=np.float32)),
        _make_boxlist(np.array([[50, 50, 100, 120]], dtype=np.float32)),
    ]
    labels = [[0], [1], [2]]
    scores = [[0.9], [0.95], [0.85]]

    fig, axes = bbox_visualize(imgs, boxes_list, labels, scores, classes, titles=["A", "B", "C"], cols=2)

    assert fig is not None
    assert axes.size >= 3

    if show_plot:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)


def run_all(show_plots: bool = True):
    tests = [
        ("to_numpy_boxes_ok", test_to_numpy_boxes_ok),
        ("to_numpy_boxes_empty", test_to_numpy_boxes_empty),
        ("get_info_default", test_get_info_default),
        ("legend_handles_used_ids", test_legend_handles_used_ids),
        ("draw_bbox_label_mismatch_raises", test_draw_bbox_label_mismatch_raises),
        ("draw_bbox_score_mismatch_raises", test_draw_bbox_score_mismatch_raises),
    ]

    for name, fn in tests:
        try:
            fn()
            print(f"{name}: OK")
        except Exception as e:
            print(f"{name}: FAILED -> {e}")
            raise

    # tests with plots
    try:
        test_draw_bbox_conf_threshold_filters_boxes(show_plots)
        print("draw_bbox_conf_threshold_filters_boxes: OK")
    except Exception as e:
        print(f"draw_bbox_conf_threshold_filters_boxes: FAILED -> {e}")
        raise

    try:
        test_bbox_visualize_multi_image(show_plots)
        print("bbox_visualize_multi_image: OK")
    except Exception as e:
        print(f"bbox_visualize_multi_image: FAILED -> {e}")
        raise

    print("All tests passed ✅")


if __name__ == "__main__":
    # Set to False if you do NOT want windows popping up
    run_all(show_plots=True)
