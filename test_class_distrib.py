import inspect
import numpy as np
from data.visualize.class_distrib import plot_class_distribution

# ---- sanity: confirm you imported the updated function (has max_xticks, rotate) ----
print("plot_class_distribution signature:", inspect.signature(plot_class_distribution))


# ---------------------------
# Test 1: numeric labels (basic)
# ---------------------------
rng = np.random.default_rng(0)

num_classes = 6
y_train = rng.choice(num_classes, size=600, p=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03])
y_val   = rng.choice(num_classes, size=200, p=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05])

class_names = ["car", "person", "bike", "bus", "truck", "other"]

fig, axes = plot_class_distribution(
    y_train=y_train,
    y_val=y_val,
    class_names=class_names,
    show=True,
)
print("Test 1 OK: numeric labels.")


# ---------------------------
# Test 2: string labels (basic)
# ---------------------------
y_train_str = np.array(rng.choice(class_names, size=500, p=[0.45, 0.25, 0.10, 0.08, 0.07, 0.05]), dtype=object)
y_val_str   = np.array(rng.choice(class_names, size=180, p=[0.35, 0.25, 0.12, 0.10, 0.10, 0.08]), dtype=object)

fig, axes = plot_class_distribution(
    y_train=y_train_str,
    y_val=y_val_str,
    class_names=class_names,
    show=True,
)
print("Test 2 OK: string labels.")


# ---------------------------
# Test 3A: many classes (default behavior)
# ---------------------------
C = 80
y_train_many = rng.integers(0, C, size=5000)
y_val_many   = rng.integers(0, C, size=1500)
many_names = [f"cls_{i}" for i in range(C)]

fig, axes = plot_class_distribution(
    y_train=y_train_many,
    y_val=y_val_many,
    class_names=many_names,
    show=True,
)
print("Test 3A OK: many classes (default max_xticks=20, rotate=60).")


# ---------------------------
# Test 3B: many classes (VISIBLY different ticks)
#   - fewer ticks: max_xticks=8
#   - no rotation: rotate=0
# ---------------------------
fig, axes = plot_class_distribution(
    y_train=y_train_many,
    y_val=y_val_many,
    class_names=many_names,
    max_xticks=8,
    rotate=0,
    show=True,
)
print("Test 3B OK: many classes (max_xticks=8, rotate=0).")


# ---------------------------
# Test 3C: many classes (show ALL ticks)
#   - max_xticks >= num_classes -> all tick labels
# ---------------------------
fig, axes = plot_class_distribution(
    y_train=y_train_many,
    y_val=y_val_many,
    class_names=many_names,
    max_xticks=200,
    rotate=90,
    show=True,
)
print("Test 3C OK: many classes (max_xticks=200 => all ticks, rotate=90).")


# ---------------------------
# Test 4: error case - string labels but class_names=None
# ---------------------------
try:
    _ = plot_class_distribution(
        y_train=np.array(["car", "person", "car"], dtype=object),
        y_val=np.array(["car", "bike"], dtype=object),
        class_names=None,
        show=False,
    )
    print("Test 4 FAILED (should have raised).")
except ValueError as e:
    print("Test 4 OK (expected error):", e)


# ---------------------------
# Test 5: error case - string label not found in class_names
# ---------------------------
try:
    _ = plot_class_distribution(
        y_train=np.array(["car", "person", "alien"], dtype=object),
        y_val=np.array(["car", "bike"], dtype=object),
        class_names=class_names,
        show=False,
    )
    print("Test 5 FAILED (should have raised).")
except ValueError as e:
    print("Test 5 OK (expected error):", e)
