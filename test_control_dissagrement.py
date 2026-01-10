import numpy as np

from data.visualize.agreement import (
    agreement_matrix,
    plot_agreement_heatmap,
    plot_agreement_ema_vs_kdd,
    plot_cross_arch_agreement,
)

# ---------- 1) basic setup ----------
class_names = ["car", "person", "bike", "bus"]
C = len(class_names)
N = 400
rng = np.random.default_rng(7)

# ---------- 2) helper to generate controlled disagreements ----------
def gen_teacher_student(
    N: int, C: int, *,
    teacher_prior=None,
    p_correct: float = 0.75,
    confusion: np.ndarray | None = None,
    seed: int = 0,
):
    r = np.random.default_rng(seed)

    if teacher_prior is None:
        teacher_prior = np.ones(C) / C
    teacher = r.choice(C, size=N, p=np.asarray(teacher_prior))
    student = teacher.copy()

    # which samples are wrong
    wrong = r.random(N) > p_correct
    idx = np.where(wrong)[0]

    if confusion is None:
        # uniform "wrong" class among other classes
        for i in idx:
            choices = [k for k in range(C) if k != teacher[i]]
            student[i] = r.choice(choices)
    else:
        # confusion[t] is a probability distribution over student classes given teacher class t
        for i in idx:
            t = int(teacher[i])
            student[i] = r.choice(C, p=confusion[t])

    return teacher.astype(int), student.astype(int)

# ---------- 3) Scenario A: EMA (pretty aligned) ----------
teacher_ema, student_ema = gen_teacher_student(
    N, C, p_correct=0.85, seed=1
)

# ---------- 4) Scenario B: KDD (more systematic confusion) ----------
# Example: if teacher says "car", student often says "bus"
#          if teacher says "person", student often says "bike"
conf_kdd = np.array([
    [0.00, 0.10, 0.05, 0.85],  # teacher car -> student mostly bus
    [0.05, 0.00, 0.85, 0.10],  # teacher person -> student mostly bike
    [0.10, 0.80, 0.00, 0.10],  # teacher bike -> student mostly person
    [0.85, 0.05, 0.05, 0.00],  # teacher bus -> student mostly car
], dtype=float)

# Make rows valid probabilities (safety)
conf_kdd = conf_kdd / conf_kdd.sum(axis=1, keepdims=True)

teacher_kdd, student_kdd = gen_teacher_student(
    N, C, p_correct=0.55, confusion=conf_kdd, seed=2
)

# ---------- 5) Test: agreement_matrix directly ----------
cm_ema = agreement_matrix(teacher_ema, student_ema, C, normalize=True)
cm_kdd = agreement_matrix(teacher_kdd, student_kdd, C, normalize=True)

print("EMA matrix row sums (should be ~1.0):", cm_ema.sum(axis=1))
print("KDD matrix row sums (should be ~1.0):", cm_kdd.sum(axis=1))

# You should see stronger diagonal on EMA, and strong off-diagonal patterns on KDD.

# ---------- 6) Test: plot_agreement_heatmap ----------
plot_agreement_heatmap(
    cm_ema, class_names, title="Agreement (EMA) - normalized"
)

plot_agreement_heatmap(
    cm_kdd, class_names, title="Agreement (KDD) - normalized", cmap="Greens"
)

# ---------- 7) Test: plot_agreement_ema_vs_kdd ----------
plot_agreement_ema_vs_kdd(
    teacher_ema, student_ema,
    teacher_kdd, student_kdd,
    class_names,
    arch_name="ResNet50"
)

# ---------- 8) Test: plot_cross_arch_agreement ----------
# Create multiple "student architectures" with different behaviors vs a fixed teacher set.
teacher_fixed = rng.choice(C, size=N, p=[0.35, 0.35, 0.20, 0.10]).astype(int)

# A "strong" student: mostly correct
student_rn18 = teacher_fixed.copy()
mask = rng.random(N) > 0.88
student_rn18[mask] = rng.integers(0, C, size=mask.sum())

# A "medium" student: some confusion
student_vit = teacher_fixed.copy()
mask = rng.random(N) > 0.72
student_vit[mask] = (teacher_fixed[mask] + 1) % C  # systematic shift

# A "weak" student: biased to predict class 1 ("person")
student_mobilenet = teacher_fixed.copy()
mask = rng.random(N) > 0.55
student_mobilenet[mask] = 1

agreement_by_arch = {
    "RN18": agreement_matrix(teacher_fixed, student_rn18, C, normalize=True),
    "ViT-Tiny": agreement_matrix(teacher_fixed, student_vit, C, normalize=True),
    "MobileNet": agreement_matrix(teacher_fixed, student_mobilenet, C, normalize=True),
}

plot_cross_arch_agreement(
    agreement_by_arch, class_names, teacher_arch="Teacher(EMA)"
)

# ---------- 9) Edge-case quick checks ----------
# (a) handles invalid labels by skipping them (per your if condition)
teacher_bad = np.array([0, 1, 2, 3, 99, -1, 1, 2])
student_bad = np.array([0, 1, 2, 2,  1,  1, 7,  2])
cm_bad = agreement_matrix(teacher_bad, student_bad, C, normalize=True)
print("With invalid labels, still row-normalized:", cm_bad.sum(axis=1))
plot_agreement_heatmap(cm_bad, class_names, title="Edge case: invalid labels skipped")
