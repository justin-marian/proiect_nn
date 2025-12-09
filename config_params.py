from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoggingConfig:
    app: str = "UnbiasedTeacher"
    level: str = "INFO"
    log_dir: str = "logs"
    serialize: bool = False
    rich_tracebacks: bool = True
    use_tqdm_sink: bool = True


@dataclass
class Metrics:
    """
    Detection metrics configuration.
    """

    num_classes: int  # number of foreground classes
    iou_thresholds: tuple[float, ...] = (
        0.5, 0.55, 0.6, 0.65, 0.7,
        0.75, 0.8, 0.85, 0.9, 0.95
    )
    score_thresh: float = 0.05
    class_agnostic: bool = False


@dataclass
class TrainerConfig:
    """
    Trainer configuration for Unbiased Teacher style SSOD.
    """

    device: str = "cuda"
    epochs: int = 180
    log_interval: int = 20
    max_grad_norm: float | None = None

    eval_interval: int = 1
    checkpoint_interval: int = 1

    use_amp: bool = True


@dataclass
class ValidatorConfig:
    """
    Validator configuration.
    """

    device: str = "cuda"


@dataclass
class DataConfig:
    """
    Dataset configuration for semi-supervised detection.
    """

    # dataset type
    dataset: str = "coco"  # "coco" or "voc"

    # root directory for datasets
    root: str = "datasets"

    # COCO paths
    coco_train_labeled_ann: str = "coco/annotations/instances_train2017_10pct.json"
    coco_train_unlabeled_ann: str = "coco/annotations/instances_train2017_unlabeled.json"
    coco_val_ann: str = "coco/annotations/instances_val2017.json"
    coco_train_img_dir: str = "coco/train2017"
    coco_val_img_dir: str = "coco/val2017"

    # VOC paths (optional)
    voc_root: str = "VOC"
    voc_years: list[str] = field(default_factory=lambda: ["2007", "2012"])

    # image size
    img_size: int = 640

    # dataloader
    batch_size: int = 8
    unsup_batch_ratio: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = True

    # number of classes (COCO = 80, VOC = 20)
    num_classes: int = 80

    # labeled percentage for COCO
    labeled_percent: float = 0.10

    # strong augmentation on student branch
    use_strong_aug: bool = True


@dataclass
class ModelConfig:
    """
    Model configuration.
    """

    arch: str = "csp"        # "csp", "repopt", "hybrid"
    version: str = "large"   # "nano", "medium", "large"

    num_classes: int = 80    # number of foreground classes


@dataclass
class OptimConfig:
    """
    Optimizer configuration.
    """

    opt: str = "sgd"                 # "sgd", "adamw", "adam", ...
    lr: float = 0.01
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    nesterov: bool = True

    head_lr_mult: float = 1.0


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.
    """

    scheme: str = "cosine"           # "cosine", "multistep", "onecycle", "lambda"
    warmup_epochs: int = 3
    warmup_bias_lr: float = 0.1
    min_lr_ratio: float = 0.05
    milestones: list[int] = field(default_factory=lambda: [60, 80])
    gamma: float = 0.1


@dataclass
class MetricsConfig(Metrics):
    """
    Default detection metrics for COCO style evaluation.
    """

    num_classes: int = 80
    iou_thresholds: tuple[float, ...] = (
        0.5, 0.55, 0.6, 0.65, 0.7,
        0.75, 0.8, 0.85, 0.9, 0.95
    )
    score_thresh: float = 0.05
    class_agnostic: bool = False


@dataclass
class SSLConfig:
    """
    Semi-supervised learning configuration for Unbiased Teacher.
    """

    # supervised loss weight (labeled data)
    sup_weight: float = 1.0

    # unsupervised loss weight (unlabeled data)
    unsup_weight: float = 4.0
    use_teacher: bool = True
    ema_decay: float = 0.9996

    # pseudo label threshold
    pseudo_conf_thresh: float = 0.7

    # schedule for unsupervised loss
    burnup_epochs: int = 1
    unsup_start_epoch: int = 1

    # matching configuration
    match_iou_thresh: float = 0.5
    max_pairs_per_image: int = 128

    # box consistency loss
    box_consistency: str = "smooth_l1"

    # augmentation behavior
    strong_aug_on_student: bool = True


@dataclass
class ExperimentConfig:
    """
    Top-level experiment configuration.
    """

    seed: int = 42
    device: str = "cuda"

    logging: LoggingConfig = field(default_factory=LoggingConfig)

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)

    ssl: SSLConfig = field(default_factory=SSLConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
