from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class VocClass:
    """For image visualization purposes when see the detected boxes with colors/labels."""
    name: str       # e.g. "aeroplane"
    color: str      # e.g. "#FF6B6B" (hex color code)

# Mapping from class index to VocClass info
VOC_CLASSES: Mapping[int, VocClass] = {
    0: VocClass("aeroplane", "#FF6B6B"),
    1: VocClass("bicycle", "#4ECDC4"),
    2: VocClass("bird", "#FFD166"),
    3: VocClass("boat", "#118AB2"),
    4: VocClass("bottle", "#073B4C"),
    5: VocClass("bus", "#EF476F"),
    6: VocClass("car", "#06D6A0"),
    7: VocClass("cat", "#7209B7"),
    8: VocClass("chair", "#F8961E"),
    9: VocClass("cow", "#83C5BE"),
    10: VocClass("diningtable", "#E29578"),
    11: VocClass("dog", "#9B5DE5"),
    12: VocClass("horse", "#00BBF9"),
    13: VocClass("motorbike", "#00F5D4"),
    14: VocClass("person", "#FF99C8"),
    15: VocClass("pottedplant", "#A7C957"),
    16: VocClass("sheep", "#577590"),
    17: VocClass("sofa", "#F9844A"),
    18: VocClass("train", "#277DA1"),
    19: VocClass("tvmonitor", "#43AA8B"),
}

#* List of URIs to download VOC dataset tar files from if not present locally
#* VOC dataset 2007 and 2012 trainval + 2007 test, general used for practical purposes
#* 2 files total (2007 and 2012) arround ~1.65GB download + ~3.5GB extracted size each trainval/test set
VOC_KAGGLE_DATASETS: dict[str, str] = {
    "2007": "zaraks/pascal-voc-2007",  # Voctest_06-Nov-2007 and Voctrainval_06-Nov-2007
    "2012": "gopalbhattrai/pascal-voc-2012-dataset"  # Voc2012_test and Voc2012_train_val
}
