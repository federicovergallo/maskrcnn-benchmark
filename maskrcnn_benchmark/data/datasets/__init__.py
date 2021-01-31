# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .abstract import AbstractDataset
from .cityscapes import CityScapesDataset
from .autovos import AUTOVOSDataset
from .synthia import SYNTHIADataset

__all__ = [
    "SYNTHIADataset",
    "AUTOVOSDataset",
    "COCODataset",
    "ConcatDataset",
    "PascalVOCDataset",
    "AbstractDataset",
    "CityScapesDataset",
]
