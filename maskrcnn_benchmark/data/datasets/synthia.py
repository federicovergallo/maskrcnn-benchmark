from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import torch
import json
import os
from PIL import Image
from pycocotools.coco import COCO
from .abstract import AbstractDataset
from .coco import COCODataset
import numpy as np

class_names = ['background','person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class SYNTHIADataset(COCODataset):
    def __init__(self, annt_file, img_dir, transforms=None, categories=None):
        # as you would do normally
        # annotation directory
        if categories is None:
            categories = class_names
        self.annotation_file = annt_file
        # image directory
        self.img_dir = img_dir
        # read the annotations
        self.imageName = None
        with open(self.annotation_file, 'r') as fp:
            self.annt_labels = json.load(fp)

        self.coco = COCO(annt_file)
        #self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        #self.categories = dict(sorted(self.categories.items()))
        self.ids =  list(sorted(self.coco.imgs.keys()))
        self.categories = categories
        # Creating a contiguos id mapping
        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(np.arange(len(class_names)))
            # enumerate(sorted(self.coco.getCatIds()))
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self._transforms = transforms


    def _getannotationfields(self, annotations):
        # this function return bbox , segmentation mask and labels
        bboxes = []
        masks = []
        classes = []
        for i in range(len(annotations)):
            bboxes.append(annotations[i]["bbox"])
            masks.append(annotations[i]["segmentation"])
            classes.append(annotations[i]["category_id"])

        return bboxes, masks, classes

    def __getitem__(self, idx):
        annotations = [ann for ann in self.annt_labels['annotations'] if ann['image_id'] == idx]
        imageName = os.path.join(self.img_dir, self.annt_labels["images"][idx]["file_name"])
        bboxes, masks, classes = self._getannotationfields(annotations)
        # opening the image
        img = Image.open(imageName).convert("RGB")

        bboxes = torch.as_tensor(bboxes).reshape(-1, 4)
        #bboxes = torch.index_select(bboxes, 1, torch.LongTensor([1,0,3,2]))
        target = BoxList(bboxes, img.size, mode='xywh').convert('xyxy')

        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, idx):
        # get img_height and img_width
        return self.coco.imgs[idx]

    def __len__(self):
        return len(self.coco.getImgIds())


