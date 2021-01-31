

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import torch
import os
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

class AUTOVOSDataset(torch.utils.data.Dataset):
    def __init__(self, annt_file, transforms=None):
        # as you would do normally
        # annotation directory 
        self.annotation_file = annt_file
        # read the annotations 
        self.imageName = None
        with open(self.annotation_file,'r') as fp:
            self.annt_labels = json.load(fp)
        
        self.coco = COCO(annt_file)
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        # Creating a contiguos id mapping
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        
        self._transforms = transforms

        
    def _getannotationfields(self, annotations):
        # this function return bbox , segmentation mask and labels
        bboxes = []
        masks =[]
        classes  = []
        for i in range(len(annotations)):
            bboxes.append(annotations[i]["bbox"])
            masks.append(annotations[i]["segmentation"])
            classes.append(annotations[i]["category_id"])
        
        return bboxes,masks,classes
    
    def __getitem__(self, idx): 
        annotations = [ann for ann in self.annt_labels['annotations'] if ann['image_id']==idx]
        imageName = self.annt_labels["images"][idx]["file_name"]
        bboxes, masks, classes = self._getannotationfields(annotations)        
        # opening the image 
        img  = Image.open(imageName).convert("RGB")
        
        bboxes = torch.as_tensor(bboxes).reshape(-1,4)
        target = BoxList(bboxes,img.size,mode='xywh').convert('xyxy')
        
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
        return self.coco.imgs[idx+1]
    
    def __len__(self):
        return len(self.coco.getImgIds())


# In[89]:


#MyDataset("datasets/auto_vos/annotations/vid1.json").__getitem__(1)

