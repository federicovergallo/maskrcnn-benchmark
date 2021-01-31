import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor_synthia import COCODemo

config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x_synthia.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)

with open("../inference/last_checkpoint", "r") as f:
    ckpt = "../"+f.readlines()[0]

cfg.MODEL.WEIGHT = ckpt

# Now we create the `COCODemo` object. It contains a few extra options for conveniency, such as the confidence
# threshold for detections to be shown.
coco_demo = COCODemo(
    cfg,
    min_image_size=600,
    confidence_threshold=0.1,
    show_mask_heatmaps=False,
    weight_loading=ckpt,
)

# Let's define a few helper functions for loading images from a URL
def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


# Let's now load an image from the COCO dataset. It's reference is in the comment
image_1 = cv2.imread('000010.png')
image_2 = cv2.imread('000011.png')


# ### Computing the predictions
# 
# We provide a `run_on_opencv_image` function, which takes an image as it was loaded by OpenCV (in `BGR` format),
# and computes the predictions on them, returning an image with the predictions overlayed on the image.

# compute predictions
predictions = coco_demo.run_on_opencv_image([image_1, image_2])
imshow(predictions)


