import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format

# All weights name
weights_name_all = ['rpn.head.conv.bias', 'rpn.head.conv.weight', 'rpn.head.bbox_pred.bias',
                    'rpn.head.bbox_pred.weight', 'rpn.head.cls_logits.bias', 'rpn.head.cls_logits.weight',
                    'mask_fcn1.bias', 'mask_fcn1.weight', 'mask_fcn2.bias', 'mask_fcn2.weight',
                    'mask_fcn3.bias', 'mask_fcn3.weight', 'mask_fcn4.bias', 'mask_fcn4.weight',
                    'conv5_mask.bias', 'conv5_mask.weight', 'mask_fcn_logits.bias', 'mask_fcn_logits.weight',
                    'fc6.bias', 'fc6.weight', 'fc7.bias', 'fc7.weight', 'bbox_pred.bias', 'bbox_pred.weight',
                    'cls_score.bias', 'cls_score.weight']


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


def random_init(d, listofkeys):
    for key in listofkeys:
        ten_shape = d[key].shape
        if 'bias' in key:
            d[key] = torch.nn.init.zeros_(torch.Tensor(ten_shape))
        else:
            d[key] = torch.nn.init.xavier_uniform_(torch.Tensor(ten_shape))
    return d


def random_init_classification_layers(d, listofkeys, num_classes):
    for key in listofkeys:
        ten_shape = d[key].shape
        if 'bbox_pred' in key:
            first_dim = num_classes*4
        else:
            first_dim = num_classes
        if 'bias' in key:
            d[key] = torch.nn.init.zeros_(torch.Tensor(first_dim))
        else:
            d[key] = torch.nn.init.xavier_uniform_(torch.Tensor(first_dim, *ten_shape[1:]))
    return d


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    #default="~/.torch/models/_detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14"
    #        ".DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    default="catalog://Caffe2Detectron/COCO/35857345/e2e_faster_rcnn_R-50-FPN_1x",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="./pretrained_model/mask_rcnn_R-50-FPN_1x_detectron_no_last_layers_caffe.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_mask_rcnn_R_50_FPN_1x_synthia.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

if 'NUM_CLASSES' in cfg['MODEL']['ROI_BOX_HEAD']:
    num_classes = cfg['MODEL']['ROI_BOX_HEAD']['NUM_CLASSES']
    keys = ['cls_score.weight', 'cls_score.bias',
            'bbox_pred.weight', 'bbox_pred.bias',
            'mask_fcn_logits.weight', 'mask_fcn_logits.bias']
    newdict['model'] = random_init_classification_layers(_d['model'], keys, num_classes)


#newdict['model'] = random_init(_d['model'],
#                             ['rpn.head.conv.bias', 'rpn.head.conv.weight', 'rpn.head.bbox_pred.bias',
#                              'rpn.head.bbox_pred.weight', 'rpn.head.cls_logits.bias', 'rpn.head.cls_logits.weight'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))
