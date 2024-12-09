# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import os
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import datasets.transforms as T

class KAISTDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(KAISTDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(KAISTDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class DualModalitiesKAISTDetection(torchvision.datasets.CocoDetection):
    def __init__(self, ir_img_folder, ir_ann_file, rgb_img_folder, rgb_ann_file, transforms, return_masks):
        super(DualModalitiesKAISTDetection, self).__init__(ir_img_folder, ir_ann_file)
        self.ir = KAISTDetection(ir_img_folder, ir_ann_file, transforms=transforms, return_masks=return_masks)
        self.rgb = KAISTDetection(rgb_img_folder, rgb_ann_file, transforms=transforms, return_masks=return_masks)

    def __getitem__(self, idx):
        ir_img, target = self.ir[idx]
        rgb_img, _ = self.rgb[idx]
        return ir_img, rgb_img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)


        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_kaist_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_kaist(image_set, args):
    if args.modality == 'both' or args.modality == 'mixed':
        
        root = os.path.join('../datasets/kaist_multispectral/')
        data_path = os.path.join(root)

        assert os.path.exists(root), f'provided DATA path {root} does not exist'

        ## For the kaist the validation is split over the json file that is why the images come from train set
        PATHS = {
            "train": (data_path, data_path, os.path.join(root, 'kaist_rgb_train.json'), os.path.join(root, 'kaist_ir_train.json')),
            "val"  : (data_path, data_path, os.path.join(root, 'kaist_rgb_valid.json'), os.path.join(root, 'kaist_ir_valid.json')),
            "test" : (data_path, data_path, os.path.join(root, 'kaist_rgb_test.json') , os.path.join(root, 'kaist_ir_test.json')),
        }

        ir_img_folder, rgb_img_folder, rgb_ann_file, ir_ann_file = PATHS[image_set]

        dataset = DualModalitiesKAISTDetection(ir_img_folder, ir_ann_file, rgb_img_folder, rgb_ann_file, transforms=make_kaist_transforms(image_set), return_masks=args.masks)
        return dataset
    
    else :
        root = os.path.join('../datasets/kaist_multispectral/')
        data_path = os.path.join(root)

        assert os.path.exists(root), f'provided DATA path {root} does not exist'

        ## For the kaist the validation is split over the json file that is why the images come from train set
        PATHS = {
            "train": (data_path, os.path.join(root, 'kaist_' + args.modality + '_train.json')),
            "val": (data_path, os.path.join(root, 'kaist_' + args.modality + '_valid.json')),
            "test": (data_path, os.path.join(root, 'kaist_' + args.modality + '_test.json')),
        }

        img_folder, ann_file = PATHS[image_set]

        dataset = KAISTDetection(img_folder, ann_file, transforms=make_kaist_transforms(image_set), return_masks=args.masks)
        return dataset
