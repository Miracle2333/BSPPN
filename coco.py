"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import re
import logging
import cv2
import csv
from skimage.measure import find_contours
import skimage.draw
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from keras.utils import plot_model

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

#import Mask RCNN
from config import Config
import utils
import model as modellib
import visualize

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1+1  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_building(self, dataset_dir, subset):
        """Load a subset of the Building dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or test
        """
        # Add classes. We have only one class to add.
        # self refers to NucleusDataset
        self.add_class("building", 1, "building")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        if subset == "test":
            dataset_dir = os.path.join(dataset_dir, subset)
            test_files = os.listdir(dataset_dir)
            for f in test_files:
                filename = f
                image_path = os.path.join(dataset_dir, filename)
                height = 650
                width = 650
                self.add_image(
                    "building",
                    image_id=filename,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height)
        else:
            dataset_dir = os.path.join(dataset_dir, subset)
            annotations = json.load(open(os.path.join(dataset_dir,
                                                      "AOI_2_Vegas_Train_Building_Solutions_modified.json")))
            # Add images
            polygons = []
            for a in annotations:
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. There are stores in the
                # shape_attributes (see json format above)
                # polygons = [r['shape_attributes'] for r in a['regions'].values()]

                if a['BuildingId'] != '1':
                    poly = {}.fromkeys(['x', 'y'])
                    poly['x'] = [float(s) for s in re.findall(r'-?\d+\.?\d*', a['X'])]
                    poly['y'] = [float(s) for s in re.findall(r'-?\d+\.?\d*', a['Y'])]
                    polygons.append(poly)
                    # load_mask() needs the image size to convert polygons to masks.
                    # Unfortunately, VIA doesn't include it in JSON, so we must read
                    # the image. This is only managable since the dataset is tiny.
                    filename = 'RGB-PanSharpen_' + a['ImageId'] + '.tif'
                    image_path = os.path.join(dataset_dir, filename)
                    # image = skimage.io.imread(image_path)
                    # height, width = image.shape[:2]
                    height = 650
                    width = 650
                else:
                    if polygons != []:
                        self.add_image(
                            "building",
                            image_id=filename,  # use file name as a unique image id
                            path=image_path,
                            width=width, height=height,
                            polygons=polygons)
                    polygons = []
                    poly = {}.fromkeys(['x', 'y'])
                    poly['x'] = [float(s) for s in re.findall(r'-?\d+\.?\d*', a['X'])]
                    poly['y'] = [float(s) for s in re.findall(r'-?\d+\.?\d*', a['Y'])]
                    polygons.append(poly)
                    # load_mask() needs the image size to convert polygons to masks.
                    # Unfortunately, VIA doesn't include it in JSON, so we must read
                    # the image. This is only managable since the dataset is tiny.
                    filename = 'RGB-PanSharpen_' + a['ImageId'] + '.tif'
                    image_path = os.path.join(dataset_dir, filename)
                    # image = skimage.io.imread(image_path)
                    # height, width = image.shape[:2]
                    height = 650
                    width = 650
        a = 1




    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_id1 = 1254
        image_info = self.image_info[image_id]
        if image_info["source"] != "building":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])],
                        dtype=np.uint8)
        for i, a in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # avoid the area of mask is 0 or return of rr, cc has no value
            x = a['x']
            y = a['y']
            if(np.size(x)<=3|np.size(y)<=3):
                continue
            elif((np.abs(np.max(x)-np.min(x)) < 1.5)|(np.abs(np.max(y)-np.min(y)) < 1.5)):
                continue
            else:
                rr, cc = skimage.draw.polygon((a['y']), (a['x']))
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

# def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
#     """Arrange resutls to match COCO specs in http://cocodataset.org/#format
#     """
#     # If no results, return an empty list
#     if rois is None:
#         return []
#
#     results = []
#     for image_id in image_ids:
#         # Loop through detections
#         for i in range(rois.shape[0]):
#             class_id = class_ids[i]
#             score = scores[i]
#             bbox = np.around(rois[i], 1)
#             mask = masks[:, :, i]
#
#             result = {
#                 "image_id": image_id,
#                 "category_id": dataset.get_source_class_id(class_id, "coco"),
#                 "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
#                 "score": score,
#                 "segmentation": maskUtils.encode(np.asfortranarray(mask))
#             }
#             results.append(result)
#     return results


def evaluate_coco(model, dataset, eval_type="bbox", limit=0, output=None, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """

    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    # coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    # t_prediction = 0
    # t_start = time.time()

    for image_id in image_ids:
        image = dataset.load_image(image_id)
        source_id_temp = dataset.image_info[image_id]["id"]  # source ID = original image name
        source_id = source_id_temp.split('.')[0]
        # image_name = source_id.split('_', 1)[1]
        r = model.detect([image])[0]
        # boxes = r['rois']
        # masks = r['masks']
        # scores = r['scores']
        # class_ids = r['class_ids']
        visualize.display_detection(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, source_id,
                                    output, r['scores'])

    # results = []
    # for i, image_id in enumerate(image_ids):
    #     # Load image
    #     image = dataset.load_image(image_id)
    #
    #     # Run detection
    #     t = time.time()
    #     r = model.detect([image])[0]
    #     t_prediction += (time.time() - t)

        # # Convert results to COCO format
        # image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
        #                                    r["rois"], r["class_ids"],
        #                                    r["scores"], r["masks"])
        # results.extend(image_results)

    # # Load results. This modifies results with additional attributes.
    # coco_results = coco.loadRes(results)
    #
    # # Evaluate
    # cocoEval = COCOeval(coco, coco_results, eval_type)
    # cocoEval.params.imgIds = coco_image_ids
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
    #
    # print("Prediction time: {}. Average {}/image".format(
    #     t_prediction, t_prediction / len(image_ids)))
    # print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--output', required=False,
                        metavar="/path/to/result",
                        help="Path to save the detection result ")
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)


    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_building(args.dataset, args.subset)
        # dataset_train.load_building(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_building(args.dataset, 'val')
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=75,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_test = CocoDataset()
        dataset_test.load_building(args.dataset, "test")
        dataset_test.prepare()
        print("Running COCO detection on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_test, "bbox", limit=int(args.limit))
        evaluate_coco(model, dataset_test, "segm", limit=int(args.limit), output=args.output)
        print("Detection results are saved at {}".format(args.output))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
