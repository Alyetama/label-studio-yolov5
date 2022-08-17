#!/usr/bin/env python
# coding: utf-8
"""
Source: https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
"""

import os
import random
from pathlib import Path

from tqdm import tqdm


def img2label_paths(img_paths) -> list:
    # Define label paths as a function of image paths
    sa = os.sep + 'images' + os.sep  # /images/ substrings
    sb = os.sep + 'labels' + os.sep  # /labels/ substrings
    return [
        sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths
    ]


def autosplit(path: str,
              weights: tuple = (0.9, 0.1, 0.0),
              annotated_only: bool = False) -> None:
    """Autosplit a dataset

    Autosplit a dataset into train/val/test splits and save 
    path/autosplit_*.txt files

    Args:
        path (str): Path to images directory
        weights (tuple): Train, val, test weights
        annotated_only (bool): Only use images with an annotated txt file
    """
    IMG_FORMATS = [
        'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'
    ]  # acceptable image suffixes

    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*')
                   if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights,
                             k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt',
           'autosplit_test.txt']  # 3 txt files
    _ = [(path.parent / x).unlink() for x in txt
         if Path(x).exists()]  # remove existing

    print(f'Autosplitting images from {path}' +
          ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths(
            [str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{Path(*img.parts[1:])}' +
                        '\n')  # add image to txt file
