#!/bin/sh

module load anaconda > /dev/null 2>&1 || module load conda > /dev/null 2>&1
conda create --name yolov5 python=3.8 --yes
conda activate yolov5
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
