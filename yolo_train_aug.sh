#!/bin/bash
cd /home/ubuntu_yannn/projects/Yannngn/reticuloc
source .venv/bin/activate
yolo detect train model=yolov8n.pt data=data/dataset.yaml project=runs name=aug imgsz=640 rect=True hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 degrees=15 translate=0.1 scale=0.5 shear=0.0 perspective=0.0 flipud=0.5 fliplr=0.5 bgr=0.0 mosaic=0.0
