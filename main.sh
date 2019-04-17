# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/aogorodnikov/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=100
WORKERS=12
EPOCHS=100
EXP="/home/aogorodnikov/model_alex"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
