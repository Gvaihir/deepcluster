# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/data/02_sudoku/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=100
WORKERS=4
EPOCHS=10
BATCH=256
EXP="/sudoku/02_sudoku/model_alex_add10"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"
RESUME="/home/aogorodnikov/model_alex"


mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} --resume ${RESUME}
