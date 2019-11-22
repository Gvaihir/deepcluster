# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/sudoku/anomaly_links/Pt04"
ARCH="alexnet"
LR=0.05
WD=-5
K=80
WORKERS=8
EPOCHS=700
BATCH=256
EXP="/home/aogorodnikov/deepclust_afterACAE"
RESUME="/sudoku/deepclust_afterACAE/checkpoint.pth.tar"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"
CLUST="Kmeans"


mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}  \
  --clustering ${CLUST} --epochs ${EPOCHS} --batch ${BATCH}
