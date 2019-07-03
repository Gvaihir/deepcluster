# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/home/aogorodnikov/classes/"
MODEL="/home/aogorodnikov/model_alex_kmeans300/checkpoint.pth.tar"
CLASSES="/home/aogorodnikov/classes.txt"
EXP="/home/aogorodnikov/classif_predict"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"
BATCH=256
WORKERS=8


mkdir -p ${EXP}

${PYTHON} classif_predict.py --model ${MODEL} --data ${DATA} \
    --batch_size ${BATCH} --verbose --exp ${EXP} \
    --class_labels ${CLASSES} --workers ${WORKERS}


