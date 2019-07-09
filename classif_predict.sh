# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/sudoku/crop_rgb/Pt11/"
MODEL="/home/aogorodnikov/linear_classif_rgb_e100/checkpoint.pth.tar"
CLASSES="/home/aogorodnikov/classes_rgb.txt"
EXP="/home/aogorodnikov/classif_predict_rgb/Pt11"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"
BATCH=128
WORKERS=4


mkdir -p ${EXP}

${PYTHON} classif_predict.py --model ${MODEL} --data ${DATA} \
    --batch_size ${BATCH} --verbose --exp ${EXP} \
    --class_labels ${CLASSES} --workers ${WORKERS}


