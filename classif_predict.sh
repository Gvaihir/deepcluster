# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/sudoku/02_sudoku/cropped/Pt04/"
MODEL="/home/aogorodnikov/kmeans100_linear_classif_imagenet/checkpoint.pth.tar"
CLASSES="/home/aogorodnikov/classes.txt"
EXP="/home/aogorodnikov/classif_predict/Pt04"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"
BATCH=256
WORKERS=8


mkdir -p ${EXP}

${PYTHON} classif_predict.py --model ${MODEL} --data ${DATA} \
    --batch_size ${BATCH} --verbose --exp ${EXP} \
    --class_labels ${CLASSES} --workers ${WORKERS}


