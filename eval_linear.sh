# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/home/aogorodnikov/classes/"
MODEL="/home/aogorodnikov/model_alex_kmeans300/checkpoint.pth.tar"
EXP="/home/aogorodnikov/kmeans300_linear_classif"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"
EPOCHS=100
BATCH=256


mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} \
    --epochs ${EPOCHS} --batch_size ${BATCH} --conv 5 --lr 0.01 \
    --make_test "True" --val_prob 0.2 --test_prob 0.1 \
    --wd -7 --verbose --exp ${EXP}


