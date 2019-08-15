# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

rm -r ~/train
rm -r ~/test
rm -r ~/val


DATA="/home/aogorodnikov/classes_rgb_augmented/"
MODEL="/home/aogorodnikov/model_alex100_rgb/checkpoint.pth.tar"
EXP="/home/aogorodnikov/linear_classif_rgb_e100_augmented"
PYTHON="/home/aogorodnikov/anaconda3/envs/imgSudoku/bin/python"
EPOCHS=100
BATCH=256


mkdir -p ${EXP}

${PYTHON} classif.py --model ${MODEL} --data ${DATA} \
    --epochs ${EPOCHS} --batch_size ${BATCH} --lr 0.01 \
    --make_test "True" --val_prob 0.2 --test_prob 0.1 \
    --weight_decay -7 --verbose --exp ${EXP} --freeze


