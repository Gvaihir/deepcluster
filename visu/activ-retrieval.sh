# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='/home/aogorodnikov/model_alex/checkpoint.pth.tar'
EXP='/home/aogorodnikov/tmp'
CONV=5
DATA='/home/aogorodnikov/val/cropped_small_4'

python activ-retrieval.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --data ${DATA}
