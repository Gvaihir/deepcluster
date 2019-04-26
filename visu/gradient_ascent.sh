# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='/data2/02_sudoku/model_alex/checkpoint.pth.tar'
ARCH='alexnet'
EXP='/data2/02_sudoku/model_alex/ascent'
CONV=5

python gradient_ascent.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --arch ${ARCH}
