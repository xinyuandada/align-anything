#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


MODEL_NAME_OR_PATH="/data/zhangyanhong-2401220256/align-anything/outputs/qwen_2_5_rm/5_16_three-epoch/slice_end" # model path 
#1、修改模型地址

TRAIN_DATASETS="" # rm dataset path
#2、修改数据路径

TRAIN_TEMPLATE="HOMEWORK" # dataset template
#3、修改训练模版

#TRAIN_SPLIT="validation" # split the dataset
#4、修改数据名称
TRAIN_DATASETS="" # dataset path
TRAIN_TEMPLATE="" # dataset template
TRAIN_SPLIT="train" # split the dataset


EVAL_DATASETS="/data/zhangyanhong-2401220256/align_anything_t2t/validation" # dataset path
EVAL_TEMPLATE="HOMEWORK" # dataset template, reuse the template of Qwen2-VL
EVAL_SPLIT="validation" # split the dataset
#EVAL_DATA_FILES="test.jsonl"


OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR

if [ -z "$OUTPUT_ROOT_DIR" ]; then
    echo "OUTPUT_ROOT_DIR is not set"
    OUTPUT_ROOT_DIR="../outputs"
fi

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/qwen_2_5_rm/5_18_three-epoch_validation" # output dir

# For wandb online logging
export WANDB_API_KEY="5947a4df1bd19d75524f2c0896d2cf97bc2dc724"
#5、增加WANDB_API_KEY

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 1 \
     --epochs 1

    



