#!/bin/bash

# Data
DATA_ROOT=./
DATA_DIR=${DATA_ROOT}/pretrained_xl/tf_enwik8/data
MODEL_DIR=${DATA_ROOT}/pretrained_xl/tf_enwik8/model

# Model
N_LAYER=24
D_MODEL=1024
D_EMBED=1024
N_HEAD=8
D_HEAD=128
D_INNER=3072

# Testing
TEST_TGT_LEN=128
TEST_MEM_LEN=3800
TEST_CLAMP_LEN=1000

TEST_CKPT_PATH=${MODEL_DIR}/model.ckpt-0

echo 'Run evaluation on test set...'
python extract_features.py \
    --data_dir=${DATA_DIR} \
    --eval_ckpt_path=${TEST_CKPT_PATH} \
    --n_layer=${N_LAYER} \
    --d_model=${D_MODEL} \
    --d_embed=${D_EMBED} \
    --n_head=${N_HEAD} \
    --d_head=${D_HEAD} \
    --d_inner=${D_INNER} \
    --dropout=0.0 \
    --dropatt=0.0 \
    --tgt_len=${TEST_TGT_LEN} \
    --mem_len=${TEST_MEM_LEN} \
    --clamp_len=${TEST_CLAMP_LEN} \
    --same_length=True \
