#!/bin/bash

source activate rdkit
python train.py \
    --data_path data/train_smiles.csv \
    --dataset_type regression \
    --save_dir test_checkpoints \
    --split_type scaffold_balanced \
    --num_folds 10