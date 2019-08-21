#!/bin/bash

source activate rdkit
VISIBLE_CUDA_DEVICES=7 python -u train.py \
        --task test_kmean \
        --init_data data/sheridan_train.csv \
        --test_data data/sheridan_test.csv \
        --pool_data data/chembl250k.csv \
        --path_to_config config/kmean.json \
        --mol_col SMILES \
        --mol_prop tb_score \
        --oracle tb
