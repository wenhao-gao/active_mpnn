#!/bin/bash

source activate rdkit
VISIBLE_CUDA_DEVICES=7 python -u train.py \
        --init_data data/sheridan_train.csv \
        --test_data data/sheridan_test.csv \
        --pool_data data/chembl250k.csv \
        --mol_col SMILES \
        --mol_prop sa_score \
        --strategy random \
        --max_data_size 10000 \
        --oracle sa &> temp.out&
