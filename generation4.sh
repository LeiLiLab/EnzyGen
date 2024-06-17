#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

data_path=/mnt/gemini/data/zhenqiaosong/protein_design/data/enzyme_substrate_data_lucky_best.json

local_root=/mnt/gemini/data/zhenqiaosong/protein_design/geometric_protein_design/models
output_path=${local_root}/naepro_substrate

proteins=(3.1.1 2.5.1 2.7.7 2.6.1 4.1.1 1.11.1 3.6.1 1.14.14)

for element in ${proteins[@]}
do
generation_path=/mnt/taurus/data2/zhenqiaosong/protein_design/geometric_protein_design/output/${element}

mkdir -p ${generation_path}
mkdir -p ${generation_path}/pred_pdbs
mkdir -p ${generation_path}/tgt_pdbs

python3 fairseq_cli/validate.py ${data_path} \
--task geometric_protein_design \
--protein-task ${element} \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery
done