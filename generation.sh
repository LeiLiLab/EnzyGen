#!/bin/bash

data_path=data/enzyme_substrate_data_lucky_best.json

output_path=models
proteins=(1.1.1)

for element in ${proteins[@]}
do
generation_path=${element}

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