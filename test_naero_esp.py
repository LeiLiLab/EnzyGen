import os
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import  json
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment


def get_substrate_pairs(file_path):
    pdb2substrate = {}
    lines = open(file_path).readlines()
    for line in lines:
        items = line.strip().split()
        substrate = items[1].strip().replace(".sdf", "").strip()
        if substrate.startswith("CHEBI"):
            substrate = substrate.replace("_", ":")
        pdb2substrate[items[0].strip()] = substrate
    return pdb2substrate

data_path = "/mnt/data7/zhenqiaosong/protein_design/geometric_protein_design/data/enzyme_data/ec_four_level"
pdb2substrate = get_substrate_pairs(os.path.join(data_path, "enzyme_substrate_pair_data", "enzyme_substrate_pair.txt"))

smiles_file = "/mnt/data7/zhenqiaosong/protein_design/geometric_protein_design/data/enzyme_data/compound_data/chebi2smiles.json"
chembi2smiles = json.load(open(smiles_file))

tests = ['1.1.1', '1.11.1', '1.14.13', '1.14.14', '1.2.1', '2.1.1', '2.3.1', '2.4.1', '2.4.2', '2.5.1', '2.6.1',
         '2.7.1', '2.7.10', '2.7.11', '2.7.4', '2.7.7', '3.1.1', "3.1.3", '3.1.4', '3.2.2', '3.4.19', '3.4.21',
         '3.5.1', '3.5.2', '3.6.1', '3.6.4', '3.6.5', '4.1.1', '4.2.1', '4.6.1']

data_dir = "/mnt/taurus/data2/zhenqiaosong/protein_design/geometric_protein_design/output"


for enzyme in tests:
    input_path = os.path.join(data_dir, enzyme, "protein.txt")
    pdb_path = os.path.join(data_dir, enzyme, "pdb.txt")
    output_path = os.path.join(data_dir, enzyme, "enzyme_substrate_pair.txt")

    proteins = open(input_path).readlines()
    pdbs = open(pdb_path).readlines()
    fw = open(output_path, "w", encoding="utf-8")
    if len(proteins) == 0:
        print(enzyme)
    for protein, pdb in zip(proteins, pdbs):
        substrate = pdb2substrate[pdb.strip()]
        print(substrate)

        if enzyme == "3.1.4" and substrate == "CHEBI:58165":
            substrate = "InChI=1S/C10H12N5O6P/c11-8-5-9(13-2-12-8)15(3-14-5)10-6(16)7-4(20-10)1-19-22(17,18)21-7/h2-4,6-7,10,16H,1H2,(H,17,18)(H2,11,12,13)/p-1/t4-,6-,7-,10-/m1/s1"

        if enzyme == "3.1.4" and substrate == "CHEBI:57746":
            substrate = "InChI=1S/C10H12N5O7P/c11-10-13-7-4(8(17)14-10)12-2-15(7)9-5(16)6-3(21-9)1-20-23(18,19)22-6/h2-3,5-6,9,16H,1H2,(H,18,19)(H3,11,13,14,17)/p-1/t3-,5-,6-,9-/m1/s1"

        if enzyme == "3.1.3" and substrate == "CHEBI:30616":
            substrate = "InChI=1S/C10H16N5O13P3/c11-8-5-9(13-2-12-8)15(3-14-5)10-7(17)6(16)4(26-10)1-25-30(21,22)28-31(23,24)27-29(18,19)20/h2-4,6-7,10,16-17H,1H2,(H,21,22)(H,23,24)(H2,11,12,13)(H2,18,19,20)/p-4/t4-,6-,7-,10-/m1/s1"

        if enzyme == "3.1.3" and substrate == "CHEBI:58579":
            substrate = "InChI=1S/C6H14O12P2/c7-2-6(18-20(13,14)15)5(9)4(8)3(17-6)1-16-19(10,11)12/h3-5,7-9H,1-2H2,(H2,10,11,12)(H2,13,14,15)/p-4/t3-,4-,5+,6+/m1/s1"

        if enzyme == "4.1.1" and substrate == "CHEBI:57538":
            substrate = "InChI=1S/C10H13N2O11P/c13-5-1-3(9(16)17)12(10(18)11-5)8-7(15)6(14)4(23-8)2-22-24(19,20)21/h1,4,6-8,14-15H,2H2,(H,16,17)(H,11,13,18)(H2,19,20,21)/p-3/t4-,6-,7-,8-/m1/s1"
        if substrate.startswith("CHEBI"):
            print(pdb)
            print(substrate)
            substrate = chembi2smiles[substrate]
            print(substrate)
        new_line = " ".join([protein.strip(), substrate])
        fw.write(new_line + "\n")
    fw.close()


