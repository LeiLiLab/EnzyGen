<h1>Generative Enzyme Design Guided by Functionally Important Sites and Small-Molecule Substrates</h1>

<h2>Model Architecture</h2>

This repository contains code, data and model weights for ICML 2024 paper [Generative Enzyme Design Guided by Functionally Important Sites and Small-Molecule Substrates](https://openreview.net/pdf/b349f5504ef1e6143231064979e2e96feaf5a6a9.pdf)

The overall model architecture is shown below:

![image](./enzygen_overview.png)


<h2>Environment</h2>
The dependencies can be set up using the following commands:

```ruby
conda create -n enzygen python=3.8 -y 
conda activate enzygen 
conda install pytorch=1.10.2 cudatoolkit=11.3 -c pytorch -y 
bash setup.sh 
```

<h2>Download Data</h2>

We provide the EnzyBench at [EnzyBench](https://drive.google.com/file/d/1VycT_gFV2JBpRMCBZlwwxLLRcZDljXCS/view?usp=drive_link) 
 and Enzyme Classification Tree (EC) ID to index dict at [EC_Dict](https://drive.google.com/file/d/1BCitsFRQpzUbGss7xBpTpvKcMcJh_oOz/view?usp=drive_link)

Please download the dataset and put them in the data folder.

```angular2html
mkdir data 
cd data 
wget https://drive.google.com/file/d/1VycT_gFV2JBpRMCBZlwwxLLRcZDljXCS/view?usp=drive_link
wget https://drive.google.com/file/d/1BCitsFRQpzUbGss7xBpTpvKcMcJh_oOz/view?usp=drive_link
```

<h2>Download Model</h2>

We provide the checkpoint used in the paper at [Model](https://drive.google.com/file/d/1Sn6uKNnL8gkDErKZ6hFic5lmEF3Vb9Ue/view?usp=sharing) 


Please download the checkpoints and put them in the models folder.

If you want to train your own model, please follow the training guidance below

<h2>Training</h2>
If you want to train a model with enzyme-substrate interaction constraint as introduced in our paper, please follow the script below:

```ruby
bash train_enzyme_substrate_33layer.sh
```

If you want to train a model without enzyme-substrate interaction constraint, please follow the script below:

```ruby
bash train_cluster_enzyme_33layer.sh
```

From our experiences, first training a model without enzyme-substrate interaction constraint for around 200,000 steps and then continue training based on sequence recovery loss, coordinate recovery loss and enzyme-substrate interaction loss will lead to the best performance!

<h2>Inference</h2>
To design enzymes for the 30 testing third-level categories, please use the following scripts:

```ruby
bash generation.sh
```

There are five items in the output directory:

1. protein.txt refers to the designed protein sequence
2. src.seq.txt refers to the ground truth sequences
3. pdb.txt refers to the target PDB ID and the corresponding chain
4. pred_pdbs refers to the directory of designed pdbs
5. tgt_pdbs refers to the directory of target pdbs

<h2>Finetune your own model</h2>
To finetune your own model based on our trained model, please follow the guidelines below:

<h3>Prepare your own data</h3>
We provide a case of training data at preprocess/case.json. For training and validation, you should prepare ['seq', 'coor', 'motif', 'pdb', 'ec4', 'substrate', 'binding', 'substrate_coor', 'substrate_feat'] features. Seq denotes the protein sequence, coor denotes the alpha-carbon coordinates which is flattened with the order of x, y, z coordinate.
motif denotes the functional sites indexing from 0. pdb denotes the pdb id and chain. ec4 dotes the fourth EC category.
substrate denotes the substrate id and binding (0 or 1) denotes if the substrates can bind to the enzyme.
substrate_coor and substrate_feat respectively denotes the coordinates and features of the substrates.
You can extract the substrate coordinates and features using preprocess/get_substrate_feature.py.

```ruby
python preprocess/get_substrate_feature.py
```

<h3>Finetuning your model</h3>
After preparing your own data, you can finetune your model using finetune.sh

```ruby
bash finetune.sh
```

<h2>Evaluation</h2>
We provide the ESP evaluation data at [ESP_data_eval](https://drive.google.com/file/d/1D3AfYSh6ESv6uh6E4zY1tVpOi1t8RTWI/view?usp=sharing)

The format for ESP evaluation is (Protein_Sequence Substrate_Representation) for each test case.

The evaluation code for ESP score is developed by Alexander Kroll, which can be found at [link](https://github.com/AlexanderKroll/ESP_prediction_function/tree/main)

<h3>Expected Results</h3>

| Protein Family |   1.1.1   |  1.11.1   |  1.14.13  |  1.14.14  |   1.2.1    |   2.1.1    |   2.3.1   |   2.4.1   |
|:---------------|:---------:|:---------:|:---------:|:---------:|:----------:|:----------:|:---------:|:---------:|
| EnzyGen        |   0.64    |   0.98    |   0.38    |   0.42    |    0.72    |    0.80    |   0.61    |   0.38    |
| Protein Family | **2.4.2** | **2.5.1** | **2.6.1** | **2.7.1** | **2.7.10** | **2.7.11** | **2.7.4** | **2.7.7** |
| EnzyGen        |   0.86    |   0.66    |   0.53    |   0.76    |    0.92    |    0.93    |   0.80    |   0.79    |
| Protein Family | **3.1.1** | **3.1.3** | **3.1.4** | **3.2.2** | **3.4.19** | **3.4.21** | **3.5.1** | **3.5.2** |
| EnzyGen        |   0.76    |   0.62    |   0.88    |   0.47    |    0.26    |    0.73    |   0.40    |   0.14    |
| Protein Family | **3.6.1** | **3.6.1** | **3.6.5** | **4.1.1** | **4.2.1**  | **4.6.1**  |    --     |  **Avg**  |
| EnzyGen        |   0.66    |   0.78    |   0.40    |   0.80    |    0.93    |    0.57    |    --     |   0.65    |


<h2>Citation</h2>
If you find our work helpful, please consider citing our paper.

```
@inproceedings{songgenerative,
  title={Generative Enzyme Design Guided by Functionally Important Sites and Small-Molecule Substrates},
  author={Song, Zhenqiao and Zhao, Yunlong and Shi, Wenxian and Jin, Wengong and Yang, Yang and Li, Lei},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
