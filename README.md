<h1>Generative Enzyme Design Guided by Functionally Important Sites and Small-Molecule Substrates</h1>

<h2>Model Architecture</h2>

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
 and EC ID dict at [EC_Dict](https://drive.google.com/file/d/1BCitsFRQpzUbGss7xBpTpvKcMcJh_oOz/view?usp=drive_link)

Please download the dataset and put them in the data folder.


<h2>Download Models</h2>

We provide the checkpoint used in the paper at [Model](https://drive.google.com/file/d/1J017IloeycjTucovPncg-URMhnHr-jie/view?usp=drive_link) 


Please download the checkpoints and put them in the models folder.

If you want to train your own model, please follow the training guidance below

<h2>Training</h2>
If you want to train a model without enzyme-substrate interaction constraint, please follow the script below:

```ruby
bash train_cluster_enzyme_33layer.sh
```

If you want to train a model with enzyme-substrate interaction constraint as introduced in our paper, please follow the script below:

```ruby
bash train_enzyme_substrate_33layer.sh
```

From our experiences, first training a model without substrate constraint for around 200000 steps and then continue training would lead to the best performance!

<h2>Inference</h2>
Taking category 1.1.1 as an example, we can design enzyme via the following scripts:

```ruby
bash generation.sh
```

There are five items in the output directory:

1. protein.txt refers to the designed protein sequence
2. src.seq.txt refers to the ground truth sequences
3. pdb.txt refers to the target PDB ID and the corresponding chains
4. pred_pdbs refers to the directory of designed pdbs
5. tgt_pdbs refers to the directory of target pdbs


<h2>Evaluation</h2>
We provide the ESP evaluation data at [ESP_data_eval](https://drive.google.com/file/d/1q8NENdVWBufz5fDk7TviS6h6_BKmfviN/view?usp=drive_link)

The evaluation code for ESP score is developed by Alexander Kroll, which can be found at [link](https://github.com/AlexanderKroll/ESP_prediction_function/tree/main)

