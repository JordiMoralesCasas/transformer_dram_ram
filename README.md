# Fixation-Guided Visual Attention Models with Transformers

### Implementation of the RAM [1] and DRAM [2] recurrent visual attention models, along with a modification that uses Transformers instead of RNNs/LSTMs as the core of the model.

This repository is part of the Final Master Thesis in the master of Computer Vision, at Universitat Autónoma de Barcelona, with the participation of the Computer Vision Center (CVC), Universitat de Barcelona (UB), Universitat Pompeu Fabra (UPF), Universitat Politècnica de Catalunya (UPC) and Universitat Oberta de Catalunya (UOC).

We show how to run these models on the MNIST (single digit classification) and SVHN (multiple digit classification) datasets

Additionally, we provide two aditional tasks based on two synthetic datasets derived from SVHN:
- SVHN with an extended background to test the localization capabilities of the model. The images below show examples of 110x110, 186x186 and 224x224 synthetic images.

<p align="center">
<img src="images/extend_background_example_110.png" alt="Description" width="200" height="200" style="border: 10px solid \#000;">
<img src="images/extend_background_example_186.png" alt="Description" width="200" height="200" style="border: 10px solid \#000;">
<img src="images/extend_background_example_224.png" alt="Description" width="200" height="200" style="border: 10px solid \#000;">
<p>

- SVHN with multiple numbers (sequences of digits) per image. 

<p align="center">
<img src="images/multiple_numbers_example.png" alt="Description" width="250" height="250">
<p>

## Getting Started

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/JordiMoralesCasas/transformer_dram_ram

   cd transformer_dram_ram
   ```

2. **Create virtual environment (CONDA)**:
    ```bash
    conda env create -n env_name

    conda activate env_name
    ```

3. **Intall required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup the data

How to create the data for the different experiments.

### MNIST dataset

All the data is already included in this repo. No additional steps are required.

### SVHN dataset

1. Download the SVHN dataset from the [official website](http://ufldl.stanford.edu/housenumbers/). All three splits from the **Format 1** section must be downloaded (*train.tar.gz*, *test.tat.gz*, *extra.tar.gz*).

2. Extract the three files into the same directory. The folder structure should look something like this:
   ```
    |____extra/
    |    |____1.png
    |    |   ...
    |    |____digitStruct.mat
    |    |____see_bboxes.m
    |
    |____test/
    |    |____1.png
    |    |   ...
    |    |____digitStruct.mat
    |    |____see_bboxes.m
    |
    |____train/
    |    |____1.png
    |    |   ...
    |    |____digitStruct.mat
    |    |____see_bboxes.m
   ```

### SVHN with an extended background

No additional steps are required since the data is created during runtime using the original SVHN dataset.

### SVHN with multiple numbers

To create the synthetic SVHN dataset with multiple numbers per sample, run the following code.

Use the default values to use the same configuration as in our work. *DATA_DIR* refers to the directory created in the **SVHN dataset** subsection.

```
python3 data/svhn/create_multinumber_dataset.py --help
    usage: data/svhn/create_multinumber_dataset.py 
                    [--data_dir DATA_DIR] 
                    [--save_dir SAVE_DIR]
                    [--img_size IMG_SIZE]
                    [--bbox_size BBOX_SIZE]
                    [--train_split_size SPLIT]
                    [--dataset_length LENGTH]
                    [--num_workers N_WORKERS]

    options:
    -h, --help                 Show this help message and exit
    --data_dir DATA_DIR        Directory with the original SVHN dataset.
    --save_dir SAVE_DIR        Directory where the new dataset will be saved.
    --img_size IMG_SIZE        Size of the dataset samples (Square).
    --bbox_size BBOX_SIZE      Size of the resized number's bounding box 
                               (Square).
    --train_split_size SPLIT   Portion of the whole dataset that is used for
                               training. The remaining is divided by two to 
                               create the validation and test partitions.
    --dataset_length LENGTH    Length of the new dataset.
    --num_workers N_WORKERS    Size of the pool of workers.
```
## Running examples

## Gridsearch example

### Integration with WandB
Log into WandB using the terminal and change the "wandb_entity" and "wandb_project" parameters in the configuration file (config.py). Then, use the flag "--wandb_name ANY_NAME" when training to log the experiment into WandB.

## Important Acknowledgements
This project has been built using @[kevinzakka](https://github.com/kevinzakka)'s excellent implementation of the RAM model as 
starting point: https://github.com/kevinzakka/recurrent-visual-attention

Also, many thanks to the GTrXL [3] model implementation by @[OpenDILab](https://github.com/opendilab), which has been of great help: https://github.com/opendilab/PPOxFamily/blob/main/chapter5_time/gtrxl.py

## References

[1] [Volodymyr Mnih et al. "Recurrent Models of Visual Attention"](https://arxiv.org/abs/1406.6247)

[2] [Jimmy Ba et al. "Multiple Object Recognition with Visual Attention"](https://arxiv.org/abs/1412.7755)

[3] [Emilio Parisotto et al. "Stabilizing Transformers for Reinforcement Learning"](https://arxiv.org/abs/1910.06764)
