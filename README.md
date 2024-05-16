# Code for Sparse Multimodal Fusion with Modal Channel Attention

This repository is the official implementation of Sparse Multimodal Fusion with Modal Channel Attention

![Model](./figures/figure0.svg)
![MCA](./figures/figure1a-x.png)


# Requirements

To install requirements:

```angular2html
pip install -r requirements.txt
```

This repository makes heavy use of Huggingface Accelerator and Datasets libraries for managing training and data.

# Training
To train the model, choose a configuration file from the configs directory and run

```angular2html
accelerate launch train_accel_gpu.py <config_file_path>
```

# Evaluation
To evaluate the model, run an inference using pretrained model weights, then train a linear probe to fit a target property.

To run a batch inference

```angular2html
accelerate launch infer_accel_gpu.py <config_file_path>
```

To train a linear probe or MLP using the embeddings generated by inference:

```angular2html
accelerate launch lp_accel_gpu.py <config_file_path>
```

# Pre-trained Models

# Results

# Contributing



## NeurIPS 2024 submission

