# Project Name

## Description

This repository contains code for the models and training.
We use the configuration framework [Hydra](https://hydra.cc) along with [Hydra Lightning Template](https://github.com/ashleve/lightning-hydra-template). You will therefore find all
relevant configurations in [configs/](configs/).

## Requirements

- this codebase has been tested on an Ubuntu 20.04 machine and with the included [Dockerfile](Dockerfile)
- you need at least Python 3.7
- a current CUDA version that works with the used PyTorch version

## Prerequesites

Install python packages with

```bash
pip install -r requirements.txt
```

## Run training

### Single training

We provide the configurations we used for sweeps and individual configurations in [configs/experiments/](configs/experiments/).

For example, to train the winner CNN+BRA model run:

```bash
python run.py +experiments/who_is_alyx=winner_bra_model
```
Note, that we use [Weights and Biases](https://wandb.ai) for monitoring the training, which is currently tightly integrated with our code, so you probably need a (free) account. We
plan to make this an optional requirement in the future.

## Sweep Configs

You find the configuration files for the hyperparameter sweeps we did with Weights & Biases in the folder `wandb_sweep_cofnigs/who_is_alyx`.