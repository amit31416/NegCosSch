# NegCosSch
Implementation of the paper:
Boosting Open Set Recognition Performance through Modulated Representation Learning

**[Link to your Paper (e.g., arXiv or conference proceedings)]**

## Overview

A brief (2-3 sentences) overview of your paper and what this codebase achieves. What problem does it solve? What is the core contribution? Why is it important?

*Example:*
This repository contains the official implementation for our work on Negative Cosine Temperature Scheduling, a novel approach for Open Set Recognition. We demonstrate that our temperature scheduling can significantly boost both open set and closed set performance over the baseline methods. Our code allows for training, evaluation, and reproduction of the key results presented in the paper.

## Visualization of Results / Method

Here we showcase some key visualizations from our paper that highlight the effectiveness of our method or illustrate its core concept.

| Figure 1: [Short Caption]      | Figure 2: [Short Caption]      |
| :----------------------------: | :----------------------------: |
| ![Figure 1 Alt Text](path/to/your/figure1.png) | ![Figure 2 Alt Text](path/to/your/figure2.png) |
| **Figure 3: [Short Caption]** | **Figure 4: [Short Caption]** |
| ![Figure 3 Alt Text](path/to/your/figure3.png) | ![Figure 4 Alt Text](path/to/your/figure4.png) |

*(Briefly explain what these figures demonstrate, e.g., "Figure 1 shows a UMAP visualization of learned representations for Dataset X, Figure 2 illustrates the architecture of our proposed model, etc.")*


## Basic Usage / Quick Start
Our Negative Cosine Scheduling can be integrated into any method using a few lines of code

```python
from temperature_schedulers import GCosineTemperatureScheduler

if(args.temperature_scheduling):
    TS=GCosineTemperatureScheduler()
for epoch in range(1,N_epochs+1):
    if(args.temperature_scheduling):
        criterion.temperature = TS.get_temperature(epoch)
    # rest of the code
    ... ...
```

### Training

A basic command to start training:
```bash
python train.py --dataset cifar100 \
                --model resnet50 \
                --epochs 200 \
                --batch_size 128 \
                --learning_rate 0.1 \
                --temperature_schedule cosine \
                --output_dir ./experiments/run001


This implementation heavily builds upon from the following repositories:
1. osr_closed_set_all_you_need(https://github.com/sgvaze/osr_closed_set_all_you_need) by Vaze et al. [2022]: data loading pipeline, base model architecture, evaluation
2. [SupContrast](https://github.com/HobbitLong/SupContrast) by Khosla et al. [2021]: We took the supervised contrastive loss from here.
