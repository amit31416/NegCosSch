# Boosting Open Set Recognition Performance through Modulated Representation Learning
Implementation of the paper:
Boosting Open Set Recognition Performance through Modulated Representation Learning


## Overview

This repository contains the official implementation for our novel negative cosine temperature scheduling (NegCosSch), a novel approach for Open Set Recognition. We demonstrate that our temperature scheduling can significantly boost both open set and closed set performance if folded into an OSR method. Our code allows for training, evaluation, and reproduction of the key results presented in the paper.


## Basic Usage / Quick Start
Our Negative Cosine Scheduling can be integrated into any method using a few lines of code

```python
from temp_schedulers import GCosineTemperatureSchedulerM

if(args.temperature_scheduling):
    TS=GCosineTemperatureSchedulerM()
for epoch in range(1,N_epochs+1):
    if(args.temperature_scheduling):
        criterion.temperature = TS.get_temperature(epoch)
    # rest of the code
    ... ...
```

### Training

Basic commands to start training are available at bash_scripts

### Credit
This implementation is built upon from the following repositories:
1. [osr_closed_set_all_you_need](https://github.com/sgvaze/osr_closed_set_all_you_need) by Vaze et al. [2022]: data loading pipeline, base model architecture, evaluation
2. [SupContrast](https://github.com/HobbitLong/SupContrast) by Khosla et al. [2021]: We took the supervised contrastive loss from here.
