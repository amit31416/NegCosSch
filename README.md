# Boosting Open Set Recognition Performance through Modulated Representation Learning
Implementation of the paper:
### Boosting Open Set Recognition Performance through Modulated Representation Learning
Accepted at ICLR 2026 | OpenReview: (https://openreview.net/pdf?id=vpBKry7kL5)


## Overview


This research explores the impact of temperature modulation on representation learning within open-set scenarios. We introduce a suite of novel temperature scheduling (TS) strategies — most notably our Negative Cosine Schedule (NegCosSch).

Our schedules can boost both open set recognition (OSR) and closed set performance for many existing (OSR) loss functions, such as cross-entropy, contrastive, or ARPL loss, with zero computational overhead, providing enhancements irrespective of the OSR scoring rule, model architecture, data augmentations, even on top of label smoothing, with benefits that are more prominently realized as the number of training classes increases.

This repository contains the official implementation of this project. Our code allows for training, evaluation, and reproduction of the key results presented in the paper.

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

### Citation

If you use this code or our paper, please cite:
<pre>
@inproceedings{kundu2026boosting,
    title={Boosting open set recognition performance through modulated representation learning},
    author={Kundu, Amit Kumar and Patil, Vaishnavi and Jaja, Joseph},
    booktitle={International conference on learning representations},
    year={2026}
}
</pre>
    
### Credit
This implementation is built upon from the following repositories:
1. [osr_closed_set_all_you_need](https://github.com/sgvaze/osr_closed_set_all_you_need) by Vaze et al. [2022]: data loading pipeline, base model architecture, evaluation
2. [SupContrast](https://github.com/HobbitLong/SupContrast) by Khosla et al. [2021]: We took the supervised contrastive loss from here.
