# DSEBM

Implementation of Deep Structured Energy Based Models(http://proceedings.mlr.press/v48/zhai16.pdf).

I couldn't find the detail of experiment in the paper (mainly around dataset settings). 


# Usage
## Create Dataset

Create a dataset which has inlier images and outlier images.
```bash
 python dataset mnist_splitter.py --inlier 2
```
## Train

```bash
 python train.py
```
