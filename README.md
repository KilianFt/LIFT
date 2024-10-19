# LIFT: Contrastive Learning Neuromotor Interface From Teacher

Check out our paper for more details [paper]()

## Setup
Create a conda environment
```
conda create -n lift python=3.11
conda activate lift
pip install -r requirements.txt
```

## Training
In order to train an EMG decoder, we first need to train a teacher that predicts ideal actions and at the same time simulates users with different beliefs. We train this meta teacher by running
```
python scripts/train_teacher_meta.py
```
Further, we need to pretrain the MI network on a dataset, such that it performs decently in online interactions from beginning on
```
python scripts/pretrain_mi.py
```
Now all necessary models for simulating interactive EMG decoder training are present. There are two main ways to train a decoder, in one session or in several iterative sessions. For one session run
```
python scripts/train_mi.py
```
or for several sessions run
```
python scripts/train_mi_iter.py
```
results will be automatically uploaded to weights and biases. This can be turned of in configs.py.



Citation
```

```