# Discrete-Constrained Regression for Local Counting Models (dcreg)
This repository contains the original pytorch code for our paper "Discrete-Constrained Regression for Local Counting Models" [[arxiv]](https://arxiv.org/abs/2207.09865) in ECCV 2022.

## Prepare environment
```
conda env create -f requirements.yaml
```

## Activate environment
```
conda activate dcreg
```

## Download raw dataset
Raw JHU dataset could be obtained from the [link](http://www.crowd-counting.com/).

## preprocess dataset
Run jhu_main_final.m in Matlab.

## Organize files
```
--> jhu_crowd_v2.0 (raw dataset)
-->data
   -->JHU_resize (processed dataset)
-->Models
   --> JHU
      --> best_epoch.pth
```
## For training
```
sh train.sh
```

## For testing
Download trained models from the [link](https://drive.google.com/file/d/1JyUXWd1jR8feGNZBLalu6l95pxyTk1tA/view?usp=sharing) and put the file in "Models/JHU".
```
sh test.sh
```
You will get MAE 64.361 and MSE 281.078.

## Sythesized Cell Dataset
A sample of synthesized dataset could be accessed from the [link](https://drive.google.com/file/d/1VxknXdOksXyf_xa_y8FVl1ZPCNgktYsm/view?usp=sharing).
More sythesized cell image could be generated with the code in the "generate_simulated_dataset.zip" file.


## References
If you find this work or code useful for your research, please cite:
```
@inproceedings{xhp2022dcreg,
  title={Discrete-Constrained Regression for Local Counting Models},
  author={Xiong, Haipeng  and Yao, Angela},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022},
  pages = {XXXX-XXXX}
}
```
