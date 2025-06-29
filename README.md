# When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation


This is the official implementation of [When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation]

This implementation is based on [DeYO implementation 🔗](https://github.com/mr-eggplant/SAR).

## Environments  
You should modify [username] and [env_name] in environment.yaml, then  
> $ conda env create --file environment.yaml  

## Baselines  
[TENT 🔗](https://arxiv.org/abs/2006.10726) (ICLR 2021)  
[EATA 🔗](https://arxiv.org/abs/2204.02610) (ICML 2022)  
[CoTTA 🔗]([https://arxiv.org/abs/2204.02610](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)) (CVPR 2022)
[SAR 🔗](https://arxiv.org/abs/2302.12400) (ICLR 2023)  
[DeYO 🔗](https://openreview.net/forum?id=9w3iw8wDuE) (ICLR 2024)
[DeYO 🔗]([https://openreview.net/forum?id=9w3iw8wDuE](https://openaccess.thecvf.com/content/WACV2024/html/Marsden_Universal_Test-Time_Adaptation_Through_Weight_Ensembling_Diversity_Weighting_and_Prior_WACV_2024_paper.html)) (WACV 2024)
## Dataset
You can download ImageNet-C from a link [ImageNet-C 🔗](https://zenodo.org/record/2235448).  
After downloading the dataset, move to the root directory ([data_root]) of datasets.  
Your [data_root] will be as follows:
```bash
data_root
├── ImageNet-C
│   ├── brightness
│   ├── contrast
│   └── ...
```
## Experiment

You can run most of the experiments in our paper by  
shell: python main.py

Moreover, we also prepare code for many pairs of models for collaboration, check the args fuction in main.py file.

## Acknowledgment
The code is inspired by the [Tent 🔗](https://github.com/DequanWang/tent), [EATA 🔗](https://github.com/mr-eggplant/EATA), [SAR 🔗](https://github.com/mr-eggplant/SAR) and [DeYO 🔗](https://github.com/Jhyun17/DeYO).
