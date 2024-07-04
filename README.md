# [MICCAI 2024] Advancing H\&E-to-IHC Virtual Staining with Task-Specific Domain Knowledge for HER2 Scoring
*Qiong Peng, Weiping Lin, Yihuang Hu, Ailisi Bao, Chenyu Lian, Weiwei wei, Meng Yue, Jingxin Liu, Lequan Yu, Liansheng Wang†*
<div align=left><img width="95%" src="./overview.png"/></div>

## Installation
- Clone this repo:
```shell
git clone https://github.com/balball/TDKstain && cd TDKstain
```
- Create a conda environment and activate it:
```shell
conda create -n TDKstain python==3.9.7
conda activate TDKstain
pip install -r requirements.txt
```

## Data Preprocess
- We first construct the dataset in `[DATASET DIR]` as the following format:
```bash
[DATASET DIR]
     ├────── train_HE
                ├────── train_1.png
                ├────── train_2.png
                             :
                └────── train_N.png
     ├────── train_IHC
                ├────── train_1.png
                ├────── train_2.png
                             :
                └────── train_N.png
     ├────── test_HE
                ├────── test_1.png
                ├────── test_2.png
                            :
                └────── test_N.png
     └────── test_IHC
                ├────── test_1.png
                ├────── test_2.png
                            :
                └────── test_N.png
```
- Then, we extract HER2 domain knowledge (DAB masks and nuclei maps) from real IHC images by [`get_dab_mask.py`](preprocess/get_dab_mask.py) and [`get_nuclei_map.py`](preprocess/get_nuclei_map.py), where you need to substitute `[DATASET DIR]` to the path of your dataset directory. 
- Specifically, you can modulate parameters in these two files to better fit your own dataset.
- In this way, we can get three additional directories in `[DATASET DIR]`, including `train_IHC_dab`, `train_IHC_dab_mask` and `train_IHC_nuclei_map`.
```shell
python -u ./preprocess/get_dab_mask.py
```
```shell
CUDA_VISIBLE_DEVICES=0 python -u ./preprocess/get_nuclei_map.py
```

## Training
```shell
python -u train.py --save_epoch_freq 10 --n_epochs 50 --n_epochs_decay 50 \
--dataroot [DATASET DIR] --checkpoints_dir ./checkpoints \
--model tdk --netD basic --netG resnet_9blocks --ndf 64 --ngf 64 --n_downsampling 3 --num_D 3 \
--norm instance --dataset_mode aligned --batch_size 1 --use_tensorboard \
--coef_L1 10.0 --coef_mask 10.0 --coef_E 10.0 --coef_nuclei 10.0 --nef 128 --n_estimator_blocks 4 \
--load_size 1024 --preprocess none --display_winsize 512 --name [EXPERIMENT NAME] --gpu_ids 0
```

## Inference
```shell
python -u test.py --results_dir ./results --eval --num_test [TEST NUMBER] \
--dataroot [DATASET DIR] --checkpoints_dir ./checkpoints \
--model tdk --netG resnet_9blocks --ngf 64 --n_downsampling 3 \
--dataset_mode aligned --batch_size 1 --load_size 1024 --preprocess none \
--display_winsize 512 --serial_batches --no_flip \
--name [EXPERIMENT NAME] --epoch [EPOCH] --gpu_ids 0
```

## Well-trained Models
We provide two well-trained models of our work in [`saved_checkpoints`](saved_checkpoints/). You can use these models to implement inference by running the command in [**Inference**](#inference), where you need to substitute `[EXPERIMENT NAME]` to `model1` or `model2`, and `[EPOCH]` to `latest`.


## Note
- Our implementation builds on the following publicly available codes.
  - [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) - Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
  - [Pix2pixHD](https://github.com/NVIDIA/pix2pixHD) - Wang, Ting-Chun, et al. "High-resolution image synthesis and semantic manipulation with conditional gans." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
  - [Decent](https://github.com/Mid-Push/Decent) - Xie, Shaoan, Qirong Ho, and Kun Zhang. "Unsupervised image-to-image translation with density changing regularization." Advances in Neural Information Processing Systems 35 (2022): 28545-28558.

## Citation
Please cite our paper if this work is helpful to your research.

## Contact
If you have any questions, please contact Qiong Peng (qpeng@stu.xmu.edu.cn).