<table>
<tr>
<td align="left">
  <h1 style="margin: 0;">
    <em>IEEE TMI 2025</em><br>
    HFF: Rethinking Brain Tumor Segmentation from the Frequency Domain Perspective
  </h1>
</td>
<td align="right">
  <img src="figs/hff_logo.png" alt="logo" width="280">
</td>
</tr>
</table>



[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://www.arxiv.org/abs/2506.10142)
[![Overview](https://img.shields.io/badge/Overview-Read-orange.svg)](#overview)
[![1. Download](https://img.shields.io/badge/Datasets-Download-yellow.svg)](#1-download)
[![Models](https://img.shields.io/badge/Training-Evaluation-purple.svg)](#check-before-training)
[![BibTeX](https://img.shields.io/badge/BibTeX-Cite-blueviolet.svg)](#citation)


Official implementation of the IEEE Transactions on Medical Imaging paper [Rethinking Brain Tumor Segmentation from the Frequency Domain Perspective](https://www.arxiv.org/abs/2506.10142), provided by [Minye Shao](https://www.linkedin.com/in/minyeshao/).



## TL;DR

<table>
<tr>
<td>

**HFF-Net** is proposed to address the persistent **performance degradation in segmenting contrast-enhancing tumor (ET) regions in brain MRI**, a task often compromised by low inter-modality contrast, heterogeneous intensity profiles, and indistinct anatomical boundaries. Built upon a frequency-aware dual-branch architecture, HFF-Net disentangles and fuses complementary feature representations through **Dual-Tree Complex Wavelet Transform (DTCWT)** for capturing shift-invariant low-frequency structure and **Nonsubsampled Contourlet Transform (NSCT)** for extracting high-frequency, multi-directional textural details. Integrated with adaptive Laplacian convolution (ALC) and frequency-domain cross-attention (FDCA), the framework significantly enhances the discriminability of ET subregions, offering precise and robust delineation across diverse MRI modalities.

</td>
<td width="45%" align="center">

<img src="figs/problem.jpg" alt="HFF-Net Comparison" style="max-width:100%; height:auto;">

<em style="font-size:2px; color:gray; text-align:center; display:block; margin-top:5px;">
Figure: Comparison of our method and prior work in a complex glioma case,  
highlighting improved segmentation of the ET region using fused frequency-domain features.
</em>

</td>
</tr>
</table>





## Overview
<div align=center>  
<img src='figs/overview.png' width="70%">
</div>



(a) Architecture of our HFF-Net: A multimodal dual-branch network decomposing and integrating multi-directional HF and LF MRI features with three components: ALC, FDCA, and FDD. It uses **L**<sub><em>unsup</em></sub> for output consistency between branches and **L**<sub><em>sup</em></sub><sup><em>H,L</em></sup> to align each branch's main and side outputs with ground truth. (b) Our ALC uses elastic weight consolidation to dynamically update weights, maintaining HF filtering functionality while extracting features from multimodal and multi-directional inputs. (c) FDCA enhances the extraction and processing of anisotropic volumetric features in MRI images through multi-dimensional cross-attention mechanisms in the frequency domain. (d) FDD processes multi-sequence MRI slices by decomposing them into HF and LF inputs using distinct frequency domain transforms. (e) The fusion block integrates the deep HF and LF features from the deep layers during the encoding process.

<div align=center>  
<img src='figs/results.jpg' width="70%">
</div>

Some experimental results are presented below for illustration. For more details, please refer to our paper.






---

## 🛠️ Install Dependencies
> ✅ Tested on Ubuntu 22.04/24.04 + Pytorch 2.1.2

Clone this repo and install environment:
```
git clone https://github.com/VinyehShaw/HFF.git
cd HFF
conda create -n hff python=3.8
conda activate hff
pip install -r requirements.txt
```
 Install [MATLAB]( https://www.mathworks.com/downloads/), please make sure to install the [Image Processing Toolbox](https://mathworks.com/products/image-processing.html) as an additional component.

> ✅ The project has been tested with the MATLAB versions R2025a, R2024b, R2023b.

## 📦 Datasets Preparation
### 1. Download
 Datasets Preparation
Please download and prepare the following training datasets:

- [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) is now available for download on [Kaggle](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019). The training set includes all samples from both the HGG and LGG subsets.


- [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) is also now available for download on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

- [MSD](https://decathlon-10.grand-challenge.org/) Brain Tumor (Task 01_BrainTumour)  is available from via [AWS](http://medicaldecathlon.com/dataaws/) or [Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).

- To download [BraTS 2023](https://www.synapse.org/Synapse:syn51156910/wiki/621282), simply create an account on Synapse and register for the challenge on the official website (register through [BraTS-Lighthouse 2025 Challenge](https://www.synapse.org/Synapse:syn64153130/wiki/631048) now).


After downloading and extracting, the data is organized under the following structure (Here, we use BraTS 2020 as an example, same layout applies to other datasets):
```
your_data_path/
└── MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_flair.nii
    │   ├── BraTS20_Training_001_t1.nii
    │   ├── BraTS20_Training_001_t1ce.nii
    │   ├── BraTS20_Training_001_t2.nii
    │   └── BraTS20_Training_001_seg.nii
    ├── BraTS20_Training_002/
    ├── ...
    └── BraTS20_Training_369/
```
---
### 2. Frequency Decomposition

To extract the low-frequency components of MRI volumes, run the Python script ```./DTCWT_LF.py```. You only need to modify the --data_root argument to point to your dataset location. For example, for BraTS 2020:

```
python DTCWT_LF.py --data_root yourpath/MICCAI_BraTS2020_TrainingData/
```


Then, to perform high-frequency transformation on MRI volumes, use the Matlab script ```./NSCT_BTS/nsct_hf.m```
Make sure the original MRI data is correctly organized. For example, in the BraTS 2020 case:
```
baseDir = 'yourpath/MICCAI_BraTS2020_TrainingData';
nsct_tbx_dir = './NSCT_BTS/nsct_toolbox'; % Ensure this path is relative to the current working directory
```
This process may take some time, so feel free to take a break while it runs. ☕️

---
After running frequency decomposition, each subject folder is expected to have the following structure:
```
your_data_path/
└── MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_flair.nii
    │   ├── BraTS20_Training_001_flair_H1.nii.gz
    │   ├── BraTS20_Training_001_flair_H2.nii.gz
    │   ├── BraTS20_Training_001_flair_H3.nii.gz
    │   ├── BraTS20_Training_001_flair_H4.nii.gz
    │   ├── BraTS20_Training_001_flair_L.nii.gz  
    │
    │   ├── BraTS20_Training_001_t1.nii
    │   ├── BraTS20_Training_001_t1_H1.nii.gz
    │   ├── ...
    │   ├── BraTS20_Training_001_t1_L.nii.gz
    │
    │   ├── BraTS20_Training_001_t1ce.nii
    │   ├── BraTS20_Training_001_t1ce_H1.nii.gz
    │   ├── ...
    │   ├── BraTS20_Training_001_t1ce_L.nii.gz
    │
    │   ├── BraTS20_Training_001_t2.nii
    │   ├── BraTS20_Training_001_t2_H1.nii.gz
    │   ├── ...
    │   ├── BraTS20_Training_001_t2_L.nii.gz
    │
    │   └── BraTS20_Training_001_seg.nii
    ├── BraTS20_Training_002/
    ├── BraTS20_Training_003/
    ├── ...
    └── BraTS20_Training_369/
```
Run split.py with ```--brain_dir``` (e.g., your_data_path/MICCAI_BraTS2020_TrainingData/), ```--save_dir``` (where to store the split .txt files), and ```--n_split ```(number of random splits) according to your needs.




## 🚀 Training
#### Check Before Training
> Make sure the ```label_filename``` and ```m_path``` variables in ```./loader/dataload3d.py``` follow the correct naming convention with underscores (e.g., _seg.nii vs -seg.nii)
---
To start training, run ```train.py``` with the following key arguments:

```--train``` and ```val_list```: path to the training and validation .txt file  

--val_list: path to the validation .txt file

```--dataset_name```: choose from ['brats19', 'brats20', 'brats23men', 'msdbts']

```--class_type```: choose segmentation target from ['et', 'tc', 'wt', 'all']

>ET: enhancing tumor, TC: tumor core, WT: whole tumor — each defines a binary segmentation task; ALL: preserves all labels for a 4-class segmentation task.

```--display_iter```: how many iterations between validation runs 

```--selected_modal```: list of input MRI modalities (see below)
> For BraTS 2020, --selected_modal should be like:

```
['flair_L', 't1_L', 't1ce_L', 't2_L', 'flair_H1', 'flair_H2', 'flair_H3', 'flair_H4', 't1_H1', 't1_H2', 't1_H3', 't1_H4', 't1ce_H1', 't1ce_H2', 't1ce_H3', 't1ce_H4', 't2_H1', 't2_H2', 't2_H3', 't2_H4']
  ```

Modify the paths and settings based on your dataset and experiment requirements.



The model will be saved under ./result/checkpoints. The training takes about 1.5 days on a single RTX 4090.



## 📊 Evaluation

To evaluate a trained model, run ```./eval.py``` with the following arguments:

```--selected_modal```: list of input modalities (same as used during training)

```--dataset_name```: one of ['brats19', 'brats20', 'brats23men', 'msdbts']

```--class_type```: segmentation target (et, tc, wt, or all)

```--checkpoint```: path to the trained model checkpoint

```--test_list```: path to the .txt file listing MRI samples to evaluate


---


## Citation
Both NSCT and DTCWT offer powerful frequency-domain signal processing tools that enhance feature extraction and show strong potential for a wide range of downstream medical imaging tasks — **including segmentation, inpainting, generation, registration, and beyond.** We encourage researchers to explore these techniques further and apply our method to future and annual [BraTS challenges](https://www.synapse.org/Synapse:syn53708126/wiki/626320) as well as other medical imaging benchmarks.


> **If you find our work helpful in your research or clinical tool development, please consider citing us:**
```
@ARTICLE{11032150,
  author={Shao, Minye and Wang, Zeyu and Duan, Haoran and Huang, Yawen and Zhai, Bing and Wang, Shizheng and Long, Yang and Zheng, Yefeng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Rethinking Brain Tumor Segmentation from the Frequency Domain Perspective}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Frequency-domain analysis;Tumors;Brain tumors;Magnetic resonance imaging;Three-dimensional displays;Biomedical imaging;Imaging;Feature extraction;Electronic mail;Convolution;Brain tumor segmentation;Frequency domain;Multi-modal feature fusion},
  doi={10.1109/TMI.2025.3579213}}
```

## Acknowledgments

This repo is based in part on the works of [Zhou et al. (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.html) and [Ganasala et al. (JDI 2014)](https://link.springer.com/article/10.1007/s10278-013-9664-x). We thank the authors for their valuable contributions, which inspired and guided our implementation.


