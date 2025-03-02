# LWTGPF-Unsupervised Learning-Person Re-ID
Unsupervised person Re-ID has gained more favor due to its potential for label-free applications. The core of unsupervised learning is to extract deep representations of persons from a backbone that are then generated pseudo-labels by the standard clustering methods. However, the changes of cameras and the ill-posed nature of person features lead to noisy pseudo-labels. To this end, this work proposes a novel unsupervised learning system, called Local enhancing Window vision Transformer with camera-aware based on Global-Part Features (LWTGPF). LWTGPF consists of 2 new components: the Local Enhancing Window Vision Transformer (LewinVIT) and the Global-Part Off/On-line associated Camera-Aware Proxy learning framework (GP-O2CAP). LewinVIT captures the contextual semantics of the image through a hierarchical structure and models the global-local relationships of the image via window-based attention. Moreover, we create a copy of the final stage of LewinVIT to construct a dual-branch structure for building fine-grained features of persons. Correspondingly, GP-O2CAP fine-tunes the top-performing unsupervised learning framework, O2CAP, where both global/part-level features are used to obtain the proxy-level centroids. Off/on-line contrastive learning losses are redefined to achieve end-to-end unsupervised learning. Extensive experiments on 4 standard and occluded person Re-ID benchmarks demonstrate that LWTGPF achieves state-of-the-art performance in unsupervised methods and further narrows the gap with supervised learning.

# Contents
- [Preliminary🔧](#Preliminary)
- [Pretrain📂](#Weights)
- [Datasets🔔](#Datasets)
- [How to Run💻](#Run)
- [Acknowledegements🙏](#Acknowledegements)
- [License🔑](#License)


# Preliminary🔧
**Overall framework**  
![image](assets/overall.jpg)
**It seems to be successful only under Linux**
```
pip install faiss-gpu 
git clone https://github.com/YanJieWen/LWTGPF-2025.git
```

# Pretrain📂
## Luperson prertrained
[![Swin](https://img.shields.io/badge/Swin_TF-LP-blue)](https://github.com/tinyvision/SOLIDER?tab=readme-ov-file)  
[![Res50](https://img.shields.io/badge/Res50-CNN-blue)](https://github.com/DengpanFu/LUPerson)  
[![VIT](https://img.shields.io/badge/VIT-TF-blue)](https://github.com/damo-cv/TransReID-SSL/tree/main)

## Imagenet pretrained
[![Img](https://img.shields.io/badge/Image-Net-green)](https://github.com/huggingface/pytorch-image-models-timm)

Putting them into the 📂[pretrain](pretrain)-->generate `imagenet` and `luperson`


# Datasets🔔
We provide 4 banchmarks in [configs](configs), before run our project, please donwload datasets and put them into the [benchmarks](benchmarks)  
[Market1501](https://www.kaggle.com/datasets/pengcw1/market-1501/data)  
[DukeMTMC](https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID)  
[CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)  
[OccDuke](https://github.com/lightas/Occluded-DukeMTMC-Dataset)


# How to Run💻
## Training
**After installing the data and weights**  
change the [train.py](train.py) of `line 31`  corresponding to the [configs](configs)
```
python train.py
```
After training, a new folder called `runs` is created. In this folder, you can get the `logs` folder, the Training log and weights are preserved.

## Evaluating
change the [predict.py](predict.py) of `line 36` & `line 37`  Run
```
python predict.py
```
**We suggest run our model based on the trained weights:**  
[![Best](https://img.shields.io/badge/LWTGPF-best-red)](https://drive.google.com/drive/folders/1LWs-UuSQD0128wRyH7OdaFKqqt0Aoxbr?usp=drive_link)




## Analysis
We provide methods based on attention maps and Grad-CAM maps in [heatmap_visual.py](heatmap_visual.py), which are implemented in [swin_rollout.py](Csu/tools/swin_rollout.py) and [grad_cam.py](Csu/tools/grad_cam.py). You need to modify the parameters from `line 25` to `line 30`.  
```
python heatmap_visual.py
```



# Acknowledegements🙏
We sincerely thank the excellent works:  

[TMGF](https://github.com/RikoLi/WACV23-workshop-TMGF)  
[O2CAP](https://github.com/Terminator8758/O2CAP)  
[SOLIDER](https://github.com/tinyvision/SOLIDER)


# License🔑
[MIT](LICENSE) © YanjieWen

