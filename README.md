## MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues

:fire::fire:**[NeurIPS 2023]** The official implementation of the paper "[MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues](https://openreview.net/pdf?id=v2oGdhbKxi)"

:fire::fire:| [Paper](https://openreview.net/pdf?id=v2oGdhbKxi) | [MonoUNI微信解读](https://mp.weixin.qq.com/s/NpLjZT2yuiV-dhIyTcdYRw)

 <div align=center> <img title='MonoUNI' src="imgs/MonoUNI_Poster.png"> </div>

## Introduction
In this paper, by taking into account thediversity of pitch angles and focal lengths, we propose a unified optimization targetnamed normalized depth, which realizes the unification of 3D detection problemsfor the two sides. Furthermore, to enhance the accuracy of monocular 3D detection,3D normalized cube depth of obstacle is developed to promote the learning ofdepth information.  We posit that the richness of depth clues is a pivotal factorimpacting the detection performance on both the vehicle and infrastructure sides. Aricher set of depth clues facilitates the model to learn better spatial knowledge, andthe 3D normalized cube depth offers sufficient depth clues. Extensive experimentsdemonstrate the effectiveness of our approach.  Without introducing any extrainformation, our method, named MonoUNI, achieves state-of-the-art performanceon five widely used monocular 3D detection benchmarks, including Rope3D and DAIR-V2X-I for the infrastructure side, KITTI and Waymo for the vehicle side,and nuScenes for the cross-dataset evaluation.

## News
- [x] ***[20231016]*** create repo
- [x] ***[20240314]*** create reporelease Rope3D dataset which is merged into 4 categries and do ROI filter for eval
- [x] ***[20240315]*** release 3D Cube Depth data proposed by MonoUNI (contained in Rope3D dataset together)
- [x] ***[20240326]*** release init train/val code
- [x] ***[20240326]*** release MonoUNI checkpoint on Rope3D
- [x] ***[20240326]*** support Rope3D dataset
- [ ] support DAIR-V2X-I dataset
- [ ] support KITTI dataset

## Installation
a. Clone this repository.
~~~
git clone https://github.com/Traffic-X/MonoUNI
~~~

b. Install the dependent libraries as follows:
* Create a new env with conda
~~~
conda create -n rope3d python=3.8
~~~

* Activate the env
~~~
conda activate rope3d
~~~

* Install the dependent python libraries:
~~~
pip install torch==1.5.0 torchvision==0.6.0 numpy==1.23.5 numba==0.58.1 scikit-image==0.21.0 opencv-python==3.4.10.37 tqdm==4.65.0 matplotlib==3.7.1 protobuf==4.22.1 pyyaml==6.0
~~~

## Dataset
- [x] Download the official Rope3D dataset from [**Here**](https://pan.baidu.com/s/1Tt014qMNcDxAMCkEWH_EZQ?pwd=d1yd).  
    ~~~
    tar -zxvf Rope3D_data.tar.gz
    ~~~
    The directory will be as follows:  
    Rope3D_data  
    ├── box3d_depth_dense  
    ├── calib  
    ├── denorm  
    ├── extrinsics  
    ├── image_2  
    ├── ImageSets  
    ├── label_2  
    ├── label_2_4cls_filter_with_roi_for_eval  
    └── label_2_4cls_for_train  

- [ ] Support the DAIR-V2X-I dataset
- [ ] Support the KITTI dataset

## Train
- [x] Rope3D dataset 

    modify the 'root_dir' in config.yaml, use your own path to the downloaded 'Rope3D_data'
    ~~~
    bash train.sh
    ~~~
- [ ] DAIR-V2X-I dataset
- [ ] KITTI dataset

## Eval
- [x] Rope3D dataset  

    modify the 'root_dir' in config.yaml, use your own path to the downloaded 'Rope3D_data'  
    modify the 'resume_model' in config.yaml (tester), use your own path to checkpoint
    ~~~
    bash eval.sh
    ~~~
- [ ] DAIR-V2X-I dataset
- [ ] KITTI dataset

## Weight
Download the checkpoint (Rope3D) from [**here**](https://pan.baidu.com/s/13H8CJzwuDISGR4q6MRg3sg?pwd=g86j)

## citation
If you find MonoUNI useful in your research, please consider giving a star ⭐ and citing:
~~~
@inproceedings{jia2023monouni,
title={MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues},
author={Jinrang Jia and Zhenjia Li and Yifeng Shi},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=v2oGdhbKxi}
}
~~~
## Acknowledgements
Many thanks to following codes that help us a lot in building this codebase:
- [GUPNet](https://github.com/SuperMHP/GUPNet/tree/main) 
- [DID-M3D](https://github.com/SPengLiang/DID-M3D)
- [MonoLSS](https://github.com/Traffic-X/MonoLSS)
- [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight)


