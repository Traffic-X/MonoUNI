## MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues

:fire::fire:**[NeurIPS 2023]** The official implementation of the paper "[MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues](https://openreview.net/pdf?id=v2oGdhbKxi)"

:fire::fire:|[Paper](https://openreview.net/pdf?id=v2oGdhbKxi) | [MonoUNI微信解读](https://mp.weixin.qq.com/s/NpLjZT2yuiV-dhIyTcdYRw)

 <div align=center> <img title='MonoUNI' src="imgs/MonoUNI_Poster.png"> </div>

## Introduction
In this paper, by taking into account thediversity of pitch angles and focal lengths, we propose a unified optimization targetnamed normalized depth, which realizes the unification of 3D detection problemsfor the two sides. Furthermore, to enhance the accuracy of monocular 3D detection,3D normalized cube depth of obstacle is developed to promote the learning ofdepth information.  We posit that the richness of depth clues is a pivotal factorimpacting the detection performance on both the vehicle and infrastructure sides. Aricher set of depth clues facilitates the model to learn better spatial knowledge, andthe 3D normalized cube depth offers sufficient depth clues. Extensive experimentsdemonstrate the effectiveness of our approach.  Without introducing any extrainformation, our method, named MonoUNI, achieves state-of-the-art performanceon five widely used monocular 3D detection benchmarks, including Rope3D and DAIR-V2X-I for the infrastructure side, KITTI and Waymo for the vehicle side,and nuScenes for the cross-dataset evaluation.

## News

- [x] create repo
- [ ] release init train/val code
- [ ] support Rope3D dataset
- [ ] support DAIR-V2X-C dataset
- [ ] support KITTI dataset


## Dataset
- Download the KITTI dataset from [**KITTI website**](https://www.cvlibs.net/datasets/kitti/index.php)
- Download the Rope3D dataset from [**Rope3D website**](https://thudair.baai.ac.cn/rope)
- Download the DAIR-V2X-C dataset from [**DAIR-V2X-C website**](https://thudair.baai.ac.cn/rope)

## Installation
Install the following environments:
~~~
python 3.7
torch 1.3.1
torchvision 0.4.2
~~~

## Weight
Download the checkpoint from [**here**](https://pan.baidu.com/s/13H8CJzwuDISGR4q6MRg3sg?pwd=g86j)

## Train
~~~
bash train.sh
~~~

## citation
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
This respository is mainly based on [**GUPNET**](https://github.com/SuperMHP/GUPNet/tree/main), [**DID-M3d**](https://github.com/SPengLiang/DID-M3D) and [**MonoLSS**](https://github.com/Traffic-X/MonoLSS)
