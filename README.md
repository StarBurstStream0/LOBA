## The offical PyTorch code for paper 
[""Bridging the Task-domain Gap in Pre-training Remote Sensing Object Detectors"", TCSVT 2026.](https://ieeexplore.ieee.org/document/11701310)

##### Author: Zicong Zhu

![Supported Python versions](https://img.shields.io/badge/python-3.7-blue.svg)
![Supported OS](https://img.shields.io/badge/Supported%20OS-Linux-yellow.svg)
![npm License](https://img.shields.io/npm/l/mithril.svg)
<a href="https://pypi.org/project/mitype/"><img src="https://img.shields.io/pypi/v/mitype.svg"></a>

```bash
#### News:
#### 2026.09.22: The paper is published on IEEE TCSVT!
#### 2026.09.30: code of LOBA is open source!
```

## INTRODUCTION

We proposes LOBA, a pre-training method specifically designed for remote sensing oriented object detection that combines a General Localization Representation (GLR) branch, which converts oriented bounding boxes into adaptive multi-scale Gaussian kernel representations for general localization learning, with an Object-aware Region Reconstruction (ORR) branch, which adaptively controls masked regions based on target geometric characteristics, thereby achieving state-of-the-art transfer performance with less data.

##
## LOBA
### Pretraining framework

<div align="center">
  <img src="resources/LOBA_framework.png" width="600"/>
</div>

The working pipeline of our LOBA. It consists of two branches, GLR and ORR, both in the pre-training phase. They provide the model with
supervision for object localization and perception, effectively eliminating the task-domain bias between pre-training and fine-tuning.

### Installation

```shell
CUDA >= 9.2
GCC >= 5
Python == 3.8
PyTorch == 1.13.0
mmcv-full
mmdet
mmrotate
```

### weights

the pretrained weights can be found here.

- [Swin-Tiny-LOBA](https://pan.baidu.com/s/1-LfDNSRMQxCxBXyxsqmvKQ?pwd=btgb)

**当前实验均在ORPN，Swin-T，Focus-v8.0-ISP-e12 下完成**

| No. | Pretrain                | Pre data    | MPS/PPS | Load part | DOTA-v1.0 | FAIR1M-v2.0 | DIOR-R | HRSC2016 | UCAS-AOD | Status |
| --- | ----------------------- | ----------- | ------- | --------- | --------- | ----------- | ------ | -------- | -------- | ------ |
| 0.0 | RandInit                | -           | -       | BB        | 59.2      | 35.0        | 49.3   | 49.2     | 77.4     |        |
| 1.0 | ISP-ImageNet-?          | ImageNet    | -       | BB        | 73.9      | 42.3        | 65.7   | 89.8     | 89.6     |        |
| 2.0 | SimMIM-ImageNet-1k-100e | ImageNet-1k | -       | BB        | 71.6      | 39.2        | 60.2   | 87.8     | 88.9     |        |
| 4.0 | Focus-v8.0-ISP-e12      | FDD-v4.0    | adp     | BB+FPN    | 73.4      | 42.5        | 66.0   | 88.4     | 89.5     |        |

**当前实验均在ORPN，Swin-T，Focus-v8.0-ISP-e24 下完成**

| No. | Pretrain                | Pre data    | MPS/PPS | Load part | DOTA-v1.0 | FAIR1M-v2.0 | DIOR-R | HRSC2016 | UCAS-AOD | Status |
| --- | ----------------------- | ----------- | ------- | --------- | --------- | ----------- | ------ | -------- | -------- | ------ |
| 0.0 | RandInit                | -           | -       | BB        | 59.2      | 35.0        | 49.3   | 49.2     | 77.4     |        |
| 1.0 | ISP-ImageNet-?          | ImageNet    | -       | BB        | 73.9      | 42.3        | 65.7   | 89.8     | 89.6     |        |
| 2.0 | SimMIM-ImageNet-1k-100e | ImageNet-1k | -       | BB        | 71.6      | 39.2        | 60.2   | 87.8     | 88.9     |        |
| 4.0 | Focus-v8.0-ISP-e12      | FDD-v4.0    | adp     | BB+FPN    | 73.4      | 42.5        | 66.0   | 88.4     | 89.5     |        |
| 5.0 | Focus-v8.0-ISP-e24      | FDD-v4.0    | adp     | BB+FPN    | 74.3      |             | 67.0   | 89.9     |          |        |

- [Intern-Tiny-LOBA](https://pan.baidu.com/s/1SS27_Mah91JQbM5bCNsaJQ?pwd=btgb)

**当前实验均在ORPN, Intern-T，Focus-v8.0-ISP-e12 下完成**

| No. | Pretrain                | Pre data    | MPS/PPS | Load part | DOTA-v1.0 | FAIR1M-v2.0 | DIOR-R | HRSC2016 | UCAS-AOD | Status |
| --- | ----------------------- | ----------- | ------- | --------- | --------- | ----------- | ------ | -------- | -------- | ------ |
| 0.0 | RandInit                | -           | -       | BB        |           |             |        |          |          |        |
| 1.0 | ISP-ImageNet-?          | ImageNet    | -       | BB        | 73.9      | 43.6        | 65.7   | 89.8     | 89.6     |        |
| 4.0 | Focus-v8.0-ISP-e12      | FDD-v4.0    | adp     | BB+FPN    | 75.8      | 45.2        | 70.2   | 90.4     | 89.7     |        |

- [Resnet-50-LOBA](https://pan.baidu.com/s/1n26bNiS2P0sos0IvfbJuLw?pwd=btgb)

**当前实验均在ORPN, Resnet-50，Focus-v8.0-ISP-e12/24 下完成**

| No. | Pretrain                | Pre data    | MPS/PPS | Load part | DOTA-v1.0 | FAIR1M-v2.0 | DIOR-R | HRSC2016 | UCAS-AOD | Status |
| --- | ----------------------- | ----------- | ------- | --------- | --------- | ----------- | ------ | -------- | -------- | ------ |
| 0.0 | RandInit                | -           | -       | BB        | 53.4      | 31.4        | 48.1   | 35.1     | 82.3     |        |
| 1.0 | ISP-ImageNet-?          | ImageNet    | -       | BB        | 72.6      | 41.4        | 63.0   | 89.7     | 89.6     |        |
| 2.0 | Focus-v8.0-ISP-e12      | FDD-v4.0    | adp     | BB+FPN    | 73.3      | 41.4        | 65.5   | 90.1     | 89.5     |        |
| 3.0 | Focus-v8.0-ISP-e24      | FDD-v4.0    | adp     | BB+FPN    |           |             |        | 90.4     |          |        |

### training, testing, and inferencing

follow the official mmrotate framework to train and test ACDet. 

## Citation
If you feel this code helpful or use this code or dataset, please cite it as
```

@ARTICLE{11701310,
  author={Zhu, Zicong and Ni, Jingen and Kang, Jian and Feng, Yingchao and Li, Junxi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Bridging the Task-domain Gap in Pre-training Remote Sensing Object Detectors}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Training;Learning (artificial intelligence);Remote sensing;Tuning;Location awareness;Algorithms;Distance measurement;Object detection;Visualization;Remote sensing;Image pretrain;Oriented object detection;General localization;Object-aware reconstruction},
  doi={10.1109/TCSVT.2026.3735572}
}

```
