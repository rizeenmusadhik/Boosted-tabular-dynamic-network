# Boosted-Dynamic-Networks
[Haichao Yu](mailto:haichao.yu@outlook.com), [Haoxiang Li](http://blog.haoxiang.org/haoxiang.html), [Gang Hua](https://www.ganghua.org), [Gao Huang](http://www.gaohuang.net), [Humphrey Shi](https://www.humphreyshi.com)

This repository is the official implementation for our paper [Boosted Dynamic Neural Networks](https://arxiv.org/abs/2211.16726). In the paper, we propose a new early-exiting dynamic neural network (EDNN) architecture, where we formulate an EDNN as an additive model inspired by gradient boosting, and propose multiple training techniques to optimize the model effectively. Our experiments show it achieves superior performance on CIFAR100 and ImageNet datasets in both anytime and budgeted-batch prediction modes.

**NEW: Tabular Data Support!** This repository now supports tabular data in addition to image data. The MSDNet and RANet architectures have been adapted to work with fully connected layers for tabular classification tasks.

![Framework](figures/arch.png)

## Features

### Image Classification
- **MSDNet**: Multi-Scale Dense Networks for image classification
- **RANet**: Resolution Adaptive Networks for efficient image processing
- Support for CIFAR-10, CIFAR-100, and ImageNet datasets

### Tabular Data Classification
- **TabularMSDNet**: Fully connected version of MSDNet for tabular data
- **TabularRANet**: Fully connected version of RANet for tabular data
- Support for custom tabular datasets and synthetic data generation
- **All data processes through all blocks** - no early exiting for tabular data

## Results in Anytime Prediction Mode
| MSDNet on CIFAR100 | MSDNet on ImageNet | RANet on CIFAR100 | RANet on ImageNet |
|:---:|:---:|:---:|:---:|
| [![](figures/cifar100_any_msdnet.png)]()  | [![](figures/imagenet_any_msdnet.png)]() | [![](figures/cifar100_any_ranet.png)]() | [![](figures/imagenet_any_ranet.png)]()|

## Results in Budgeted-batch Prediction Mode
| MSDNet on CIFAR100 | MSDNet on ImageNet | RANet on CIFAR100 | RANet on ImageNet |
|:---:|:---:|:---:|:---:|
| [![](figures/cifar100_dynamic_msdnet.png)]()  | [![](figures/imagenet_dynamic_msdnet.png)]() | [![](figures/cifar100_dynamic_ranet.png)]() | [![](figures/imagenet_dynamic_ranet.png)]()|

## Usage 

### Image Classification
Please use the scripts in `msdnet_scripts/` and `ranet_scripts/` for model training and evaluation. For ImageNet experiments, please first download the dataset and put it into the proper folder.

### Tabular Data Classification

#### Quick Start
```bash
# Check device availability
python check_device.py

# Run the example script
python example_tabular.py

# Train with custom parameters
python train_tabular.py --dataset tabular --num_features 100 --num_classes 2 --arch msdnet
```

#### Supported Tabular Datasets
- **Custom**: Generate synthetic data with specified number of features and classes
- **Adult**: Adult income prediction dataset (real data from OpenML, ~48K samples, 14 features, 2 classes)
- **HELOC**: Home Equity Line of Credit dataset (real data from OpenML, ~10K samples, 23 features, 2 classes)
- **Covertype**: Forest cover type prediction (real data from OpenML, ~580K samples, 54 features, 7 classes)
- **Credit**: Credit card fraud detection (simulated data, 284K samples, 30 features, 2 classes)  
- **Diabetes**: Diabetes prediction (simulated data, 768 samples, 8 features, 2 classes)

#### Key Parameters for Tabular Data
- `--dataset tabular`: Enable tabular data mode
- `--num_features`: Number of input features
- `--num_classes`: Number of output classes
- `--arch`: Architecture choice (`msdnet` or `ranet`)
- `--nChannels`: Base number of channels/neurons
- `--nBlocks`: Number of network blocks
- `--nScales`: Number of scales for multi-scale processing

**Note**: For tabular data, all samples process through all blocks without early exiting. This ensures consistent processing and may improve accuracy for complex tabular datasets.

### Device Support
- **Automatic Device Selection**: The code automatically uses GPU if available, otherwise falls back to CPU
- **Cross-Platform**: Works seamlessly on both GPU and CPU-only systems
- **No Manual Configuration**: Device selection is handled automatically

## Citation
```
@article{yu2022boostdnn,
	title        = {Boosted Dynamic Neural Networks},
	author       = {Yu, Haichao and Li, Haoxiang and Hua, Gang and Huang, Gao and Shi, Humphrey},
	year         = 2022,
	url          = {https://arxiv.org/abs/2211.16726},
	eprint       = {2211.16726},
	archiveprefix = {arXiv},
	primaryclass = {cs.LG}
}
```

## Acknowledgments
This repository is built based on previous open-sourced efforts:
* [MSDNet-PyTorch](https://github.com/kalviny/MSDNet-PyTorch)
* [IMTA](https://github.com/kalviny/IMTA)
* [RANet-pytorch](https://github.com/yangle15/RANet-pytorch)
