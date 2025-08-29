# Boosted-Dynamic-Networks
[Haichao Yu](mailto:haichao.yu@outlook.com), [Haoxiang Li](http://blog.haoxiang.org/haoxiang.html), [Gang Hua](https://www.ganghua.org), [Gao Huang](http://www.gaohuang.net), [Humphrey Shi](https://www.humphreyshi.com)

This repository is the official implementation for our paper [Boosted Dynamic Neural Networks](https://arxiv.org/abs/2211.16726). In the paper, we propose a new early-exiting dynamic neural network (EDNN) architecture, where we formulate an EDNN as an additive model inspired by gradient boosting, and propose multiple training techniques to optimize the model effectively. Our experiments show it achieves superior performance on CIFAR100 and ImageNet datasets in both anytime and budgeted-batch prediction modes.

**NEW: Tabular Data Support!** This repository now supports tabular data in addition to image data. The MSDNet and RANet architectures have been adapted to work with fully connected layers for both tabular classification and regression tasks with automatic task detection.

![Framework](figures/arch.png)

## Features

### Image Classification
- **MSDNet**: Multi-Scale Dense Networks for image classification
- **RANet**: Resolution Adaptive Networks for efficient image processing
- Support for CIFAR-10, CIFAR-100, and ImageNet datasets

### Tabular Data Support
- **TabularMSDNet**: Fully connected version of MSDNet for tabular data
- **TabularRANet**: Fully connected version of RANet for tabular data
- **Classification & Regression**: Automatic task detection and appropriate metrics
- **Real-world Datasets**: Integration with OpenML (Adult, Covertype, California Housing)
- **Automatic Preprocessing**: Feature scaling, target normalization, data splitting
- **Robust Evaluation**: Multiple runs with standard deviation reporting
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

### Tabular Data Support

This framework supports both **classification** and **regression** tasks on tabular data with automatic task detection and appropriate metrics.

#### Quick Start
```bash
# Check device availability
python check_device.py

# Classification example (Adult dataset)
python train_tabular.py --dataset tabular --tabular_dataset adult --arch msdnet --epochs 1 --nBlocks 2

# Regression example (California Housing)
python train_tabular.py --dataset tabular --tabular_dataset california_housing --arch msdnet --epochs 1 --nBlocks 2 --lr 0.01

# Evaluate trained model
python eval_tabular.py --dataset tabular --tabular_dataset adult --arch msdnet --evaluate-from results/exp_xxx/best_model.pth
```

#### Supported Datasets

##### Classification Datasets
- **adult**: Adult income prediction (14 features, binary classification, 79.5% accuracy)
- **covertype**: Forest cover type prediction (54 features, 7-class classification, 74.1% accuracy)
- **heloc**: Home Equity Line of Credit (23 features, binary - data quality issues)
- **custom**: Synthetic classification data with configurable parameters
- **credit**: Credit card fraud simulation (30 features, binary)
- **diabetes**: Diabetes prediction simulation (8 features, binary)

##### Regression Datasets
- **california_housing**: California housing prices (8 features, continuous target, R²: 0.53)

#### Training Command Examples

##### Binary Classification
```bash
# Adult income prediction
python train_tabular.py --dataset tabular --tabular_dataset adult --arch msdnet --epochs 10 --nBlocks 3

# Custom synthetic binary classification
python train_tabular.py --dataset tabular --tabular_dataset custom --num_features 20 --num_classes 2 --arch ranet
```

##### Multi-class Classification
```bash
# Forest cover type prediction (7 classes)
python train_tabular.py --dataset tabular --tabular_dataset covertype --arch msdnet --epochs 5 --nBlocks 2
```

##### Regression
```bash
# California housing price prediction
python train_tabular.py --dataset tabular --tabular_dataset california_housing --arch msdnet --epochs 10 --lr 0.01
```

#### Evaluation Examples
```bash
# Classification evaluation (shows accuracy and AUC)
python eval_tabular.py --dataset tabular --tabular_dataset adult --arch msdnet --evaluate-from results/exp_xxx/best_model.pth

# Regression evaluation (shows MSE, RMSE, R²)
python eval_tabular.py --dataset tabular --tabular_dataset california_housing --arch msdnet --evaluate-from results/exp_xxx/best_model.pth
```

#### Key Parameters for Tabular Data
- `--dataset tabular`: Enable tabular data mode
- `--tabular_dataset`: Specific dataset name (adult, covertype, california_housing, etc.)
- `--arch`: Architecture choice (`msdnet` or `ranet`)
- `--nBlocks`: Number of network blocks (default: 2)
- `--nChannels`: Base number of neurons (default: 32)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (use 0.01 for regression, 0.1 for classification)

#### Features
- **Automatic Task Detection**: Framework automatically detects classification vs regression
- **Target Scaling**: Regression targets are automatically normalized to prevent gradient issues  
- **Feature Scaling**: All features are standardized using StandardScaler
- **Robust Evaluation**: Multiple evaluation runs with standard deviation reporting
- **Metadata Persistence**: Training configuration is saved in model checkpoints for reliable evaluation
- **Real-world Datasets**: Integration with OpenML for accessing real datasets

#### Performance Metrics
- **Classification**: Accuracy, AUC, Classification Report
- **Regression**: MSE, MAE, RMSE, R² Score

**Note**: For tabular data, all samples process through all blocks without early exiting. This ensures consistent processing and reliable performance on structured data.

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
