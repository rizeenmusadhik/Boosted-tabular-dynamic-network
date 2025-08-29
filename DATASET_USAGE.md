# Tabular Dataset Usage Guide

This guide explains how to use the real tabular datasets (Adult Income, HELOC, and Covertype) with the Boosted Dynamic Networks codebase.

## Available Datasets

### 1. Adult Income Dataset
- **Source**: OpenML (ID: 1590)
- **Task**: Predict whether income exceeds $50K/year
- **Samples**: ~48,842
- **Features**: 14 (mix of numerical and categorical)
- **Classes**: 2 (≤50K, >50K)
- **Use Case**: Income prediction, demographic analysis

### 2. HELOC Dataset
- **Source**: OpenML (ID: 45066)
- **Task**: Predict credit risk (Good vs Bad)
- **Samples**: ~10,459
- **Features**: 23 (numerical)
- **Classes**: 2 (Good, Bad)
- **Use Case**: Credit risk assessment, financial modeling

### 3. Covertype Dataset
- **Source**: OpenML (ID: 150)
- **Task**: Predict forest cover type
- **Samples**: ~581,012
- **Features**: 54 (numerical)
- **Classes**: 7 (different forest cover types)
- **Use Case**: Land classification, environmental modeling

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install OpenML and other required packages.

## Device Support

The code automatically handles device selection:
- **GPU**: Automatically used if CUDA is available
- **CPU**: Automatically used if no GPU is available
- **No Configuration**: Works out of the box on any system

To check your device setup:
```bash
python check_device.py
```

## Quick Start

### Test Dataset Loading

To verify that all datasets can be loaded correctly:

```bash
python test_dataset_loading.py
```

This script will attempt to load all datasets and display their properties.

### Run Example with All Datasets

To test the models on all datasets:

```bash
python example_tabular.py
```

This will test both TabularMSDNet and TabularRANet on each dataset.

### Train on Real Datasets

To train models on the real datasets:

```bash
python train_real_datasets.py
```

This will train both models on Adult, HELOC, and Covertype datasets.

## Command Line Usage

### Using the Training Script

```bash
# Train on Adult dataset
python train_tabular.py --dataset tabular --tabular_dataset adult --arch msdnet

# Train on HELOC dataset
python train_tabular.py --dataset tabular --tabular_dataset heloc --arch ranet

# Train on Covertype dataset
python train_tabular.py --dataset tabular --tabular_dataset covertype --arch msdnet
```

### Custom Dataset

```bash
# Use custom synthetic data
python train_tabular.py --dataset tabular --tabular_dataset custom --num_features 50 --num_classes 3 --arch msdnet
```

## Dataset Characteristics

### Adult Income
- **Categorical Features**: Workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Numerical Features**: Age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- **Preprocessing**: Categorical variables are label-encoded, all features are standardized

### HELOC
- **Features**: All numerical, representing various financial ratios and metrics
- **Preprocessing**: Missing values filled with median, all features standardized
- **Note**: This dataset has some missing values that are automatically handled

### Covertype
- **Features**: All numerical, representing various terrain and environmental factors
- **Preprocessing**: All features standardized
- **Note**: This is a multi-class problem with 7 different forest cover types

## Data Preprocessing

All datasets automatically undergo the following preprocessing:

1. **Categorical Encoding**: Categorical variables are label-encoded
2. **Missing Value Handling**: Missing values are filled with median (for numerical) or mode (for categorical)
3. **Feature Scaling**: All features are standardized using StandardScaler
4. **Data Splitting**: 70% train, 15% validation, 15% test

## Model Configuration

The models automatically adapt to each dataset:

- **Input Features**: Automatically set based on dataset
- **Output Classes**: Automatically set based on dataset
- **Architecture**: Configurable via command line arguments

## Troubleshooting

### OpenML Connection Issues
If you encounter connection issues with OpenML:

```bash
# Set OpenML cache directory
export OPENML_CACHE_DIR=./openml_cache

# Or use a different OpenML server
export OPENML_SERVER=https://test.openml.org/api/v1/xml
```

### Memory Issues
For large datasets like Covertype, you might need to:

1. Reduce batch size: `--batch-size 64`
2. Use fewer model blocks: `--nBlocks 2`
3. Reduce model complexity: `--nChannels 32`

### Dataset-Specific Issues

#### Adult Income
- If you get encoding errors, the dataset might have changed. Check OpenML for updates.

#### HELOC
- Missing values are automatically handled, but you might want to investigate the pattern.

#### Covertype
- This is a large dataset. Consider using a subset for initial testing.

## Performance Expectations

Based on typical performance on these datasets:

- **Adult Income**: 80-85% accuracy
- **HELOC**: 70-75% accuracy  
- **Covertype**: 85-90% accuracy

Actual performance may vary depending on:
- Model architecture configuration
- Training parameters
- Data preprocessing choices
- Random seed

## Advanced Usage

### Custom Preprocessing
You can modify the preprocessing in `dataloader.py`:

```python
def load_adult_income():
    # ... existing code ...
    
    # Add your custom preprocessing here
    # e.g., feature selection, different encoding, etc.
    
    return X_encoded, y
```

### Adding New Datasets
To add a new dataset:

1. Add the dataset choice to `args.py`
2. Create a loading function in `dataloader.py`
3. Add the case to `create_tabular_data()`
4. Update the README and this guide

## Support

If you encounter issues:

1. Check the error messages carefully
2. Verify OpenML connectivity
3. Check dataset availability on OpenML
4. Review the preprocessing steps in the code

The datasets are loaded directly from OpenML, so they should always be available and up-to-date.
