#!/usr/bin/env python3
"""
Test script to verify dataset loading for all supported tabular datasets
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from args import args
from dataloader import create_tabular_data


def test_dataset_loading():
    """Test loading of all supported datasets"""
    
    # Test datasets
    test_datasets = ['custom', 'adult', 'heloc', 'covertype', 'credit', 'diabetes']
    
    for dataset_name in test_datasets:
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Set up arguments for this dataset
            args.dataset = 'tabular'
            args.tabular_dataset = dataset_name
            
            # Set reasonable defaults for custom dataset
            if dataset_name == 'custom':
                args.num_features = 20
                args.num_classes = 2
            
            print(f"Loading {dataset_name} dataset...")
            
            # Load the dataset
            X_train, X_val, X_test, y_train, y_val, y_test = create_tabular_data(args)
            
            print(f"✓ Successfully loaded {dataset_name}")
            print(f"  - Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"  - Val set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
            print(f"  - Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            print(f"  - Number of classes: {args.num_classes}")
            print(f"  - Feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")
            print(f"  - Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            
            # Test device compatibility
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"  - Device compatibility: {device} (GPU available: {torch.cuda.is_available()})")
            
        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("DATASET LOADING TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_dataset_loading()
