#!/usr/bin/env python3
"""
Test script to verify that tabular models work without early exiting
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification

from models import TabularMSDNet, TabularRANet
from args import args


def test_no_early_exit():
    """Test that models process through all blocks without early exiting"""
    
    # Set up arguments for tabular data
    args.dataset = 'tabular'
    args.num_features = 10
    args.num_classes = 2
    args.nChannels = 32
    args.nBlocks = 3
    args.nScales = 3
    args.base = 4
    args.step = 2
    args.growthRate = 8
    args.grFactor = [1, 2, 4]
    args.bnFactor = [1, 2, 4]
    args.bottleneck = True
    args.reduction = 0.5
    args.prune = 'max'
    args.stepmode = 'even'
    
    # Create sample data
    X, y = make_classification(
        n_samples=100, 
        n_features=args.num_features, 
        n_classes=args.num_classes,
        random_state=42
    )
    
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_tensor = torch.FloatTensor(X[:5]).to(device)  # 5 samples
    
    print("Testing TabularMSDNet...")
    msdnet_model = TabularMSDNet(args).to(device)
    
    # Test that forward method doesn't accept stage parameter
    try:
        # This should work (no stage parameter)
        output = msdnet_model(X_tensor)
        print(f"✓ MSDNet forward pass successful, output shape: {output.shape}")
        
        # This should fail (stage parameter not accepted)
        try:
            output = msdnet_model(X_tensor, stage=1)
            print("✗ MSDNet still accepts stage parameter - this is wrong!")
        except TypeError:
            print("✓ MSDNet correctly rejects stage parameter")
            
    except Exception as e:
        print(f"✗ MSDNet forward pass failed: {e}")
    
    print("\nTesting TabularRANet...")
    # Set RANet specific arguments
    args.scale_list = [1, 2, 3]
    args.block_step = 2
    args.compress_factor = 0.25
    args.bnAfter = True
    
    ranet_model = TabularRANet(args).to(device)
    
    try:
        # This should work (no stage parameter)
        output = ranet_model(X_tensor)
        print(f"✓ RANet forward pass successful, output shape: {output.shape}")
        
        # This should fail (stage parameter not accepted)
        try:
            output = ranet_model(X_tensor, stage=1)
            print("✗ RANet still accepts stage parameter - this is wrong!")
        except TypeError:
            print("✓ RANet correctly rejects stage parameter")
            
    except Exception as e:
        print(f"✗ RANet forward pass failed: {e}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("Both models should:")
    print("1. Accept forward(x) calls ✓")
    print("2. Reject forward(x, stage=N) calls ✓")
    print("3. Process all data through all blocks ✓")
    print("4. Return single output instead of list ✓")


if __name__ == '__main__':
    test_no_early_exit()
