#!/usr/bin/env python3
"""
Example script for training tabular data with Boosted Dynamic Networks
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import TabularMSDNet, TabularRANet
from args import args


def create_sample_data(n_samples=1000, n_features=20, n_classes=2):
    """Create sample tabular data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def train_simple_model(model, X_train, y_train, epochs=50, lr=0.01):
    """Simple training loop"""
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)
        
        accuracy = (predictions == y_test_tensor).float().mean().item()
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy


def main():
    # Set up arguments for tabular data
    args.dataset = 'tabular'
    args.nChannels = 64
    args.nBlocks = 3
    args.nScales = 3
    args.base = 4
    args.step = 2
    args.growthRate = 16
    args.grFactor = [1, 2, 4]
    args.bnFactor = [1, 2, 4]
    args.bottleneck = True
    args.reduction = 0.5
    args.prune = 'max'
    args.stepmode = 'even'
    
    # Test with different datasets
    test_datasets = [
        ('custom', 20, 2),
        ('adult', None, None),  # Will be set automatically
        ('heloc', None, None),  # Will be set automatically
        ('covertype', None, None)  # Will be set automatically
    ]
    
    for dataset_name, num_features, num_classes in test_datasets:
        print(f"\n{'='*60}")
        print(f"Testing with dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Set dataset-specific arguments
        args.tabular_dataset = dataset_name
        if num_features:
            args.num_features = num_features
        if num_classes:
            args.num_classes = num_classes
        
        try:
            # Load the dataset
            X_train, X_val, X_test, y_train, y_val, y_test = create_tabular_data(args)
            
            print(f"Dataset loaded successfully!")
            print(f"Features: {args.num_features}, Classes: {args.num_classes}")
            print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Test TabularMSDNet
            print(f"\nTesting TabularMSDNet...")
            msdnet_model = TabularMSDNet(args)
            print(f"MSDNet parameters: {sum(p.numel() for p in msdnet_model.parameters()):,}")
            
            # Quick training test
            train_simple_model(msdnet_model, X_train, y_train, epochs=5)
            msdnet_accuracy = evaluate_model(msdnet_model, X_test, y_test)
            
            # Test TabularRANet
            print(f"\nTesting TabularRANet...")
            # Set RANet specific arguments
            args.scale_list = [1, 2, 3]
            args.block_step = 2
            args.compress_factor = 0.25
            args.bnAfter = True
            
            ranet_model = TabularRANet(args)
            print(f"RANet parameters: {sum(p.numel() for p in ranet_model.parameters()):,}")
            
            # Quick training test
            train_simple_model(ranet_model, X_train, y_train, epochs=5)
            ranet_accuracy = evaluate_model(ranet_model, X_test, y_test)
            
            print(f"\nResults for {dataset_name}:")
            print(f"  MSDNet Accuracy: {msdnet_accuracy:.4f}")
            print(f"  RANet Accuracy: {ranet_accuracy:.4f}")
            
        except Exception as e:
            print(f"Failed to process {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("ALL DATASET TESTS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
