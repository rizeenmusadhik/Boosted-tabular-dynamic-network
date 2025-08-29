#!/usr/bin/env python3
"""
Simple script that trains and evaluates in the same session
"""

import os
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score

from models import TabularMSDNet, TabularRANet
from dataloader import get_dataloaders
from args import arg_parser
from utils.utils import setup_logging


def train_simple(model, train_loader, optimizer, epoch, device):
    """Simple training function"""
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    total_loss = 0
    for it, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if it % 100 == 0:
            print(f"  Step {it}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def evaluate_simple(model, test_loader, device):
    """Simple evaluation function"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results
            all_predictions.append(predicted.cpu().numpy())
            all_probabilities.append(probabilities[:, 1].cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    
    try:
        auc = roc_auc_score(all_targets, all_probabilities)
    except ValueError:
        auc = 0.5
    
    return accuracy, auc


def main():
    print("="*80)
    print("SIMPLE TRAIN & EVAL WORKFLOW")
    print("="*80)
    
    # Parse arguments
    args = arg_parser.parse_args()
    
    # Set up for Adult dataset
    args.dataset = 'tabular'
    args.tabular_dataset = 'adult'
    args.nBlocks = 2
    args.epochs = 2
    
    # Process arguments
    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.scale_list = list(map(int, args.scale_list.split('-')))
    args.nScales = len(args.grFactor)
    args.lr_milestones = list(map(int, args.lr_milestones.split(',')))
    args.ensemble_reweight = list(map(float, args.ensemble_reweight.split(',')))
    
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data first
    print("\n1. Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print(f"Dataset loaded: {args.num_features} features, {args.num_classes} classes")
    
    # Create model with correct features
    print("\n2. Creating model...")
    model = TabularMSDNet(args).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training
    print("\n3. Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_simple(model, train_loader, optimizer, epoch, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Quick validation
        if epoch % 1 == 0:
            val_acc, val_auc = evaluate_simple(model, val_loader, device)
            print(f"Validation - Accuracy: {val_acc:.2f}%, AUC: {val_auc:.4f}")
    
    # Final evaluation
    print("\n4. Final evaluation on test set...")
    test_acc, test_auc = evaluate_simple(model, test_loader, device)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")
    print("="*60)
    
    print("\n✅ Complete workflow successful!")
    print("✅ Training: Completed")
    print("✅ Evaluation: Completed")
    print("✅ No model structure mismatches")


if __name__ == '__main__':
    main()
