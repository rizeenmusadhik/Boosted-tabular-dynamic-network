#!/usr/bin/env python3
"""
Training script for real tabular datasets (Adult, HELOC, Covertype)
"""

import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from models import TabularMSDNet, TabularRANet
from dataloader import get_dataloaders
from utils.utils import setup_logging
from args import args


def train_model(model, train_loader, val_loader, args, model_name):
    """Train a single model"""
    print(f"\nTraining {model_name}...")
    
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(accuracy)
        
        # Early stopping
        if accuracy > best_acc:
            best_acc = accuracy
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'best_{model_name}_{args.tabular_dataset}.pth')
        else:
            patience_counter += 1
            if patience_counter >= 20:  # Early stopping patience
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_acc


def main():
    # Set up logging
    setup_logging('training_log.txt')
    logging.info("Starting training for real tabular datasets")
    
    # Test datasets
    test_datasets = ['adult', 'heloc', 'covertype']
    
    for dataset_name in test_datasets:
        print(f"\n{'='*80}")
        print(f"TRAINING ON DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Set up arguments for this dataset
            args.dataset = 'tabular'
            args.tabular_dataset = dataset_name
            args.epochs = 100
            args.batch_size = 128
            
            # Model architecture parameters
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
            
            # Load data
            print(f"Loading {dataset_name} dataset...")
            train_loader, val_loader, test_loader = get_dataloaders(args)
            
            print(f"Dataset loaded successfully!")
            print(f"Features: {args.num_features}, Classes: {args.num_classes}")
            print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # Train MSDNet
            print(f"\nTraining TabularMSDNet on {dataset_name}...")
            msdnet_model = TabularMSDNet(args)
            msdnet_acc = train_model(msdnet_model, train_loader, val_loader, args, 'MSDNet')
            
            # Train RANet
            print(f"\nTraining TabularRANet on {dataset_name}...")
            # Set RANet specific arguments
            args.scale_list = [1, 2, 3]
            args.block_step = 2
            args.compress_factor = 0.25
            args.bnAfter = True
            
            ranet_model = TabularRANet(args)
            ranet_acc = train_model(ranet_model, train_loader, val_loader, args, 'RANet')
            
            # Results
            print(f"\n{'='*60}")
            print(f"RESULTS FOR {dataset_name.upper()}")
            print(f"{'='*60}")
            print(f"MSDNet Best Accuracy: {msdnet_acc:.2f}%")
            print(f"RANet Best Accuracy: {ranet_acc:.2f}%")
            
        except Exception as e:
            print(f"Failed to train on {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("ALL TRAINING COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
