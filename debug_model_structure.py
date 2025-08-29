#!/usr/bin/env python3
"""
Debug script to check model structure
"""

import torch
from models import TabularMSDNet
from args import arg_parser

def debug_model_structure():
    """Debug the current model structure"""
    
    # Parse arguments
    args = arg_parser.parse_args()
    
    # Set up for Adult dataset
    args.dataset = 'tabular'
    args.tabular_dataset = 'adult'
    args.nBlocks = 2
    
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
    
    # Create model
    model = TabularMSDNet(args)
    
    print("Current model structure:")
    print("=" * 50)
    
    # Print model structure
    for name, module in model.named_modules():
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            print(f"{name}: {type(module).__name__}")
            if hasattr(module, 'weight'):
                print(f"  Weight shape: {module.weight.shape}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"  Bias shape: {module.bias.shape}")
    
    print("\n" + "=" * 50)
    print("Model state dict keys:")
    state_dict = model.state_dict()
    for key in state_dict.keys():
        print(f"  {key}: {state_dict[key].shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == '__main__':
    debug_model_structure()
