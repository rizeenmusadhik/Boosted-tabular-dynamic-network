#!/usr/bin/env python3
"""
Script to check device availability and capabilities
"""

import torch


def check_device():
    """Check available devices and their capabilities"""
    print("="*60)
    print("DEVICE AVAILABILITY CHECK")
    print("="*60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA device count
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices: {device_count}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA Device: {current_device}")
        
        # Get device properties
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
        
        # Set device
        device = torch.device('cuda')
        print(f"\nUsing device: {device}")
        
        # Test CUDA tensor operations
        try:
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("✓ CUDA tensor operations successful")
        except Exception as e:
            print(f"✗ CUDA tensor operations failed: {e}")
    
    else:
        # Use CPU
        device = torch.device('cpu')
        print(f"Using device: {device}")
        
        # Test CPU tensor operations
        try:
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            z = torch.mm(x, y)
            print("✓ CPU tensor operations successful")
        except Exception as e:
            print(f"✗ CPU tensor operations failed: {e}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device selected: {device}")
    
    # Test model creation on device
    try:
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        # Test forward pass
        x = torch.randn(32, 100).to(device)
        output = model(x)
        
        print(f"✓ Model creation and forward pass successful on {device}")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Model device: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
    
    print("\n" + "="*60)
    print("DEVICE CHECK COMPLETE")
    print("="*60)


if __name__ == '__main__':
    check_device()
