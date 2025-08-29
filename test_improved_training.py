#!/usr/bin/env python3
"""
Test script to demonstrate improved training with AUC and standard deviation
"""

import subprocess
import sys

def test_improved_training():
    """Test the improved training script with 2 epochs and 2 blocks"""
    
    print("="*60)
    print("TESTING IMPROVED TRAINING SCRIPT")
    print("="*60)
    
    print("\n1. **How to Define Number of Blocks:**")
    print("   - Use --nBlocks argument: --nBlocks 2")
    print("   - Use --epochs argument: --epochs 2")
    print("   - Example: python train_tabular.py --dataset tabular --tabular_dataset adult --arch msdnet --epochs 2 --nBlocks 2")
    
    print("\n2. **Available Block-Related Arguments:**")
    print("   - --nBlocks: Number of blocks (default: 1)")
    print("   - --nChannels: Number of channels per block (default: 32)")
    print("   - --base: Base number of layers (default: 4)")
    print("   - --step: Number of layers per step (default: 1)")
    print("   - --growthRate: Growth rate of channels (default: 6)")
    
    print("\n3. **Running Test with 2 Epochs and 2 Blocks:**")
    
    # Command to run
    cmd = [
        "python", "train_tabular.py",
        "--dataset", "tabular",
        "--tabular_dataset", "adult",
        "--arch", "msdnet",
        "--epochs", "2",
        "--nBlocks", "2"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print("\nError output:")
            print(result.stderr)
            print("\nStandard output:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running training: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == '__main__':
    test_improved_training()
