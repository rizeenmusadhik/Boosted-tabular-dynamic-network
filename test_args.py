#!/usr/bin/env python3
"""
Test script to check if arguments are properly parsed
"""

from args import arg_parser

def test_args():
    """Test argument parsing"""
    
    print("Testing argument parsing...")
    
    # Test with minimal arguments
    test_args = [
        '--dataset', 'tabular',
        '--tabular_dataset', 'adult',
        '--arch', 'msdnet'
    ]
    
    try:
        args = arg_parser.parse_args(test_args)
        print("✓ Arguments parsed successfully!")
        print(f"  - Dataset: {args.dataset}")
        print(f"  - Tabular dataset: {args.tabular_dataset}")
        print(f"  - Architecture: {args.arch}")
        print(f"  - Result dir: {args.result_dir}")
        print(f"  - Num features: {args.num_features}")
        print(f"  - Num classes: {args.num_classes}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Epochs: {args.epochs}")
        
        # Check if required attributes exist
        required_attrs = [
            'nBlocks', 'nChannels', 'base', 'step', 'growthRate',
            'grFactor', 'bnFactor', 'bottleneck', 'reduction', 'prune'
        ]
        
        print("\nChecking required attributes:")
        for attr in required_attrs:
            if hasattr(args, attr):
                print(f"  ✓ {attr}: {getattr(args, attr)}")
            else:
                print(f"  ✗ {attr}: Missing!")
                
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_args()
