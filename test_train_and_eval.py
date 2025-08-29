#!/usr/bin/env python3
"""
Test script to run training and evaluation in sequence
"""

import subprocess
import time
import os
import glob

def find_latest_model():
    """Find the latest trained model"""
    # Look for the most recent results directory
    results_dirs = glob.glob("results/exp_*")
    if not results_dirs:
        return None
    
    # Get the most recent one
    latest_dir = max(results_dirs, key=os.path.getctime)
    
    # Look for best_model.pth in that directory
    model_path = os.path.join(latest_dir, "best_model.pth")
    if os.path.exists(model_path):
        return model_path
    
    return None

def test_train_and_eval():
    """Test training and evaluation workflow"""
    
    print("="*80)
    print("TESTING COMPLETE TRAIN & EVAL WORKFLOW")
    print("="*80)
    
    print("\n1. **Training Phase**")
    print("   - Training MSDNet on Adult dataset")
    print("   - 2 epochs, 2 blocks")
    print("   - Will save best model")
    
    # Training command
    train_cmd = [
        "python", "train_tabular.py",
        "--dataset", "tabular",
        "--tabular_dataset", "adult",
        "--arch", "msdnet",
        "--epochs", "2",
        "--nBlocks", "2"
    ]
    
    print(f"\nTraining command: {' '.join(train_cmd)}")
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        # Run training
        start_time = time.time()
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print(f"Training time: {training_time:.1f} seconds")
            
            # Find the trained model
            model_path = find_latest_model()
            if model_path:
                print(f"✅ Model saved to: {model_path}")
                
                print("\n2. **Evaluation Phase**")
                print("   - Evaluating trained model on test set")
                print("   - Multiple runs for standard deviation")
                
                # Evaluation command
                eval_cmd = [
                    "python", "eval_tabular.py",
                    "--dataset", "tabular",
                    "--tabular_dataset", "adult",
                    "--arch", "msdnet",
                    "--evaluate_from", model_path
                ]
                
                print(f"\nEvaluation command: {' '.join(eval_cmd)}")
                print("\nStarting evaluation...")
                print("-" * 60)
                
                try:
                    # Run evaluation
                    eval_start_time = time.time()
                    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
                    eval_time = time.time() - eval_start_time
                    
                    if eval_result.returncode == 0:
                        print("✅ Evaluation completed successfully!")
                        print(f"Evaluation time: {eval_time:.1f} seconds")
                        
                        print("\n" + "="*80)
                        print("COMPLETE WORKFLOW SUCCESS! 🎉")
                        print("="*80)
                        print("✅ Training: Completed")
                        print("✅ Model saving: Completed")
                        print("✅ Evaluation: Completed")
                        print("✅ Standard deviation: Calculated")
                        print("✅ Results: Saved")
                        
                        print("\nFinal evaluation output:")
                        print(eval_result.stdout)
                        
                    else:
                        print("❌ Evaluation failed!")
                        print("Error output:")
                        print(eval_result.stderr)
                        print("Standard output:")
                        print(eval_result.stdout)
                        
                except subprocess.TimeoutExpired:
                    print("⏰ Evaluation timed out after 5 minutes")
                except Exception as e:
                    print(f"❌ Error running evaluation: {e}")
                    
            else:
                print("❌ Could not find trained model!")
                print("Training output:")
                print(result.stdout)
                
        else:
            print("❌ Training failed!")
            print("Error output:")
            print(result.stderr)
            print("Standard output:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 10 minutes")
    except Exception as e:
        print(f"❌ Error running training: {e}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    test_train_and_eval()
