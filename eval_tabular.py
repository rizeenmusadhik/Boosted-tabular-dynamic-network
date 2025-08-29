#!/usr/bin/env python3
"""
Evaluation script for tabular data models
"""

import os
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, r2_score

from models import TabularMSDNet, TabularRANet
from dataloader import get_dataloaders
from args import arg_parser
from utils.utils import setup_logging


def evaluate_model(model, test_loader, device, task_type='classification'):
    """Evaluate model on test set with multiple metrics"""
    model.eval()
    
    if task_type == 'regression':
        # Regression evaluation
        all_predictions = []
        all_targets = []
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                outputs = model(x).squeeze()  # Remove extra dimensions for regression
                
                # Calculate MSE and MAE
                mse = torch.nn.functional.mse_loss(outputs, y, reduction='sum')
                mae = torch.nn.functional.l1_loss(outputs, y, reduction='sum')
                
                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += y.size(0)
                
                # Store results
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Calculate final metrics
        mse = total_mse / total_samples
        mae = total_mae / total_samples
        rmse = np.sqrt(mse)
        
        # Calculate R² score
        r2 = r2_score(all_targets, all_predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    else:
        # Classification evaluation (original code)
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
                if outputs.shape[1] >= 2:  # Binary or multi-class
                    all_probabilities.append(probabilities[:, 1].cpu().numpy() if outputs.shape[1] == 2 else probabilities.cpu().numpy())
                else:  # Single class (shouldn't happen in classification)
                    all_probabilities.append(probabilities.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions)
        all_probabilities = np.concatenate(all_probabilities)
        all_targets = np.concatenate(all_targets)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        
        try:
            if len(np.unique(all_targets)) > 2:  # Multi-class
                auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='macro')
            else:  # Binary
                auc = roc_auc_score(all_targets, all_probabilities)
        except ValueError:
            auc = 0.5  # Default value if AUC cannot be calculated
        
        # Generate classification report
        report = classification_report(all_targets, all_predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'classification_report': report
        }


def run_multiple_evaluations(model, test_loader, device, task_type='classification', n_runs=10):
    """Run multiple evaluations to get standard deviation"""
    print(f"Running {n_runs} evaluation runs for standard deviation...")
    
    if task_type == 'regression':
        mses, maes, rmses, r2s = [], [], [], []
        
        for run in range(n_runs):
            if run % 2 == 0:
                print(f"  Run {run + 1}/{n_runs}")
            
            # Set different random seeds for each run
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            results = evaluate_model(model, test_loader, device, task_type)
            mses.append(results['mse'])
            maes.append(results['mae'])
            rmses.append(results['rmse'])
            r2s.append(results['r2'])
        
        # Calculate statistics
        return {
            'mse_mean': np.mean(mses),
            'mse_std': np.std(mses),
            'mae_mean': np.mean(maes),
            'mae_std': np.std(maes),
            'rmse_mean': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'r2_mean': np.mean(r2s),
            'r2_std': np.std(r2s),
            'mse_values': mses,
            'mae_values': maes,
            'rmse_values': rmses,
            'r2_values': r2s
        }
    
    else:
        accuracies = []
        aucs = []
        
        for run in range(n_runs):
            if run % 2 == 0:
                print(f"  Run {run + 1}/{n_runs}")
            
            # Set different random seeds for each run
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            results = evaluate_model(model, test_loader, device, task_type)
            accuracies.append(results['accuracy'])
            aucs.append(results['auc'])
        
        # Calculate statistics
        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'accuracy_values': accuracies,
            'auc_values': aucs
        }


def main():
    # Parse arguments
    args = arg_parser.parse_args()
    
    # Check if model path is provided
    if not hasattr(args, 'evaluate_from') or not args.evaluate_from:
        print("Error: Please provide --evaluate_from <path_to_model>")
        print("Example: python eval_tabular.py --dataset tabular --tabular_dataset adult --arch msdnet --evaluate_from results/exp_xxx/best_model.pth")
        return
    
    # Load checkpoint early to pull correct metadata
    checkpoint = torch.load(args.evaluate_from, map_location='cpu')
    ckpt_args = checkpoint.get('args', None)
    meta = checkpoint.get('meta', {})

    # Safely coerce list-like args if they are strings (robust to already-parsed lists)
    def ensure_list_int(val, sep='-'):
        if isinstance(val, str):
            return list(map(int, val.split(sep)))
        return list(val)

    # Apply metadata/args from checkpoint to ensure correct num_features/classes prior to dataloaders/model
    if ckpt_args:
        # Merge minimal fields needed for tabular consistency
        args.dataset = ckpt_args.get('dataset', getattr(args, 'dataset', 'tabular'))
        args.tabular_dataset = ckpt_args.get('tabular_dataset', getattr(args, 'tabular_dataset', 'custom'))
        args.num_features = ckpt_args.get('num_features', getattr(args, 'num_features', None))
        args.num_classes = ckpt_args.get('num_classes', getattr(args, 'num_classes', None))
        args.arch = ckpt_args.get('arch', getattr(args, 'arch', 'msdnet'))
        # Arch hyperparams that affect shapes
        args.nBlocks = ckpt_args.get('nBlocks', getattr(args, 'nBlocks', 1))
        args.nChannels = ckpt_args.get('nChannels', getattr(args, 'nChannels', 32))
        args.base = ckpt_args.get('base', getattr(args, 'base', 4))
        args.step = ckpt_args.get('step', getattr(args, 'step', 1))
        args.stepmode = ckpt_args.get('stepmode', getattr(args, 'stepmode', 'even'))
        args.growthRate = ckpt_args.get('growthRate', getattr(args, 'growthRate', 6))
        args.grFactor = ensure_list_int(ckpt_args.get('grFactor', getattr(args, 'grFactor', '1-2-4')))
        args.bnFactor = ensure_list_int(ckpt_args.get('bnFactor', getattr(args, 'bnFactor', '1-2-4')))
        # RANet specific
        args.scale_list = ensure_list_int(ckpt_args.get('scale_list', getattr(args, 'scale_list', '1-2-3')))
        args.block_step = ckpt_args.get('block_step', getattr(args, 'block_step', 2))
        args.compress_factor = ckpt_args.get('compress_factor', getattr(args, 'compress_factor', 0.25))
        args.bnAfter = ckpt_args.get('bnAfter', getattr(args, 'bnAfter', True))
        # General
        args.reduction = ckpt_args.get('reduction', getattr(args, 'reduction', 0.5))
        args.prune = ckpt_args.get('prune', getattr(args, 'prune', 'max'))
    else:
        # Fall back to meta if present
        if 'num_features' in meta and meta['num_features'] is not None:
            args.num_features = meta['num_features']
        if 'num_classes' in meta and meta['num_classes'] is not None:
            args.num_classes = meta['num_classes']
        if 'dataset' in meta and meta['dataset'] is not None:
            args.dataset = meta['dataset']
        if 'tabular_dataset' in meta and meta['tabular_dataset'] is not None:
            args.tabular_dataset = meta['tabular_dataset']
        if 'arch' in meta and meta['arch'] is not None:
            args.arch = meta['arch']

        # Final fallback: infer num_features/num_classes from state_dict if still missing
        sd = checkpoint.get('model_state_dict', {})
        if getattr(args, 'num_features', None) in [None, 0]:
            inferred_nf = None
            # Prefer explicit first layer weights
            for key, tensor in sd.items():
                if not hasattr(tensor, 'shape'):
                    continue
                if key.endswith('first_layer.net.0.weight') and len(tensor.shape) == 2:
                    inferred_nf = int(tensor.shape[1])
                    break
            # MSDNet tabular fallback pattern
            if inferred_nf is None:
                for key, tensor in sd.items():
                    if not hasattr(tensor, 'shape'):
                        continue
                    if 'blocks.0' in key and key.endswith('layers.0.net.0.weight') and len(tensor.shape) == 2:
                        inferred_nf = int(tensor.shape[1])
                        break
            if inferred_nf is not None:
                args.num_features = inferred_nf

        if getattr(args, 'num_classes', None) in [None, 0]:
            inferred_nc = None
            # Look for final classifier last linear layer
            for key, tensor in sd.items():
                if not hasattr(tensor, 'shape'):
                    continue
                if len(tensor.shape) == 2 and (
                    ('.classifier.' in key or '.classifiers.' in key) and key.endswith('.3.weight')
                ):
                    inferred_nc = int(tensor.shape[0])
                    break
            if inferred_nc is not None:
                args.num_classes = inferred_nc

    # Process arguments that need conversion (idempotent)
    args.grFactor = ensure_list_int(getattr(args, 'grFactor', '1-2-4'))
    args.bnFactor = ensure_list_int(getattr(args, 'bnFactor', '1-2-4'))
    args.scale_list = ensure_list_int(getattr(args, 'scale_list', '1-2-3'))
    args.nScales = len(args.grFactor)
    # lr_milestones/ensemble_reweight are unused during eval, but keep consistent
    if isinstance(args.lr_milestones, str):
        args.lr_milestones = list(map(int, args.lr_milestones.split(',')))
    if isinstance(args.ensemble_reweight, str):
        args.ensemble_reweight = list(map(float, args.ensemble_reweight.split(',')))
    
    # Set splits
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data (for tabular datasets this may override num_features; ensure args reflects checkpoint)
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(args)
    # Guard: ensure args.num_features matches actual dataset features
    try:
        if args.dataset == 'tabular' and hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'features'):
            detected_features = int(train_loader.dataset.features.shape[1])
            if getattr(args, 'num_features', None) != detected_features:
                print(f"Adjusting num_features from {getattr(args, 'num_features', None)} to detected {detected_features}")
                args.num_features = detected_features
    except Exception as e:
        logging.warning(f"Could not verify dataset feature count: {e}")
    
    # Determine task type
    task_type = getattr(args, 'task_type', 'classification')
    print(f"Dataset loaded: {args.num_features} features, {args.num_classes} {'outputs' if task_type == 'regression' else 'classes'}")
    print(f"Task type: {task_type}")
    
    # Create model
    if args.arch == 'msdnet':
        model_func = TabularMSDNet
    elif args.arch == 'msdnet_ge':
        model_func = TabularMSDNet
    elif args.arch == 'ranet':
        model_func = TabularRANet
    else:
        raise Exception('unknown model name')
    
    model = model_func(args).to(device)
    
    # Load trained weights
    print(f"Loading model from: {args.evaluate_from}")
    # checkpoint already loaded on CPU; move tensors to device afterwards
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if task_type == 'regression':
        if 'best_mse' in checkpoint:
            print(f"Best training MSE: {checkpoint['best_mse']:.4f}")
        if 'best_r2' in checkpoint:
            print(f"Best training R²: {checkpoint['best_r2']:.4f}")
    else:
        if 'best_acc' in checkpoint:
            print(f"Best training accuracy: {checkpoint['best_acc']:.2f}%")
        if 'best_auc' in checkpoint:
            print(f"Best training AUC: {checkpoint['best_auc']:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Single evaluation
    print("\nSingle evaluation:")
    results = evaluate_model(model, test_loader, device, task_type)
    
    if task_type == 'regression':
        print(f"MSE: {results['mse']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"R²: {results['r2']:.4f}")
    else:
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"AUC: {results['auc']:.4f}")
    
    # Multiple evaluations for standard deviation
    print("\nMultiple evaluations (with standard deviation):")
    stats = run_multiple_evaluations(model, test_loader, device, task_type, n_runs=10)
    
    print(f"\nFinal Results:")
    if task_type == 'regression':
        print(f"MSE: {stats['mse_mean']:.4f} ± {stats['mse_std']:.4f}")
        print(f"MAE: {stats['mae_mean']:.4f} ± {stats['mae_std']:.4f}")
        print(f"RMSE: {stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}")
        print(f"R²: {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")
    else:
        print(f"Accuracy: {stats['accuracy_mean']:.2f}% ± {stats['accuracy_std']:.2f}%")
        print(f"AUC: {stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}")
    
    # Save results
    result_dir = os.path.dirname(args.evaluate_from)
    eval_results = {
        'single_eval': results,
        'multiple_eval_stats': stats,
        'model_path': args.evaluate_from,
        'dataset': args.tabular_dataset,
        'architecture': args.arch,
        'task_type': task_type,
        'num_features': args.num_features,
        'num_classes': args.num_classes
    }
    
    results_path = os.path.join(result_dir, 'evaluation_results.pth')
    torch.save(eval_results, results_path)
    print(f"\nEvaluation results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
