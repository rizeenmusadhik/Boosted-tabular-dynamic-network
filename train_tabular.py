import os
import logging
import shutil
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from torch.utils.tensorboard import SummaryWriter

from models import TabularMSDNet, TabularRANet
from models.dynamic_net import DynamicNet
from models.dynamic_net_ranet import DynamicNet as DynamicRANet
from op_counter import measure_model
from dataloader import get_dataloaders
from utils.utils import setup_logging
from args import arg_parser


def test(model, test_loader, device, task_type='classification'):
    model.eval_all()

    n_blocks = model.nBlocks
    
    if task_type == 'regression':
        # Initialize metrics for each block
        block_mses = [0.0] * n_blocks
        block_maes = [0.0] * n_blocks
        block_r2s = [0.0] * n_blocks
        total_samples = 0
        all_predictions_per_block = [[] for _ in range(n_blocks)]
        all_targets = []
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outs = model.forward(x)  # Get outputs from all blocks
            
            # Store targets (same for all blocks)
            if len(all_targets) == 0:
                all_targets = y.cpu().numpy()
            else:
                all_targets = np.concatenate([all_targets, y.cpu().numpy()])
            
            # Calculate metrics for each block
            for i, out in enumerate(outs):
                out = out.squeeze()  # Remove extra dimensions for regression
                
                # Calculate MSE and MAE for this block
                mse = torch.nn.functional.mse_loss(out, y, reduction='sum')
                mae = torch.nn.functional.l1_loss(out, y, reduction='sum')
                
                block_mses[i] += mse.item()
                block_maes[i] += mae.item()
                
                # Store predictions for R² calculation
                all_predictions_per_block[i].append(out.cpu().numpy())
            
            total_samples += y.size(0)
        
        # Calculate final metrics for each block
        from sklearn.metrics import r2_score
        results = []
        for i in range(n_blocks):
            mse = block_mses[i] / total_samples
            mae = block_maes[i] / total_samples
            rmse = np.sqrt(mse)
            
            # Calculate R² score
            all_preds = np.concatenate(all_predictions_per_block[i])
            r2 = r2_score(all_targets, all_preds)
            
            results.append((mse, mae, rmse, r2))
        
        return results
    
    else:
        # Classification metrics - per block accuracy
        corrects = [0] * n_blocks
        totals = [0] * n_blocks
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outs = model.forward(x)  # Get outputs from all blocks
            
            for i, out in enumerate(outs):
                corrects[i] += (torch.argmax(out, 1) == y).sum().item()
                totals[i] += y.shape[0]
        
        # Return accuracy for each block
        return [c / t * 100 for c, t in zip(corrects, totals)]


def log_step(step, name, value, sum_writer, silent=False):
    if not silent:
        logging.info(f'step {step}, {name} {value:.4f}')
    sum_writer.add_scalar(f'{name}', value, step)


def train(model, train_loader, optimizer, epoch, sum_writer, device, task_type='classification'):
    model.train_all()
    
    if task_type == 'regression':
        criterion = torch.nn.MSELoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    n_blocks = model.nBlocks
    for it, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        if task_type == 'regression':
            y = y.float()  # Ensure target is float for regression
        
        # Forward pass through all blocks with ensemble predictions
        preds, pred_ensembles = model.forward_all(x, n_blocks - 1)
        loss_all = 0
        
        for stage in range(n_blocks):
            # Train weak learner with ensemble prediction
            with torch.no_grad():
                if not isinstance(pred_ensembles[stage], torch.Tensor):
                    if task_type == 'regression':
                        out = torch.unsqueeze(torch.Tensor([pred_ensembles[stage]]), 0)  # 1x1
                        out = out.expand(x.shape[0], 1).to(device)  # Expand for regression (single output)
                    else:
                        out = torch.unsqueeze(torch.Tensor([pred_ensembles[stage]]), 0)  # 1x1
                        out = out.expand(x.shape[0], preds[stage].shape[1]).to(device)  # Expand for classification
                else:
                    out = pred_ensembles[stage]
                out = out.detach()

            if task_type == 'regression':
                pred_input = preds[stage].squeeze() + out.squeeze()
                loss = criterion(pred_input, y)
            else:
                loss = criterion(preds[stage] + out, y)
            
            if it % 50 == 0:
                log_step(epoch * len(train_loader) + it, f'stage_{stage}_loss', loss, sum_writer)
            loss_all = loss_all + loss

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()


def main():
    # Parse arguments
    args = arg_parser.parse_args()
    
    # Process arguments that need conversion
    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.scale_list = list(map(int, args.scale_list.split('-')))
    args.nScales = len(args.grFactor)
    args.lr_milestones = list(map(int, args.lr_milestones.split(',')))
    args.ensemble_reweight = list(map(float, args.ensemble_reweight.split(',')))
    
    # Set splits
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']
    
    torch.backends.cudnn.benchmark = True

    # Create result directory if it doesn't exist
    os.makedirs(args.result_dir, exist_ok=True)
    
    setup_logging(os.path.join(args.result_dir, 'log.txt'))
    sum_writer = SummaryWriter(os.path.join(args.result_dir, 'summary'))

    if args.arch == 'msdnet':
        model_func = TabularMSDNet
    elif args.arch == 'msdnet_ge':
        model_func = TabularMSDNet  # Use tabular version for now
    elif args.arch == 'ranet':
        model_func = TabularRANet
    else:
        raise Exception('unknown model name')

    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data first to get the correct number of features
    print(f"Before data loading - num_features: {args.num_features}")
    train_loader, val_loader, _ = get_dataloaders(args)
    print(f"After data loading - num_features: {args.num_features}")
    
    # Determine task type
    task_type = getattr(args, 'task_type', 'classification')
    print(f"Task type: {task_type}")
    
    # Log arguments AFTER data loading (so num_features is correct)
    logging.info("running arguments: %s", args)
    
    # Skip model measurement for tabular data (measure_model is designed for image data)
    print("Skipping FLOP measurement for tabular data")
    
    # Create backbone again and wrap with dynamic network
    backbone = model_func(args)
    if args.arch == 'ranet':
        model = DynamicRANet(backbone, args)
    else:
        model = DynamicNet(backbone, args)
    
    # Move to device
    if device.type == 'cuda':
        model = model.cuda_all()
    else:
        model = model.cpu_all()

    # Get number of blocks
    n_blocks = args.nBlocks

    # Process ensemble_reweight (from args.py)
    if isinstance(args.ensemble_reweight, str):
        args.ensemble_reweight = list(map(float, args.ensemble_reweight.split(',')))
    
    assert len(args.ensemble_reweight) in [1, 2, n_blocks]
    if len(args.ensemble_reweight) == 1:
        args.ensemble_reweight = args.ensemble_reweight * n_blocks
    elif len(args.ensemble_reweight) == 2:
        args.ensemble_reweight = list(np.linspace(args.ensemble_reweight[0], args.ensemble_reweight[1], n_blocks))
    
    # Update model's reweight with the processed ensemble_reweight
    model.reweight = args.ensemble_reweight

    # Optimizer configuration (following CIFAR implementation)
    if args.arch != 'ranet':
        # For MSDNet: per-block weight decay
        if isinstance(args.weight_decay, str):
            args.weight_decay = list(map(float, args.weight_decay.split(',')))
        if len(args.weight_decay) == 1:
            args.weight_decay = args.weight_decay * n_blocks
        else:
            args.weight_decay = list(np.linspace(args.weight_decay[0], args.weight_decay[-1], n_blocks))
        
        params_group = []
        for i in range(n_blocks):
            param_i = model.parameters_m(i, separate=False)
            params_group.append({'params': param_i, 'weight_decay': args.weight_decay[i]})
    else:
        # For RANet: single weight decay for all parameters
        if isinstance(args.weight_decay, str):
            args.weight_decay = list(map(float, args.weight_decay.split(',')))
        assert len(args.weight_decay) == 1
        params_group = [{'params': model.parameters_all(n_blocks-1, all_classifiers=True),
                         'weight_decay': args.weight_decay[0]}]
    
    optimizer = torch.optim.SGD(params_group, args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=0.1)

    # Training loop
    if task_type == 'regression':
        best_mse = float('inf')
        best_r2 = -float('inf')
    else:
        best_acc = 0
        best_auc = 0
        
    for epoch in range(args.start_epoch, args.epochs):
        train(model, train_loader, optimizer, epoch, sum_writer, device, task_type)
        scheduler.step()

        # Validation (following CIFAR implementation with per-block evaluation)
        if epoch % 1 == 0:  # Every epoch to see per-block progress
            if task_type == 'regression':
                # Get per-block results
                results_test = test(model, val_loader, device, task_type)
                results_train = test(model, train_loader, device, task_type)
                
                # Log per-block metrics
                for i, (mse_test, mae_test, rmse_test, r2_test) in enumerate(results_test):
                    log_step((epoch + 1) * len(train_loader), f'stage_{i}_mse', mse_test, sum_writer)
                    log_step((epoch + 1) * len(train_loader), f'stage_{i}_r2', r2_test, sum_writer)
                
                for i, (mse_train, mae_train, rmse_train, r2_train) in enumerate(results_train):
                    log_step((epoch + 1) * len(train_loader), f'stage_{i}_mse_train', mse_train, sum_writer)
                    log_step((epoch + 1) * len(train_loader), f'stage_{i}_r2_train', r2_train, sum_writer)
                
                # Print progress for final block (best performance)
                final_mse, final_mae, final_rmse, final_r2 = results_test[-1]
                logging.info(f'Epoch {epoch}: Final Block - MSE: {final_mse:.4f}, R²: {final_r2:.4f}')
                
                # Save best model based on final block MSE
                if final_mse < best_mse:
                    best_mse = final_mse
                    best_r2 = final_r2
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_mse': best_mse,
                        'best_r2': best_r2,
                        'args': vars(args),
                        'meta': {
                            'dataset': args.dataset,
                            'tabular_dataset': getattr(args, 'tabular_dataset', None),
                            'num_features': getattr(args, 'num_features', None),
                            'num_classes': getattr(args, 'num_classes', None),
                            'task_type': task_type,
                            'arch': args.arch,
                        }
                    }, os.path.join(args.result_dir, 'best_model.pth'))
                    
            else:
                # Classification with per-block accuracy
                accus_test = test(model, val_loader, device, task_type)
                accus_train = test(model, train_loader, device, task_type)
                
                # Log per-block accuracies
                for i, accu in enumerate(accus_test):
                    log_step((epoch + 1) * len(train_loader), f'stage_{i}_accu', accu, sum_writer)
                
                for i, accu in enumerate(accus_train):
                    log_step((epoch + 1) * len(train_loader), f'stage_{i}_accu_train', accu, sum_writer)
                
                # Print progress for final block
                final_acc = accus_test[-1]
                logging.info(f'Epoch {epoch}: Final Block Accuracy: {final_acc:.2f}%')
                
                # Save best model based on final block accuracy
                if final_acc > best_acc:
                    best_acc = final_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'args': vars(args),
                        'meta': {
                            'dataset': args.dataset,
                            'tabular_dataset': getattr(args, 'tabular_dataset', None),
                            'num_features': getattr(args, 'num_features', None),
                            'num_classes': getattr(args, 'num_classes', None),
                            'task_type': task_type,
                            'arch': args.arch,
                        }
                    }, os.path.join(args.result_dir, 'best_model.pth'))

        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # Persist full training configuration for reliable evaluation
                'args': vars(args),
                'meta': {
                    'dataset': args.dataset,
                    'tabular_dataset': getattr(args, 'tabular_dataset', None),
                    'num_features': getattr(args, 'num_features', None),
                    'num_classes': getattr(args, 'num_classes', None),
                    'task_type': task_type,
                    'arch': args.arch,
                }
            }, os.path.join(args.result_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    # Final evaluation with per-block results
    final_results = test(model, val_loader, device, task_type)
    
    if task_type == 'regression':
        print("\nPer-Block Regression Results:")
        for i, (mse, mae, rmse, r2) in enumerate(final_results):
            print(f"Block {i}: MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Final block performance
        final_mse, final_mae, final_rmse, final_r2 = final_results[-1]
        print(f"\nFinal Block Performance:")
        print(f"MSE: {final_mse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"R²: {final_r2:.4f}")
        print(f"Best MSE: {best_mse:.4f}")
        print(f"Best R²: {best_r2:.4f}")
        
    else:
        print("\nPer-Block Classification Results:")
        for i, acc in enumerate(final_results):
            print(f"Block {i}: Accuracy: {acc:.2f}%")
        
        # Final block performance
        final_acc = final_results[-1]
        print(f"\nFinal Block Performance:")
        print(f"Accuracy: {final_acc:.2f}%")
        print(f"Best Accuracy: {best_acc:.2f}%")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
