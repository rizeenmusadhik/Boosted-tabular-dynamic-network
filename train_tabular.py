import os
import logging
import shutil
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from torch.utils.tensorboard import SummaryWriter

from models import TabularMSDNet, TabularRANet
from op_counter import measure_model
from dataloader import get_dataloaders
from utils.utils import setup_logging
from args import arg_parser


def test(model, test_loader, device, task_type='classification'):
    model.eval()

    if task_type == 'regression':
        # Regression metrics
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = model(x).squeeze()  # Remove extra dimensions for regression
                
                # Calculate MSE and MAE
                mse = torch.nn.functional.mse_loss(outputs, y, reduction='sum')
                mae = torch.nn.functional.l1_loss(outputs, y, reduction='sum')
                
                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += y.size(0)
                
                # Store for R² calculation
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Calculate final metrics
        mse = total_mse / total_samples
        mae = total_mae / total_samples
        rmse = np.sqrt(mse)
        
        # Calculate R² score
        from sklearn.metrics import r2_score
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        r2 = r2_score(all_targets, all_predictions)
        
        return mse, mae, rmse, r2
    
    else:
        # Classification metrics (original code)
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # Store predictions and targets for AUC calculation
                probabilities = torch.softmax(outputs, dim=1)
                if outputs.shape[1] >= 2:  # Binary or multi-class
                    all_predictions.append(probabilities[:, 1].cpu().numpy() if outputs.shape[1] == 2 else probabilities.cpu().numpy())
                else:  # Single class (shouldn't happen in classification)
                    all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        try:
            if len(np.unique(all_targets)) > 2:  # Multi-class
                auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr', average='macro')
            else:  # Binary
                auc = roc_auc_score(all_targets, all_predictions)
        except ValueError:
            auc = 0.5  # Default value if AUC cannot be calculated
        
        return accuracy, auc


def log_step(step, name, value, sum_writer, silent=False):
    if not silent:
        logging.info(f'step {step}, {name} {value:.4f}')
    sum_writer.add_scalar(f'{name}', value, step)


def train(model, train_loader, optimizer, epoch, sum_writer, device, task_type='classification'):
    model.train()
    
    if task_type == 'regression':
        criterion = torch.nn.MSELoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    for it, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass through all blocks
        outputs = model(x)
        
        if task_type == 'regression':
            outputs = outputs.squeeze()  # Remove extra dimensions for regression
            y = y.float()  # Ensure target is float for regression
        
        loss = criterion(outputs, y)
        
        if it % 50 == 0:
            log_step(epoch * len(train_loader) + it, 'loss', loss, sum_writer)
        
        optimizer.zero_grad()
        loss.backward()
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
    
    # Now create the model with the correct number of features
    model = model_func(args).to(device)

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=float(args.weight_decay))
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                        weight_decay=float(args.weight_decay))
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=float(args.weight_decay))
    else:
        raise Exception('unknown optimizer')

    # Learning rate scheduler
    if args.lr_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, args.decay_rate)
    elif args.lr_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        raise Exception('unknown lr type')

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

        # Validation
        if epoch % 10 == 0:
            if task_type == 'regression':
                # Run multiple validation passes for standard deviation
                mses, maes, rmses, r2s = [], [], [], []
                for _ in range(5):  # 5 runs for std calculation
                    mse, mae, rmse, r2 = test(model, val_loader, device, task_type)
                    mses.append(mse)
                    maes.append(mae)
                    rmses.append(rmse)
                    r2s.append(r2)
                
                # Calculate mean and standard deviation
                mean_mse = np.mean(mses)
                std_mse = np.std(mses)
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                mean_rmse = np.mean(rmses)
                std_rmse = np.std(rmses)
                mean_r2 = np.mean(r2s)
                std_r2 = np.std(r2s)
                
                logging.info(f'Epoch {epoch}: MSE: {mean_mse:.4f} ± {std_mse:.4f}, RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}, R²: {mean_r2:.4f} ± {std_r2:.4f}')
                
                # Save best model based on MSE (lower is better)
                if mean_mse < best_mse:
                    best_mse = mean_mse
                    best_r2 = mean_r2
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_mse': best_mse,
                        'best_r2': best_r2,
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
                    }, os.path.join(args.result_dir, 'best_model.pth'))
                    
            else:
                # Classification (original code)
                # Run multiple validation passes for standard deviation
                accuracies = []
                aucs = []
                for _ in range(5):  # 5 runs for std calculation
                    accuracy, auc = test(model, val_loader, device, task_type)
                    accuracies.append(accuracy)
                    aucs.append(auc)
                
                # Calculate mean and standard deviation
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                
                logging.info(f'Epoch {epoch}: Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%, AUC: {mean_auc:.4f} ± {std_auc:.4f}')
                
                # Save best model based on accuracy
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_auc = mean_auc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'best_auc': best_auc,
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
    
    if task_type == 'regression':
        # Run multiple final validation passes for regression
        final_mses, final_maes, final_rmses, final_r2s = [], [], [], []
        for _ in range(10):  # 10 runs for final std calculation
            mse, mae, rmse, r2 = test(model, val_loader, device, task_type)
            final_mses.append(mse)
            final_maes.append(mae)
            final_rmses.append(rmse)
            final_r2s.append(r2)
        
        final_mean_mse = np.mean(final_mses)
        final_std_mse = np.std(final_mses)
        final_mean_mae = np.mean(final_maes)
        final_std_mae = np.std(final_maes)
        final_mean_rmse = np.mean(final_rmses)
        final_std_rmse = np.std(final_rmses)
        final_mean_r2 = np.mean(final_r2s)
        final_std_r2 = np.std(final_r2s)
        
        print(f"Final MSE: {final_mean_mse:.4f} ± {final_std_mse:.4f}")
        print(f"Final MAE: {final_mean_mae:.4f} ± {final_std_mae:.4f}")
        print(f"Final RMSE: {final_mean_rmse:.4f} ± {final_std_rmse:.4f}")
        print(f"Final R²: {final_mean_r2:.4f} ± {final_std_r2:.4f}")
        print(f"Best MSE: {best_mse:.4f}")
        print(f"Best R²: {best_r2:.4f}")
        
    else:
        # Run multiple final validation passes for classification
        final_accuracies = []
        final_aucs = []
        for _ in range(10):  # 10 runs for final std calculation
            accuracy, auc = test(model, val_loader, device, task_type)
            final_accuracies.append(accuracy)
            final_aucs.append(auc)
        
        final_mean_acc = np.mean(final_accuracies)
        final_std_acc = np.std(final_accuracies)
        final_mean_auc = np.mean(final_aucs)
        final_std_auc = np.std(final_aucs)
        
        print(f"Final Accuracy: {final_mean_acc:.2f}% ± {final_std_acc:.2f}%")
        print(f"Final AUC: {final_mean_auc:.4f} ± {final_std_auc:.4f}")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print(f"Best AUC: {best_auc:.4f}")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
