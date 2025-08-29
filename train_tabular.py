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


def test(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def log_step(step, name, value, sum_writer, silent=False):
    if not silent:
        logging.info(f'step {step}, {name} {value:.4f}')
    sum_writer.add_scalar(f'{name}', value, step)


def train(model, train_loader, optimizer, epoch, sum_writer, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for it, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass through all blocks
        outputs = model(x)
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
    logging.info("running arguments: %s", args)
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
    train_loader, val_loader, _ = get_dataloaders(args)
    
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
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train(model, train_loader, optimizer, epoch, sum_writer, device)
        scheduler.step()

        # Validation
        if epoch % 10 == 0:
            accuracy = test(model, val_loader, device)
            logging.info(f'Epoch {epoch}: Accuracy: {accuracy:.2f}%')
            
            # Save best model
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(args.result_dir, 'best_model.pth'))

        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.result_dir, f'checkpoint_epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()
