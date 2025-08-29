import os
import time
from operator import itemgetter

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Try to import openml, but make it optional
try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    print("Warning: OpenML not available. Install with: pip install openml")


# From https://github.com/catalyst-team/catalyst/blob/ea3fadbaa6034dabeefbbb53ab8c310186f6e5d0/catalyst/data/sampler.py#L522
class DatasetFromSampler(torch.utils.data.Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.distributed.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class TabularDataset(torch.utils.data.Dataset):
    """Dataset for tabular data"""
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label


def load_adult_income():
    """Load Adult Income dataset from OpenML"""
    if not OPENML_AVAILABLE:
        raise ImportError("OpenML not available. Install with: pip install openml")
    
    print("Loading Adult Income dataset...")
    dataset = openml.datasets.get_dataset(1590)  # Adult dataset ID
    data = dataset.get_data(
        target=dataset.default_target_attribute
    )
    X, y = data[0], data[1]
    
    # Convert to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # For simplicity, treat all columns as numerical (label encode if needed)
    # In production, you might want to use proper one-hot encoding
    X_encoded = X.copy()
    
    # Convert categorical columns to numerical
    for col in range(X_encoded.shape[1]):
        if X_encoded.dtype == 'object' or X_encoded[:, col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[:, col] = le.fit_transform(X_encoded[:, col].astype(str))
    
    # Convert to float
    X_encoded = X_encoded.astype(float)
    
    # Convert to float
    X_encoded = X_encoded.astype(float)
    
    # Convert target to binary (0, 1)
    y = (y == '>50K').astype(int)
    
    print(f"Adult Income dataset loaded: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
    return X_encoded, y


def load_heloc():
    """Load HELOC dataset from OpenML"""
    if not OPENML_AVAILABLE:
        raise ImportError("OpenML not available. Install with: pip install openml")
    
    print("Loading HELOC dataset...")
    dataset = openml.datasets.get_dataset(45066)  # HELOC dataset ID
    data = dataset.get_data(
        target=dataset.default_target_attribute
    )
    X, y = data[0], data[1]
    
    # Convert to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # Handle missing values (replace with median)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Convert target to binary (0, 1)
    y = (y == 'Good').astype(int)
    
    print(f"HELOC dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def load_covertype():
    """Load Covertype dataset from OpenML"""
    if not OPENML_AVAILABLE:
        raise ImportError("OpenML not available. Install with: pip install openml")
    
    print("Loading Covertype dataset...")
    dataset = openml.datasets.get_dataset(150)  # Covertype dataset ID
    data = dataset.get_data(
        target=dataset.default_target_attribute
    )
    X, y = data[0], data[1]
    
    # Convert to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # Convert target to 0-based indexing (original is 1-7)
    y = y - 1
    
    print(f"Covertype dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def create_tabular_data(args):
    """Create tabular data from various sources"""
    if args.tabular_dataset == 'custom':
        # Create synthetic classification data
        X, y = make_classification(
            n_samples=10000, 
            n_features=args.num_features, 
            n_classes=args.num_classes,
            n_informative=args.num_features // 2,
            n_redundant=args.num_features // 4,
            random_state=42
        )
        print(f"Custom synthetic dataset created: {X.shape[0]} samples, {X.shape[1]} features")
        
    elif args.tabular_dataset == 'adult':
        # Load real Adult Income dataset
        X, y = load_adult_income()
        args.num_features = X.shape[1]
        args.num_classes = len(np.unique(y))
        
    elif args.tabular_dataset == 'heloc':
        # Load real HELOC dataset
        X, y = load_heloc()
        args.num_features = X.shape[1]
        args.num_classes = len(np.unique(y))
        
    elif args.tabular_dataset == 'covertype':
        # Load real Covertype dataset
        X, y = load_covertype()
        args.num_features = X.shape[1]
        args.num_classes = len(np.unique(y))
        
    elif args.tabular_dataset == 'credit':
        # Credit card fraud simulation (keeping for backward compatibility)
        X, y = make_classification(
            n_samples=284807, 
            n_features=30, 
            n_classes=2,
            n_informative=20,
            n_redundant=5,
            random_state=42
        )
        args.num_features = 30
        args.num_classes = 2
        print(f"Credit card fraud simulation created: {X.shape[0]} samples, {X.shape[1]} features")
        
    elif args.tabular_dataset == 'diabetes':
        # Diabetes dataset simulation (keeping for backward compatibility)
        X, y = make_classification(
            n_samples=768, 
            n_features=8, 
            n_classes=2,
            n_informative=6,
            n_redundant=1,
            random_state=42
        )
        args.num_features = 8
        args.num_classes = 2
        print(f"Diabetes simulation created: {X.shape[0]} samples, {X.shape[1]} features")
    
    else:
        raise ValueError(f"Unknown tabular dataset: {args.tabular_dataset}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    print(f"Classes: {np.unique(y_train)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    
    if args.dataset == 'tabular':
        # Create tabular data
        X_train, X_val, X_test, y_train, y_val, y_test = create_tabular_data(args)
        
        train_set = TabularDataset(X_train, y_train)
        val_set = TabularDataset(X_val, y_val)
        test_set = TabularDataset(X_test, y_test)
        
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True, download=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data_root, 'train')
        # traindir = os.path.join(args.data_root, 'train_subset')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    else:
        raise Exception('Invalid dataset name')

    if args.use_valid:
        if os.path.exists(os.path.join(args.result_dir, 'index.pth')):
            # print('!!!!!! Load train_set_index !!!!!!')
            time.sleep(30)
            train_set_index = torch.load(os.path.join(args.result_dir, 'index.pth'))
        else:
            if not args.distributed or dist.get_rank() == 0:
                train_set_index = torch.randperm(len(train_set))
                torch.save(train_set_index, os.path.join(args.result_dir, 'index.pth'))
            # print('!!!!!! Save train_set_index !!!!!!')
            time.sleep(30)
            train_set_index = torch.load(os.path.join(args.result_dir, 'index.pth'))

        if args.dataset == 'tabular':
            num_sample_valid = len(train_set) // 10  # 10% for validation
        elif args.dataset.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_set_index[:-num_sample_valid])
            if args.distributed:
                train_sampler  = DistributedSamplerWrapper(train_sampler, shuffle=True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.workers,
                pin_memory=True)
        if 'val' in args.splits:
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-num_sample_valid:])
            if args.distributed:
                val_sampler  = DistributedSamplerWrapper(val_sampler, shuffle=False)
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=args.val_workers,
                pin_memory=True)
        if 'test' in args.splits:
            if args.distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                additional_args = {'shuffle': False, 'sampler': test_sampler}
            else:
                additional_args = {'shuffle': False}
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size,
                num_workers=args.val_workers,
                pin_memory=True,
                **additional_args)
    else:
        if 'train' in args.splits:
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
                additional_args = {'shuffle': False, 'sampler': train_sampler}
            else:
                additional_args = {'shuffle': True}
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                **additional_args)
        if 'val' in args.splits:
            if args.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                additional_args = {'shuffle': False, 'sampler': val_sampler}
            else:
                additional_args = {'shuffle': False}
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size,
                num_workers=args.val_workers,
                pin_memory=True,
                **additional_args)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
