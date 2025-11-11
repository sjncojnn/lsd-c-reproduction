from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
from torchvision import datasets
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler


class CIFAR10_ALL(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    label_type = 'labels'
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, nb_train=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        data_train = []
        data_test = []
        targets_train = []
        targets_test = []

        # now load the picked numpy arrays
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data_train.append(entry['data'])
                targets_train.extend(entry[self.label_type])

        data_train = np.vstack(data_train).reshape(-1, 3, 32, 32)
        data_train = data_train.transpose((0, 2, 3, 1))  # convert to HWC

        for file_name, checksum in self.test_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data_test.append(entry['data'])
                targets_test.extend(entry[self.label_type])

        data_test = np.vstack(data_test).reshape(-1, 3, 32, 32)
        data_test = data_test.transpose((0, 2, 3, 1))  # convert to HWC
        self.data = np.concatenate((data_train, data_test), 0)
        self.targets = targets_train + targets_test
        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                        ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class CIFAR100_ALL(data.Dataset):
    """Merge CIFAR-100 train and test into one set, similar to CIFAR10_ALL"""
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.dataset_train = datasets.CIFAR100(root=root, train=True, download=download)
        self.dataset_test = datasets.CIFAR100(root=root, train=False, download=download)

        self.data = np.concatenate((self.dataset_train.data, self.dataset_test.data), axis=0)
        self.targets = self.dataset_train.targets + self.dataset_test.targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.data)


class STL10_ALL(data.Dataset):
    """Use both labeled and unlabeled STL10 data"""
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        split = 'train+unlabeled' if split == 'train' else 'test'
        base = datasets.STL10(root=root, split=split, download=download)
        self.data = base.data
        self.targets = base.labels

    def __getitem__(self, index):
        img = Image.fromarray(np.transpose(self.data[index], (1, 2, 0)))
        target = int(self.targets[index]) if self.targets is not None else -1
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.data)


class MNIST_ALL(data.Dataset):
    """Merge MNIST train + test and convert to RGB for consistency"""
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        transform_rgb = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
        ])
        base_train = datasets.MNIST(root=root, train=True, download=download)
        base_test = datasets.MNIST(root=root, train=False, download=download)

        imgs = np.concatenate((base_train.data.numpy(), base_test.data.numpy()), axis=0)
        self.data = np.stack([transform_rgb(Image.fromarray(img)) for img in imgs])
        self.targets = base_train.targets.tolist() + base_test.targets.tolist()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.data)


class COIL100_ALL(data.Dataset):
    """Load COIL100 dataset from local directory (/kaggle/input/coil100/coil-100/coil-100/)"""
    def __init__(self, root="/kaggle/input/coil100/coil-100/coil-100/", train=True, transform=None):
        from glob import glob
        self.img_paths = sorted(glob(os.path.join(root, "*.png")))
        self.transform = transform
        self.targets = list(range(len(self.img_paths)))  # pseudo labels

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0, index  # dummy label for unsupervised use

    def __len__(self):
        return len(self.img_paths)
        
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TransformThrice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        out3 = self.transform(inp)
        return out1, out2, out3


# Dictionary of transforms
dict_transform = {
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'cifar_train_simclr': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'cifar_train_justflip': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'stl10_train': transforms.Compose([
        transforms.Pad(12),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.44671062, 0.43980984, 0.40664645), (0.26034098, 0.25657727, 0.27126738)),
    ]),
    'stl10_train_justflip': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.44671062, 0.43980984, 0.40664645), (0.26034098, 0.25657727, 0.27126738)),
    ]),
    'stl10_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.44671062, 0.43980984, 0.40664645), (0.26034098, 0.25657727, 0.27126738)),
    ]),
    'reuters': transforms.Compose([
        transforms.ToTensor(),
    ])
}
dict_transform.update({
    'cifar100_train': dict_transform['cifar_train'],
    'cifar100_test': dict_transform['cifar_test'],
    'mnist_train': transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'mnist_test': transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'coil100_train': transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'coil100_test': transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
})
