from torchvision import datasets, transforms
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np
import os
from math import inf
from scipy import stats
from torch.nn import functional as F


MNIST_MEAN = [0.1307]
MNIST_STD = [0.3081]
CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_HUMAN_NOISE_PATH = 'data/CIFAR-10_human.pt'
CIFAR100_HUMAN_NOISE_PATH = 'data/CIFAR-100_human.pt'


def get_asym_T_mnist(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # 7 -> 1
    T[7, 7], T[7, 1] = 1. - eta, eta
    # 2 -> 7
    T[2, 2], T[2, 7] = 1. - eta, eta
    # 5 <-> 6
    T[5, 5], T[5, 6] = 1. - eta, eta
    T[6, 6], T[6, 5] = 1. - eta, eta
    # 3 -> 8
    T[3, 3], T[3, 8] = 1. - eta, eta
    
    return T

def get_asym_T_cifar10(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # truck -> automobile (9 -> 1)
    T[9, 9], T[9, 1] = 1. - eta, eta
    # bird -> airplane (2 -> 0)
    T[2, 2], T[2, 0] = 1. - eta, eta
    # cat <-> dog (3 <-> 5)
    T[3, 3], T[3, 5] = 1. - eta, eta
    T[5, 5], T[5, 3] = 1. - eta, eta
    # deer -> horse (4 -> 7)
    T[4, 4], T[4, 7] = 1. - eta, eta
    
    return T
    
def get_asym_T_cifar100(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 100
    num_superclasses = 20
    num_subclasses = 5

    T = np.eye(num_classes)

    for i in np.arange(num_superclasses):
        # build T for one superclass
        T_superclass = (1. - eta) * np.eye(num_subclasses)
        for j in np.arange(num_subclasses - 1):
            T_superclass[j, j + 1] = eta
        T_superclass[num_subclasses - 1, 0] = eta
        
        init, end = i * num_subclasses, (i + 1) * num_subclasses
        T[init:end, init:end] = T_superclass

    return T

def create_noisy_labels(labels, trans_matrix):
    '''
    create noisy labels from labels and noisy matrix
    '''
    
    if trans_matrix is None:
        raise ValueError('Noisy matrix is None')
    
    num_trans_matrix = trans_matrix.copy()
    labels = labels.copy()
    
    num_classes = len(trans_matrix)
    class_idx = [np.where(np.array(labels) == i)[0]
                 for i in range(num_classes)]
    num_samples_class = [len(class_idx[idx])
                         for idx in range(num_classes)]
    for real_label in range(num_classes):
        for trans_label in range(num_classes):
            num_trans_matrix[real_label][trans_label] = \
                trans_matrix[real_label][trans_label] * num_samples_class[real_label]
    num_trans_matrix = num_trans_matrix.astype(int)

    for real_label in range(num_classes):
        for trans_label in range(num_classes):

            if real_label == trans_label:
                continue

            num_trans = num_trans_matrix[real_label][trans_label]
            if num_trans == 0:
                continue

            trans_samples_idx = np.random.choice(class_idx[real_label],
                                                 num_trans,
                                                 replace=False)
            class_idx[real_label] = np.setdiff1d(class_idx[real_label],
                                                 trans_samples_idx)
            for idx in trans_samples_idx:
                labels[idx] = trans_label
    
    return labels

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    to_tensor = transforms.ToTensor()

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = to_tensor(x).cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

class MyMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets.numpy(), trans_matrix)

class MyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None, noisy_targets=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)
        if noisy_targets is not None:
            self.targets = noisy_targets.tolist()

class MyCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None, noisy_targets=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)
        if noisy_targets is not None:
            self.targets = noisy_targets.tolist()

class WebVisionDataset:
    def __init__(self, root, file_name='webvision_mini_train.txt',
                 transform=None, target_transform=None):
        self.target_list = []
        self.root = root
        self.load_file(os.path.join(root, file_name))
        self.transform = transform
        self.target_transform = target_transform
        return

    def load_file(self, filename):
        f = open(filename, "r")
        for line in f:
            train_file, label = line.split()
            if int(label) > 49:
                break
            self.target_list.append((train_file, int(label)))
        f.close()
        return

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        impath, target = self.target_list[index]
        img = Image.open(os.path.join(self.root, impath)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class ImageNetMini:
    def __init__(self, root, split='val', transform=None):
        self.new_targets = []
        self.target_list = []
        self.root_file = root
        #the file contains full path to the images of the first 50 classes and the label
        #read file
        with open(self.root_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            file, cls_id = line.strip().split()
            cls_id = int(cls_id)
            file = os.path.join(self.root_file, file)
            self.new_targets.append(cls_id)
            self.target_list.append((file, cls_id))

        self.imgs = self.target_list
        self.targets = self.new_targets
        self.samples = self.imgs
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
class Clothing1MDataset:
    def __init__(self, path, type='train', transform=None, target_transform=None):
        self.path = path
        if type == 'test':
            flist = os.path.join(path, "annotations/clean_test.txt")
        elif type == 'valid':
            flist = os.path.join(path, "annotations/clean_val.txt")
        elif type == 'train':
            flist = os.path.join(path, "annotations/noisy_train.txt")
        else:
            raise('Unknown type')

        self.imlist = self.flist_reader(flist)
        self.transform = transform

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = Image.open(impath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def flist_reader(self, flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impath = self.path + row[0]
                imlabel = row[1]
                imlist.append((impath, int(imlabel)))
        return imlist

class Clothing1MDatasetLoader:
    def __init__(self, train_batch_size=512, eval_batch_size=256, data_path=None, num_of_workers=4, use_cutout=True, cutout_length=112):
        if data_path is None:
            raise ValueError('data_path is None')
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_path = data_path
        self.num_of_workers = num_of_workers
        self.use_cutout = use_cutout 
        self.cutout_length = cutout_length
        self.data_loaders = self.loadData()

    def getDataLoader(self):
        return self.data_loaders['train_dataset'], self.data_loaders['valid_dataset'], self.data_loaders['test_dataset']

    def loadData(self):
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),                      # Resize to 256×256
            transforms.CenterCrop(224),                         # Center crop to 224×224
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)            # ImageNet normalization
        ])
        

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),                      # Resize to 256×256
            transforms.CenterCrop(224),                         # Center crop to 224×224
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)            # ImageNet normalization
        ])

        if self.use_cutout:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(self.cutout_length))

        train_dataset = Clothing1MDataset(path=self.data_path,
                                          type='train',
                                          transform=train_transform)

        test_dataset = Clothing1MDataset(path=self.data_path,
                                         type='test',
                                         transform=test_transform)

        valid_dataset = Clothing1MDataset(path=self.data_path,
                                          type='valid',
                                          transform=test_transform)

        data_loaders = {}
        from torch.utils.data import DataLoader
        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        data_loaders['valid_dataset'] = DataLoader(dataset=valid_dataset,
                                                   batch_size=self.eval_batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)
        return data_loaders
    
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
def mnist(root, noise_type, noise_rate, tuning=False):
    if noise_type == 'asym':
        T = get_asym_T_mnist(noise_rate)
    else:
        raise ValueError('Wrong noise type! Must be sym or asym')
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    
    if tuning:
        train_dataset = MyMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        num_train = int(len(train_dataset) * 0.8)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T
        
        eval_dataset = MyMNIST(root=root,
                               train=True,
                               transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:
        train_dataset = MyMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        
        eval_dataset = MyMNIST(root=root,
                               train=False,
                               transform=eval_transform)
    
    return train_dataset, eval_dataset

def cifar10(root, noise_type, noise_rate, seed, tuning=False):
    if noise_type == 'asym':
        T = get_asym_T_cifar10(noise_rate)
    elif noise_type == 'human':
        noisy_targets = torch.load(CIFAR10_HUMAN_NOISE_PATH,weights_only=False)['worse_label']
    elif noise_type == 'instance':
        #use instance-dependent noise
        #load cifar10 training set without any noise
        base_dataset = datasets.CIFAR10(root=root, train=True, download=True)
        base_data = base_dataset.data
        base_labels = np.array(base_dataset.targets)
        #convert to torch tensor
        base_data = torch.from_numpy(base_data).float()
        base_labels = torch.from_numpy(base_labels)
        feature_size = 3*32*32
        noisy_targets = get_instance_noisy_label(noise_rate, base_dataset, base_labels, 10, feature_size, 0.1, seed)
    else:
        raise ValueError('Wrong noise type! Must be either asym, human or instance')
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    
    if tuning:
        train_dataset = MyCIFAR10(root=root,
                                  train=True,
                                  transform=train_transform,
                                  trans_matrix=T)
        num_train = int(len(train_dataset) * 0.8)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T

        eval_dataset = MyCIFAR10(root=root,
                                 train=True,
                                 transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:#use human noise only for validation
        if noise_type not in ['human','instance']:
            train_dataset = MyCIFAR10(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=T)
        else:
            train_dataset = MyCIFAR10(root=root,
                                    train=True,
                                    transform=train_transform,
                                    noisy_targets=noisy_targets)
            

        eval_dataset = MyCIFAR10(root=root,
                                 train=False,
                                 transform=eval_transform)
    
    return train_dataset, eval_dataset

def cifar100(root, noise_type, noise_rate, seed, tuning=False):
    if noise_type == 'asym':
        T = get_asym_T_cifar100(noise_rate)
    elif noise_type == 'human':
        noisy_targets = torch.load(CIFAR100_HUMAN_NOISE_PATH,weights_only=False)['noisy_label']
    elif noise_type == 'instance':
        #use instance-dependent noise
        #load cifar100 training set without any noise
        base_dataset = datasets.CIFAR100(root=root, train=True, download=True)
        base_data = base_dataset.data
        base_labels = np.array(base_dataset.targets)
        #convert to torch tensor
        base_data = torch.from_numpy(base_data).float()
        base_labels = torch.from_numpy(base_labels)
        feature_size = 3*32*32
        noisy_targets = get_instance_noisy_label(noise_rate, base_dataset, base_labels, 100, feature_size, 0.1, seed)
    else:
        raise ValueError('Wrong noise type! Must be either sym,  asym, human or instance')
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    if tuning:
        train_dataset = MyCIFAR100(root=root,
                                   train=True,
                                   transform=train_transform,
                                   trans_matrix=T)
        num_train = int(len(train_dataset) * 0.8)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T

        eval_dataset = MyCIFAR100(root=root,
                                  train=True,
                                  transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:
        if noise_type not in ['human','instance']:
            train_dataset = MyCIFAR100(root=root,
                                    train=True,
                                    transform=train_transform,
                                    trans_matrix=T)
        else:
            train_dataset = MyCIFAR100(root=root,
                                    train=True,
                                    transform=train_transform,
                                    noisy_targets=noisy_targets)
        
        eval_dataset = MyCIFAR100(root=root,
                                  train=False,
                                  transform=eval_transform)

    return train_dataset, eval_dataset

def webvision(train_data_path, val_data_path, tuning=False):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    # use "mini" setting

    train_dataset = WebVisionDataset(root=train_data_path,
                                     file_name='info/train_filelist_google.txt',
                                     transform=train_transform)
    
    if tuning:
        num_train = int(len(train_dataset) * 0.8)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        
        eval_dataset = WebVisionDataset(root=train_data_path,
                                       file_name='info/train_filelist_google.txt',
                                       transform=test_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))
    else:
        eval_dataset = WebVisionDataset(root=val_data_path,
                                       file_name='info/val_filelist.txt',
                                       transform=test_transform)

    
    return train_dataset, eval_dataset
