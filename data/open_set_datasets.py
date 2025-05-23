from data.cifar import get_cifar_10_10_datasets, get_cifar_10_100_datasets
from data.tinyimagenet import get_tiny_image_net_datasets
#from data.svhn import get_svhn_datasets
#from data.mnist import get_mnist_datasets
from data.cub import get_cub_datasets
from data.stanford_cars import get_scars_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.cifar100 import get_cifar_100_10_datasets
#from data.pku_aircraft import get_pku_aircraft_datasets

from data.open_set_splits.osr_splits import osr_splits
from data.augmentations import get_transform
#from config import osr_split_dir

osr_split_dir = '/fs/nexus-scratch/amit314/Project_OOD_detection/Project_OSR/osr_closed_set_all_you_need/data/open_set_splits'

import os
import sys
import pickle
import torch

"""
For each dataset, define function which returns:
    training set
    validation set
    open_set_known_images
    open_set_unknown_images
"""

get_dataset_funcs = {
    'cifar-10-100': get_cifar_10_100_datasets,
    'cifar-10-10': get_cifar_10_10_datasets,
    'cifar-100-10':get_cifar_100_10_datasets,#OOD detection all classes 
    #'mnist': get_mnist_datasets,
    #'svhn': get_svhn_datasets,
    'tinyimagenet': get_tiny_image_net_datasets,
    'cub': get_cub_datasets,
    'scars': get_scars_datasets,
    'aircraft': get_aircraft_datasets,
    #'pku-aircraft': get_pku_aircraft_datasets
}

def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading datasets...')

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                  train_classes=train_classes,
                                  open_set_classes=open_set_classes,
                                  balance_open_set_eval=balance_open_set_eval,
                                  split_train_val=split_train_val,
                                  seed=seed)
    else:
        raise NotImplementedError

    return datasets

def get_class_splits(dataset, split_idx=0, cifar_plus_n=10, cub_osr='all'):

    if dataset in ('cifar-10-10', 'mnist', 'svhn'):
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(10) if x not in train_classes]

    elif dataset == 'cifar-10-100':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = osr_splits['cifar-10-100-{}'.format(cifar_plus_n)][split_idx]

    elif dataset == 'tinyimagenet':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(200) if x not in train_classes]

    elif dataset == 'cub':

        osr_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)
        
        if(split_idx==0):
           train_classes = class_info['known_classes']
        if(split_idx==1):
            train_classes = [193, 144, 76, 166, 99, 81, 9, 177, 179, 94, 91, 199, 39, 198, 133, 131, 149, 48, 157, 136, 26, 95, 69, 70, 150, 80, 174, 161, 19, 7, 12, 146, 53, 82, 135, 31, 186, 28, 23, 93, 88, 44, 171, 176, 162, 87, 184, 67, 119, 124, 30, 97, 172, 197, 15, 51, 196, 38, 43, 189, 147, 98, 141, 50, 167, 55, 40, 163, 0, 106, 158, 74, 61, 137, 111, 121, 27, 84, 11, 17, 79, 126, 134, 190, 168, 77, 5, 143, 178, 34, 72, 129, 154, 42, 64, 89, 123, 18, 56, 116]
        if(split_idx==2):
            train_classes = [116, 31, 5, 19, 15, 146, 179, 172, 89, 97, 51, 186, 42, 131, 166, 91, 190, 119, 189, 171, 167, 87, 144, 163, 157, 198, 7, 67, 124, 197, 196, 98, 53, 158, 17, 154, 9, 64, 38, 43, 150, 76, 199, 61, 121, 133, 141, 126, 0, 129, 34, 193, 99, 84, 55, 18, 161, 147, 162, 81, 39, 80, 176, 72, 184, 11, 95, 30, 27, 77, 48, 56, 174, 74, 134, 136, 178, 28, 168, 93, 143, 79, 137, 123, 94, 106, 177, 82, 50, 44, 135, 111, 12, 69, 26, 149, 70, 88, 40, 23]
        if(split_idx==3):
            train_classes = [157, 133, 99, 40, 27, 74, 30, 144, 48, 31, 79, 172, 146, 53, 119, 88, 55, 199, 93, 171, 174, 9, 166, 177, 162, 17, 176, 116, 168, 23, 38, 126, 7, 94, 137, 82, 135, 154, 106, 77, 18, 64, 136, 56, 72, 43, 158, 95, 131, 193, 196, 15, 147, 161, 141, 197, 26, 163, 190, 184, 34, 189, 42, 124, 123, 84, 61, 87, 12, 89, 111, 28, 11, 186, 198, 149, 121, 97, 5, 51, 19, 91, 67, 76, 50, 134, 70, 69, 80, 0, 143, 44, 39, 98, 178, 179, 129, 167, 150, 81]
        if(split_idx==4):
            train_classes = [43, 126, 116, 166, 31, 30, 70, 50, 161, 88, 5, 172, 121, 178, 190, 81, 0, 143, 84, 55, 94, 146, 179, 150, 44, 48, 39, 99, 133, 123, 18, 134, 136, 168, 199, 135, 186, 56, 93, 97, 34, 98, 171, 198, 28, 79, 40, 131, 42, 87, 106, 141, 119, 27, 137, 147, 177, 154, 53, 95, 149, 167, 15, 82, 163, 89, 19, 72, 61, 64, 158, 196, 12, 176, 69, 157, 38, 91, 51, 74, 9, 124, 23, 162, 184, 77, 11, 76, 189, 111, 7, 67, 197, 144, 17, 26, 129, 80, 193, 174]



        open_set_classes = class_info['unknown_classes']
        #if(cub_osr=='easy'):
        #  open_set_classes['easy'] = open_set_classes['Easy']
        #elif(cub_osr=='hard'):
        #  open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] 
        if(cub_osr=='all'):
          open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
        
    elif dataset == 'aircraft':

        osr_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        if(split_idx==0):
           train_classes = class_info['known_classes']
        if(split_idx==1):
            train_classes = [71, 0, 39, 29, 47, 58, 64, 4, 45, 63, 1, 17, 30, 56, 24, 46, 16, 5, 92, 27, 22, 66, 79, 73, 28, 77, 23, 37, 57, 53, 3, 14, 2, 11, 65, 36, 67, 21, 76, 44, 33, 38, 99, 10, 43, 48, 52, 95, 19, 41]

        if(split_idx==2):
            train_classes = [21, 23, 56, 4, 38, 17, 64, 76, 27, 30, 33, 29, 79, 24, 47, 57, 43, 2, 19, 1, 63, 65, 44, 5, 45, 36, 46, 14, 58, 16, 77, 3, 67, 53, 52, 71, 22, 92, 73, 95, 10, 37, 28, 48, 66, 99, 11, 41, 39, 0]

        if(split_idx==3):
            train_classes = [39, 17, 4, 43, 24, 95, 11, 28, 16, 2, 64, 29, 92, 3, 45, 41, 99, 53, 52, 44, 71, 67, 56, 57, 37, 23, 73, 63, 27, 0, 36, 21, 58, 66, 79, 47, 76, 30, 1, 77, 33, 10, 48, 38, 14, 65, 22, 46, 19, 5]

        if(split_idx==4):
            train_classes = [65, 45, 21, 3, 63, 99, 2, 24, 29, 27, 56, 67, 17, 58, 53, 28, 22, 57, 14, 11, 71, 33, 52, 36, 64, 47, 41, 16, 76, 46, 48, 10, 37, 73, 38, 19, 79, 4, 39, 44, 0, 1, 43, 92, 30, 95, 23, 77, 5, 66]

        open_set_classes = class_info['unknown_classes']
        #if(cub_osr=='easy'):
        #  open_set_classes = open_set_classes['Easy']
        #elif(cub_osr=='hard'):
        #  open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium']
        if(cub_osr=='all'):
          open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif dataset == 'scars':

        osr_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        
        if(split_idx==0):
           train_classes = class_info['known_classes']
        if(split_idx==1):
            train_classes = [22, 161, 93, 157, 149, 122, 136, 159, 7, 172, 105, 41, 95, 84, 139, 195, 129, 160, 162, 140, 181, 145, 151, 138, 123, 163, 194, 158, 153, 141, 46, 97, 192, 191, 178, 167, 0, 154, 104, 53, 1, 188, 28, 25, 189, 146, 180, 11, 173, 133, 184, 102, 182, 134, 16, 168, 2, 170, 148, 44, 135, 193, 127, 142, 9, 143, 125, 186, 171, 185, 137, 175, 176, 155, 187, 164, 54, 100, 144, 174, 169, 112, 81, 166, 152, 147, 20, 75, 82, 50, 156, 150, 165, 26, 177, 117, 98, 38]

        if(split_idx==2):
            train_classes = [182, 191, 95, 25, 152, 160, 177, 123, 145, 155, 167, 186, 28, 194, 188, 137, 134, 154, 136, 180, 81, 175, 156, 162, 170, 26, 141, 157, 20, 178, 184, 1, 148, 166, 174, 97, 7, 181, 41, 139, 75, 146, 144, 38, 161, 163, 100, 138, 112, 22, 151, 173, 142, 176, 93, 54, 193, 153, 165, 9, 125, 11, 168, 195, 0, 192, 135, 147, 150, 171, 158, 189, 133, 105, 16, 46, 104, 129, 164, 98, 185, 187, 84, 2, 169, 140, 159, 127, 149, 143, 102, 172, 122, 53, 82, 50, 117, 44]

        if(split_idx==3):
            train_classes = [98, 173, 135, 50, 193, 168, 139, 38, 156, 186, 100, 162, 147, 123, 153, 194, 144, 146, 28, 134, 25, 54, 167, 102, 161, 82, 141, 53, 152, 133, 189, 180, 169, 149, 170, 165, 41, 172, 181, 151, 75, 95, 105, 175, 174, 166, 122, 104, 1, 81, 187, 46, 44, 0, 112, 11, 185, 138, 154, 20, 84, 184, 195, 158, 188, 155, 16, 171, 142, 182, 159, 140, 137, 145, 117, 127, 9, 178, 2, 97, 7, 150, 163, 176, 164, 157, 143, 192, 136, 148, 129, 125, 160, 26, 177, 191, 22, 93]

        if(split_idx==4):
            train_classes = [153, 82, 123, 182, 174, 41, 173, 102, 168, 127, 22, 163, 44, 169, 146, 26, 151, 180, 100, 53, 38, 181, 7, 141, 167, 187, 50, 25, 171, 161, 135, 144, 46, 160, 133, 170, 195, 105, 143, 155, 11, 158, 137, 125, 192, 188, 154, 185, 162, 147, 189, 134, 28, 81, 112, 184, 152, 193, 97, 186, 149, 175, 159, 93, 54, 0, 157, 177, 172, 142, 140, 16, 150, 139, 84, 136, 145, 95, 129, 178, 98, 166, 156, 164, 194, 138, 75, 122, 104, 191, 148, 176, 165, 2, 20, 9, 1, 117]



        open_set_classes = class_info['unknown_classes']
        #if(cub_osr=='easy'):
        #  open_set_classes = open_set_classes['Easy']
        #elif(cub_osr=='hard'):
        #  open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium']
        if(cub_osr=='all'):
            open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif dataset == 'pku-aircraft':
        print('Warning: PKU-Aircraft dataset has only one open-set split')
        train_classes = list(range(180))
        open_set_classes = list(range(120))
    elif dataset == 'cifar-100-10':
        train_classes = list(range(100))
        open_set_classes = list(range(10))
    else:

        raise NotImplementedError

    return train_classes, open_set_classes

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
