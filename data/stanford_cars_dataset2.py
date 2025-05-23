import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader 
from scipy import io as mat_io


class CarsDataset2(Dataset):

    def __init__(self, data_dir, train=True, limit=0, transform=None):


        super().__init__()
        self.transform = transform
        self.loader = default_loader
        self.limit = limit
        self.train = train # Store train flag if needed later, though not used in current logic

        root = Path(data_dir)
        train_path = root / 'train'
        meta_path = root / 'devkit' / 'cars_meta.mat'
        if not train_path.is_dir():
             raise FileNotFoundError(f"Training directory not found at: {train_path}")

        # --- 1. Build class_to_idx mapping from the 'train' directory ---

        labels_meta = mat_io.loadmat(str(meta_path))

        self.class_to_idx = {}
        self.classes = []

        class_names_array = labels_meta['class_names'][0]
        for i, name_array in enumerate(class_names_array):
                class_name = name_array[0] 
                if(class_name == 'Ram C/V Cargo Van Minivan 2012'):
                    class_name = 'Ram C_V Cargo Van Minivan 2012' # naming problem
                
                self.class_to_idx[class_name] = i 
                self.classes.append(class_name)
        #print(self.classes)
        #print(self.class_to_idx)
        num_classes = len(self.classes)
        print(f"Loaded {num_classes} classes from metadata file.")
        # --- 2. Load image paths and labels for the target split---
        self.data = []
        self.target = []

        split_name = 'train' if train else 'test'
        split_path = root / split_name
        if not split_path.is_dir():
             raise FileNotFoundError(f"{split_name.capitalize()} directory not found at: {split_path}")

        print(f"Loading data from: {split_path}")
        # Iterate through class folders within the specified split (train or test)
        for class_dir in sorted([d for d in split_path.iterdir() if d.is_dir()]):
            class_name = class_dir.name
            if class_name in self.class_to_idx:
                label = self.class_to_idx[class_name]
                
                for image_file in sorted(class_dir.glob('*.jpg')): 
                    self.data.append(str(image_file)) 
                    self.target.append(label)
            else:
            
                 print(f"Warning: Class '{class_name}' found in '{split_name}' split but not in train set. Skipping images in this directory.")

        print(f"Found {len(self.data)} images in the {split_name} split.")

        #print(self.target)

        # --- 4. Final setup ---
        self.uq_idxs = np.array(range(len(self.data))) 
        self.target_transform = None 

        if not self.data:
            raise RuntimeError(f"No images found in {split_path} matching the criteria.")

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = self.data[idx]
        target = self.target[idx] 

        try:
            image = self.loader(img_path)
        except Exception as e:
            print(f"Error loading image: {img_path}")
            print(e)
            
            raise RuntimeError(f"Failed to load image {img_path}") from e


        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        original_index = self.uq_idxs[idx] # This is just idx itself

        return image, target, original_index


