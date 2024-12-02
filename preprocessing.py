import os
import pathlib
import shutil
import natsort
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split

class readDataset:
    def __init__(self, imagesPathes, masksPathes, output_dir):
        self.imagesPathes = imagesPathes
        self.masksPathes = masksPathes
        self.output_dir = output_dir
        self.images = None
        self.masks = None
        self.val_images = None
        self.val_masks = None
        self.test_images = None
        self.test_masks = None
        
        # Create output directories with the new structure
        os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', 'masks'), exist_ok=True)
        
    def readPathes(self,):
        self.images = natsort.natsorted(list(pathlib.Path(self.imagesPathes).glob('*.*')))
        self.masks = natsort.natsorted(list(pathlib.Path(self.masksPathes).glob('*.*')))
        try:
            shutil.rmtree(os.path.join(self.imagesPathes, ".ipynb_checkpoints"))
            shutil.rmtree(os.path.join(self.masksPathes, ".ipynb_checkpoints"))
            print(f".ipynb_checkpoints directory deleted successfully.")
        except Exception as e:
            print(f"just checking .ipynb_checkpoints (nothing)")
        
    def readImages(self, data, typeData):
        images = []
        height = 256
        width = 512
        for img in data:
            img_name = img.name
            img = cv2.imread(str(img), 0)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            if typeData == 'm':
                img = np.where(img > 0, 1, 0)
            img = np.expand_dims(img, axis=-1)
            images.append(img)
        print("(INFO..) Read Image Done")
        return np.array(images)

    def normalizeImages(self, images):
        normalized_images = []
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(3,3))
        for img in images:
            img = img.astype(np.uint8)
            img = clahe.apply(img)
            img = img / 255.
            img = np.expand_dims(img, axis=-1)
            normalized_images.append(img)
        print("(INFO..) Normalization Image Done")
        return np.array(normalized_images)
            
    def dataAugmentation(self, images, masks):
        augmentation = A.Compose([
            A.HorizontalFlip(p=1),
            # Vertical Translation
            A.ShiftScaleRotate(
                shift_limit_x=0,
                shift_limit_y=(-0.1, 0.05),
                scale_limit=(-0.05, 0.05), 
                rotate_limit=0,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_AREA,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            
            A.RandomBrightnessContrast(p=0.5),
        ], bbox_params=None)
        
        if len(images) != len(masks):
            raise ValueError("Number of images and masks must be the same.")

        augmented_images = []
        augmented_masks = []

        for image, mask in zip(images, masks):
            # Original image and mask
            augmented_images.append(image)
            augmented_masks.append(mask)

            # Perform augmentations
            # Generate 6 augmented versions per image
            for _ in range(6):
                # Ensure image and mask are in the right format
                # Squeeze if needed and ensure correct dimensionality
                img = image.squeeze()
                msk = mask.squeeze()
                
                # Handle single-channel images
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                if len(msk.shape) == 2:
                    msk = np.expand_dims(msk, axis=-1)

                # Apply augmentation
                augmented = augmentation(image=img, mask=msk)
                
                augmented_images.append(augmented['image'])
                augmented_masks.append(augmented['mask'])

        print("(INFO..) Augmentation Image Done")
        return np.array(augmented_images), np.array(augmented_masks)
    
    def saveAugmentedData(self, images, masks, split_type):
        for i, (img, mask) in enumerate(zip(images, masks)):
            # Create unique filenames
            img_filename = f"{split_type}_{i:04d}.png"
            mask_filename = f"{split_type}_{i:04d}_mask.png"
            
            # Save paths
            img_save_path = os.path.join(self.output_dir, split_type, 'images', img_filename)
            mask_save_path = os.path.join(self.output_dir, split_type, 'masks', mask_filename)
            
            # Save images (denormalize if needed)
            img_to_save = (img * 255).astype(np.uint8)
            mask_to_save = (mask.squeeze() * 255).astype(np.uint8)
            
            cv2.imwrite(img_save_path, img_to_save)
            cv2.imwrite(mask_save_path, mask_to_save)
    
    def splitDataset(self, images, masks, val_size=0.1, test_size=0.1, random_state=42):
        data = list(zip(images, masks))
        train_data, test_data = train_test_split(data, test_size=(val_size + test_size), random_state=random_state)
        val_data, test_data = train_test_split(test_data, test_size=(test_size / (val_size + test_size)), random_state=random_state)

        train_images, train_masks = zip(*train_data)
        val_images, val_masks = zip(*val_data)
        test_images, test_masks = zip(*test_data)
        
        # Augment train data
        train_images_aug, train_masks_aug = self.dataAugmentation(np.array(train_images), np.array(train_masks))
        
        # Save augmented data
        self.saveAugmentedData(train_images_aug, train_masks_aug, 'train')
        self.saveAugmentedData(val_images, val_masks, 'val')
        self.saveAugmentedData(test_images, test_masks, 'test')
        
        print("(INFO..) Splitting and Saving Data Done")
        return (np.array(train_images_aug), np.array(train_masks_aug), 
                np.array(val_images), np.array(val_masks), 
                np.array(test_images), np.array(test_masks))


images_path = './dataset/images'
masks_path = './dataset/masks'
output_dir = './dataset/prepos_dataset'
    
dataset = readDataset(images_path, masks_path, output_dir)
dataset.readPathes()
    
# Read images
images = dataset.readImages(dataset.images, 'i')
masks = dataset.readImages(dataset.masks, 'm')
    
# Normalize images
normalized_images = dataset.normalizeImages(images)
normalized_masks = dataset.normalizeImages(masks)
    
# Split and augment dataset
dataset.splitDataset(normalized_images, normalized_masks)