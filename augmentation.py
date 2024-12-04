import os
import pathlib
import shutil
import natsort
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split

class augmentDataset:
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
        
    def readImages(self, data, typeData, width, height):
        images = []
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
        # clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(3,3))
        for img in images:
            img = img.astype(np.uint8)
            # img = clahe.apply(img)
            img = img / 255.
            # img = np.expand_dims(img, axis=-1)
            normalized_images.append(img)
        print("(INFO..) Normalization Image Done")
        return np.array(normalized_images)
            
    def dataAugmentation(self, images, masks, n_augments):
        augmentation = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            # Vertical Translation
            A.ShiftScaleRotate(
                shift_limit_x=(-0.05, 0.05),
                shift_limit_y=(-0.1, 0.05),
                scale_limit=(-0.05, 0.05), 
                rotate_limit=0,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_AREA,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomGamma(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.4),
                                       contrast_limit=(-0.25, 0.35),
                                       p=0.5),
            A.ElasticTransform(alpha=10, 
                               sigma=10, 
                               interpolation=cv2.INTER_AREA,
                               mask_interpolation=cv2.INTER_AREA,
                               p=0.2)
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
            for _ in range(n_augments):
                # Ensure image and mask are in the right format                
                image = image.astype(np.uint8)
                mask = mask.astype(np.uint8)
                
                # Handle single-channel images
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)

                # Apply augmentation
                augmented = augmentation(image=image, mask=mask)
                
                augmented_images.append(augmented['image'])
                augmented_masks.append(augmented['mask'])

        # Calculate and print total augmented images
        total_original_images = len(images)
        total_augmented_images = len(augmented_images)

        print(f"(INFO..) Original Train Images: {total_original_images}")
        print(f"(INFO..) Total Augmented Train Images: {total_augmented_images}")
        print(f"(INFO..) Augmentation Multiplier: {total_augmented_images / total_original_images:.2f}x")
        print("(INFO..) Augmentation Image Done")
        return np.array(augmented_images), np.array(augmented_masks)
    
    def saveData(self, images, masks, split_type):
        for i, (img, mask) in enumerate(zip(images, masks)):
            # Create unique filenames
            img_filename = f"{split_type}_{i:04d}.png"
            mask_filename = f"{split_type}_{i:04d}_mask.png"
            
            # Save paths
            img_save_path = os.path.join(self.output_dir, split_type, 'images', img_filename)
            mask_save_path = os.path.join(self.output_dir, split_type, 'masks', mask_filename)
            
            img_to_save = (img).astype(np.uint8)
            mask_to_save = (mask.squeeze() * 255).astype(np.uint8)
            
            cv2.imwrite(img_save_path, img_to_save)
            cv2.imwrite(mask_save_path, mask_to_save)
    
    def splitDataset(self, images, masks, val_size=20, test_size=10):
        # Only perform the split, return the split indices
        data = list(zip(images, masks))
        train_data, test_data = train_test_split(data, test_size=(val_size + test_size), random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=(test_size / (val_size + test_size)), random_state=42)

        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data
        }
    
    def main(self, width=512, height=256, val_size=20, test_size=10, n_augments=5):
        """
        Main method to handle the entire preprocessing workflow
        
        Args:
        - width (int): Resize width for images
        - height (int): Resize height for images
        - val_size (int): Percentage of data for validation
        - test_size (int): Percentage of data for testing
        """
        # Read image paths
        self.readPathes()
        
        # Read and process images
        images = self.readImages(self.images, 'i', width=width, height=height)
        masks = self.readImages(self.masks, 'm', width=width, height=height)
        
        # Split the dataset
        split_data = self.splitDataset(images, masks, val_size, test_size)
        
        # Extract split data
        train_data = split_data['train_data']
        val_data = split_data['val_data']
        test_data = split_data['test_data']
        
        # Separate images and masks
        train_images, train_masks = zip(*train_data)
        val_images, val_masks = zip(*val_data)
        test_images, test_masks = zip(*test_data)
        
        # Augment train data
        train_images_aug, train_masks_aug = self.dataAugmentation(np.array(train_images), np.array(train_masks), n_augments=n_augments)
        
        # Save augmented data
        self.saveData(train_images_aug, train_masks_aug, 'train')
        self.saveData(val_images, val_masks, 'val')
        self.saveData(test_images, test_masks, 'test')
        
        print("(INFO..) Dataset Preprocessing Complete")

images_path = './dataset/images'
masks_path = './dataset/masks'
output_dir = './dataset/prepos_dataset'

dataset = augmentDataset(images_path, masks_path, output_dir)
dataset.main(width=1024, height=512, val_size=20, test_size=10, n_augments=7)