import os
import pathlib
import shutil
import natsort
import cv2
import numpy as np
import tensorflow as tf
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
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks', 'test'), exist_ok=True)
        
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
        height = 512
        width = 1024
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
        if len(images) != len(masks):
            raise ValueError("Number of images and masks must be the same.")

        augmented_images = []
        augmented_masks = []

        for image, mask in zip(images, masks):
            # Original images
            augmented_images.append(image)
            augmented_masks.append(mask)

            # Horizontal Flip
            flipped_image = tf.image.flip_left_right(image)
            flipped_mask = tf.image.flip_left_right(mask)
            augmented_images.append(flipped_image)
            augmented_masks.append(flipped_mask)

            # Random Contrast Adjustment
            for _ in range(2):
                # Generate random contrast factor between 0.5 and 1.5
                contrast_factor = tf.random.uniform([], minval=0.5, maxval=1.5)
                contrasted_image = tf.image.adjust_contrast(tf.cast(image, tf.float32), contrast_factor)
                contrasted_image = tf.cast(contrasted_image, image.dtype)
                augmented_images.append(contrasted_image)
                augmented_masks.append(mask)

            # Random Brightness Adjustment
            for _ in range(2):
                # Generate random brightness delta between -0.3 and 0.3
                brightness_delta = tf.random.uniform([], minval=-0.3, maxval=0.3)
                brightened_image = tf.image.adjust_brightness(tf.cast(image, tf.float32), delta=brightness_delta)
                brightened_image = tf.cast(brightened_image, image.dtype)
                augmented_images.append(brightened_image)
                augmented_masks.append(mask)
                
            # Random Vertical Translation
            num_translations = 2  # Number of random translations
            for _ in range(num_translations):
                # Generate random translation between -50 and 50
                shift = tf.random.uniform([], minval=-50, maxval=50, dtype=tf.int32).numpy()

                # Create a copy of the original image and mask
                translated_image = image.copy().squeeze()
                translated_mask = mask.copy().squeeze()

                # Create a blank canvas with the same shape as the original image
                canvas_image = np.zeros_like(translated_image)
                canvas_mask = np.zeros_like(translated_mask)

                # Determine shift direction
                if shift > 0:
                    # Shift down
                    canvas_image[shift:, :] = translated_image[:-shift, :]
                    canvas_mask[shift:, :] = translated_mask[:-shift, :]
                else:
                    # Shift up
                    canvas_image[:translated_image.shape[0]+shift, :] = translated_image[-shift:, :]
                    canvas_mask[:translated_mask.shape[0]+shift, :] = translated_mask[-shift:, :]

                # Reshape and expand dimensions
                canvas_image = np.expand_dims(canvas_image, axis=-1)
                canvas_mask = np.expand_dims(canvas_mask, axis=-1)

                augmented_images.append(canvas_image)
                augmented_masks.append(canvas_mask)

        print("(INFO..) Augmentation Image Done")
        return np.array(augmented_images), np.array(augmented_masks)
    
    def saveAugmentedData(self, images, masks, split_type):
        for i, (img, mask) in enumerate(zip(images, masks)):
            # Create unique filenames
            img_filename = f"{split_type}_{i:04d}.png"
            mask_filename = f"{split_type}_{i:04d}_mask.png"
            
            # Save paths
            img_save_path = os.path.join(self.output_dir, 'images', split_type, img_filename)
            mask_save_path = os.path.join(self.output_dir, 'masks', split_type, mask_filename)
            
            # Save images (denormalize if needed)
            img_to_save = (img * 255).astype(np.uint8)
            mask_to_save = (mask * 255).astype(np.uint8)
            
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
output_dir = './prepos_dataset'
    
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