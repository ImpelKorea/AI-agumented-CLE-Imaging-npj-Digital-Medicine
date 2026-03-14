from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_pattern(image, sigma=150, inverted=False):
    rows, cols = image.shape[1], image.shape[2]
    center_x, center_y = cols // 2, rows // 2

    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    mask = np.exp(-distance_from_center**2 / (2 * sigma**2))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    if inverted:
        mask = 1 - mask

    return mask.astype(np.float32)


class Vpix_Dataset(Dataset):
    def __init__(self, root_dir, processor, transform=None, mode="train", gradcam_processor=None):
        self.data = []
        self.processor = processor
        self.gradcam_processor = gradcam_processor
        self.transform = transform
        self.mode = mode  # Add mode parameter


        self.classes =  ["Glioma", "Meningioma", "Non-tumor", "Other Tumors", "PitNET", "Schwannoma"]
        
        self.subject_dict = {}
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for current_root, _, files in os.walk(class_path):
                for file_name in files:
                    if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):  
                        image_path = os.path.join(current_root, file_name)
                        imagename = file_name.split(".")[0]
                        
                        subject_folder_name = current_root.replace(class_path, "").strip(os.sep)
                        subject_folder_name = subject_folder_name.split(os.sep)[0]
                        if subject_folder_name.split("-")[-1].isdigit():
                            annotation = subject_folder_name.split("-")[-2]
                        else:
                            annotation = subject_folder_name.split("-")[-1]
                        
                        if subject_folder_name not in self.subject_dict.keys():
                            self.subject_dict[subject_folder_name] = len(self.subject_dict)
                        
                        unique_subject_idx = self.subject_dict[subject_folder_name]
                        
                        self.data.append(
                            (image_path, class_idx, unique_subject_idx,
                             subject_folder_name, imagename, annotation)
                        )

    def __len__(self):
        return len(self.data)

    def generate_vignetting(self, image, sigma=150, inverted=False):
        return generate_pattern(image, sigma, inverted)

    def __getitem__(self, idx):
        image_path, label, subject_idx, subject_name, imagename, annotation = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        
        def process_with_processor(proc):
            processed_image = proc(image)

            # Create additional channels
            if annotation == "N":  
                additional_channel = np.zeros((processed_image.shape[1], processed_image.shape[2]), dtype=np.float32)
            elif annotation == "C":  
                additional_channel = self.generate_vignetting(processed_image, inverted=True)
            else:  
                additional_channel = self.generate_vignetting(processed_image)

            additional_channel = torch.tensor(additional_channel, dtype=torch.float32).unsqueeze(0)
            processed_image = torch.cat([processed_image, additional_channel], dim=0)
            return processed_image

        processed_image = process_with_processor(self.processor)

        if self.transform:
            processed_image = self.transform(processed_image)
            
        if self.gradcam_processor:
            gradcam_image = process_with_processor(self.gradcam_processor)
            if self.transform:
                gradcam_image = self.transform(gradcam_image)
            return processed_image, gradcam_image, label, subject_idx, imagename, subject_name, annotation
        
        return processed_image, label, subject_idx, imagename, subject_name, annotation
