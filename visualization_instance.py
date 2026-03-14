import torch
import torchvision.transforms as transforms
import numpy as np
import PIL
import matplotlib.pyplot as plt
from PIL import ImageFile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import  DataLoader
import pandas as pd
import torch.nn.functional as F
import gc
# models
import timm
import torch.nn as nn
from tqdm import tqdm
from model import  BagLevelSwinModelWithProxyFeature
from data_loader import Vpix_Dataset
from gradientCAM.AGCAM import AGCAM
import json
import argparse
from data_loader import generate_pattern

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = 'timm/swin_base_patch4_window7_224.ms_in1k'
IMG_SIZE = 224

classes = ["Glioma", "Meningioma", "Non-tumor", "Other Tumors", "PitNET", "Schwannoma"]



def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.4560, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return tensor * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)


def process_single_image(image, processor, annotation):
    processed_image = processor(image)
    if annotation == "N":  
        additional_channel = np.zeros((processed_image.shape[1], processed_image.shape[2]), dtype=np.float32)
    elif annotation == "C":  
        additional_channel = generate_pattern(processed_image, inverted=True)
    else:  
        additional_channel = generate_pattern(processed_image)

    additional_channel = torch.tensor(additional_channel, dtype=torch.float32).unsqueeze(0)
    processed_image = torch.cat([processed_image, additional_channel], dim=0)
    return processed_image.unsqueeze(0)  # Add batch dim


def load_4channel_swin(model_path: str, device: str):

    model = BagLevelSwinModelWithProxyFeature(
            model_name='swin_base_patch4_window7_224', 
            num_classes=len(classes),
            num_heads=8,
            max_instance_per_bag= 10,
            relu = True,
            sigmoid = True
        )

    # Get original Conv information
    original_conv = model.feature_extractor.backbone.patch_embed.proj
    
    # Modify to 4 channels (in_channels=4)
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=(original_conv.bias is not None)
    )
    
    
    
    model.feature_extractor.backbone.patch_embed.proj = new_conv
    
    # Load weights
    state_dict = torch.load(model_path)
    
    # Save instance evaluation only
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if "image_classifier" in k:
            new_key = k.replace("image_classifier", "instance_classifier")
        new_state_dict[new_key] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Visualization Script for Single Image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the single image file')
    parser.add_argument('--save_dir', type=str, default='./visualization_results', help='Path to save results')
    parser.add_argument('--annotation', type=str, default='', help='Annotation type (e.g., "N", "C", or "P")')
    parser.add_argument('--model_paths', nargs='+', default=[
        'trained_models/fold4/best_swin_epoch3_acc0.6695.pt',
        'trained_models/fold1/best_swin_epoch14_acc0.6039.pt',
        'trained_models/fold2/best_swin_epoch47_acc0.6746.pt',
        'trained_models/fold3/best_swin_epoch5_acc0.6640.pt',
        'trained_models/fold5/best_swin_epoch22_acc0.6147.pt'
    ], help='List of model paths, model1 will be the main model')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    model_dict = {}
    for i, path in enumerate(args.model_paths):
        print(f"Loading model {i+1}: {path}")
        model_dict[f'model_{i+1}'] = load_4channel_swin(path, device=device)
    
    # Set all models to eval mode
    for m in model_dict.values():
        m.eval()
    
    # Initialize AGCAM
    ours_method = AGCAM(list(model_dict.values())[0], layer_num=1)
    
    
    processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    print(f"Processing image: {image_path}")
    from PIL import Image
    pil_img = Image.open(image_path).convert("RGB")
    
    images = process_single_image(pil_img, processor, args.annotation)
    
    imagename = os.path.basename(image_path).split('.')[0]
    subject_name = "SingleSubject"

    # Result saving folder
    save_folder = args.save_dir
    
    if not os.path.exists(save_folder):
        print(f"Creating save folder: {save_folder}")
        os.makedirs(save_folder, exist_ok=True)
    else:
        print(f"Save folder already exists: {save_folder}")

    # -----------------------------
    # Single image processing
    # -----------------------------
    # (1) Move data to GPU
    if device == 'cuda': torch.cuda.synchronize()
    
    images = images.to(device)

    if device == 'cuda': torch.cuda.synchronize()

    # (2) Original Model Inference
    if device == 'cuda': torch.cuda.synchronize()

    # Original Image Inference for Prediction and Confidence
    with torch.no_grad():
         output_orig = ours_method.model(images)
         confidence_orig = F.softmax(output_orig, dim=1)
         pred_idx = int(torch.argmax(confidence_orig, dim=1).item())

    if device == 'cuda': torch.cuda.synchronize()

    # (3) GradCAM Generation
    if device == 'cuda': torch.cuda.synchronize()

    with torch.set_grad_enabled(True):
        predictions, ours_heatmap_list, att_heatmap, grad_heatmap, confidence = ours_method.generate(images, cls_idx=pred_idx)
        single_grad_heatmap = torch.stack(grad_heatmap[0:2], dim=0).mean(dim=0)
        single_grad_heatmap = single_grad_heatmap.detach().cpu()

    del ours_heatmap_list, att_heatmap, grad_heatmap

    if device == 'cuda': torch.cuda.synchronize()

    single_grad_heatmap = single_grad_heatmap.squeeze(0).numpy()  # shape: (224, 224)
    
    confidence_list = [round(float(v), 6) for v in confidence_orig.flatten()]  
    confidence_dict = {cls_name: confidence_list[i] for i, cls_name in enumerate(classes)}
    
    # (4) Ensemble Inference
    if device == 'cuda': torch.cuda.synchronize()

    ensemble_probs = []
    for key, model in model_dict.items():
        if key == 'model_1':
            probs = confidence_orig
        else:
            with torch.no_grad():
                output = model(images)
            probs = torch.softmax(output, dim=1)
        
        ensemble_probs.append(probs[0, pred_idx].item())
        
    sum_probs = sum(ensemble_probs)
    final_conf = sum_probs / len(ensemble_probs)
    
    if device == 'cuda': torch.cuda.synchronize()
            
    # (5) Visualization with matplotlib
    
    orig_img = images[0, :3].cpu().detach()
    orig_img_unnorm = unnormalize(orig_img.unsqueeze(0)).squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
    
    hm = single_grad_heatmap
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    heatmap_colored = plt.get_cmap('jet')(hm)[:, :, :3]
    
    overlay = 0.5 * orig_img_unnorm + 0.5 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Adjust plot ratio (reduce text space)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1, 0.4]})
    
    axes[0].imshow(orig_img_unnorm)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title('GradCAM Overlay')
    axes[1].axis('off')
    
    axes[2].axis('off')
    pred_idx_int = int(pred_idx)
    text_info = (
        f"Predicted: {classes[pred_idx_int]}\n"
        f"Confidence: {final_conf:.4f}\n"
    )

    # Remove text box (bbox) and adjust position
    axes[2].text(0.0, 0.5, text_info, fontsize=14, verticalalignment='center')
    
    # Save path
    save_path_img = os.path.join(
        save_folder,
        f"{imagename}_cam.png"
    )
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
    plt.savefig(save_path_img, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved result to {save_path_img}")
    print("-" * 30)
    print("Done!")

if __name__ == "__main__":
    main()
    
    
