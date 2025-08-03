import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image

from model import UNetWithAttention
from dataset import get_test_loader
from utils import dice_coef, iou_score, sensitivity, specificity, mean_iou, accuracy, mean_pixel_accuracy, print_metrics

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

def permute_attention_projection(
    layer: nn.Linear, 
    permutation: Optional[torch.Tensor] = None,
    num_heads: Optional[int] = None,
    num_key_value_groups: Optional[int] = None,
) -> None:
    '''
    permute the weight of the q, k, v projection layer so that the head order is permuted
    '''
    
    if permutation is None:
        return

    assert permutation.dim() == 1

    assert permutation.shape[0] == num_heads
    if num_key_value_groups is not None and num_key_value_groups != 1:
        num_heads = num_heads * num_key_value_groups
        original_permutation_tensor = permutation
        original_permutation = permutation.tolist()
        permutation = []
        for index in original_permutation:
            permutation.extend([index * num_key_value_groups + i for i in range(num_key_value_groups)])

        # turn permutation to tensor
        permutation = torch.tensor(permutation, device=original_permutation_tensor.device, dtype=original_permutation_tensor.dtype)

    input_dim = layer.in_features
    output_dim = layer.out_features
    head_dim = output_dim // num_heads

    original_weights = layer.weight.data
    weights_reshaped = original_weights.reshape(num_heads, head_dim, input_dim)
    weights_permuted = weights_reshaped[permutation].reshape_as(original_weights)

    assert weights_permuted.is_contiguous()
        
    layer.weight.data = weights_permuted

    if layer.bias is not None:
        original_bias = layer.bias.data
        bias_reshaped = original_bias.reshape(num_heads, head_dim)
        bias_permuted = bias_reshaped[permutation].reshape_as(original_bias)

        assert bias_permuted.is_contiguous()

        layer.bias.data = bias_permuted
                                                                            
    return

def permute_output_projection(
    layer: nn.Linear, 
    permutation: Optional[torch.Tensor] = None,
    num_heads: Optional[int] = None,
    num_key_value_groups: Optional[int] = None,
) -> None:
    '''
    permute the weight of o_proj so that it can take in the permuted hidden states
    '''
    
    if permutation is None:
        return

    assert permutation.dim() == 1

    assert permutation.shape[0] == num_heads

    if num_key_value_groups is not None and num_key_value_groups != 1:
        num_heads = num_heads * num_key_value_groups
        original_permutation_tensor = permutation
        original_permutation = permutation.tolist()
        permutation = []
        for index in original_permutation:
            permutation.extend([index * num_key_value_groups + i for i in range(num_key_value_groups)])

        # turn permutation to tensor
        permutation = torch.tensor(permutation, device=original_permutation_tensor.device, dtype=original_permutation_tensor.dtype)


    input_dim = layer.in_features
    output_dim = layer.out_features
    head_dim = input_dim // num_heads

    original_weights = layer.weight.data
    weights_reshaped = original_weights.reshape(output_dim, num_heads, head_dim)
    weights_permuted = weights_reshaped.transpose(0,1)[permutation].transpose(0,1).reshape_as(original_weights)

    assert weights_permuted.is_contiguous()
        
    layer.weight.data = weights_permuted
                                                                            
    return
  

def permute_lut(
    lut: torch.Tensor,
    permutation: Optional[torch.Tensor] = None,
    num_heads: int = 32,
):

    if permutation is None:
        return lut
    
    assert permutation.dim() == 1
    assert permutation.shape[0] == num_heads
    assert lut.shape[0] == num_heads

    permuted_lut = lut[permutation]

    if not permuted_lut.is_contiguous():
        permuted_lut = permuted_lut.contiguous()

    return permuted_lut


def lut_to_permutation(
    lut_list: List[torch.Tensor],
    num_heads: int = 32,
    num_key_value_groups: int = 1,
):
    '''
    Find a permutation, such that after applying the permutation, the heads with same hyperparameters are clustered together

    Parameters:
        lut_list: (`List[torch.Tensor]`):
            A list of lut table
    
    Return:
        Permutation: (`torch.Tensor`):
            A 1D Tensor of shape `(num_heads,)` representing the original index of the heads before permutation
        Cluster: (`Dict[int, Tuple[int, int]]`):
            A dictionary of the form `{cluster_index: (start, end)}` where `start` and `end` are the range of the cluster after permutation
    '''

    assert lut_list[0].shape[0] == num_heads
    assert lut_list[0].dim() == 3
    assert num_heads % num_key_value_groups == 0

    # return a permutation, such that after applying the permutation, same heads clustered together
    sorted_indices_list = []
    cluster_list = []

    if num_key_value_groups > 1:
        # check that the pattern within a group is the same
        for i in range(num_key_value_groups):
            raise NotImplementedError

    for lut in lut_list:
        serialized_heads = [tuple(head.reshape(-1).tolist()) for head in lut]



        sorted_indices = sorted(range(len(serialized_heads)), key=lambda i: (serialized_heads[i], i))
        
        # get the cluster, in the form of dictionary. like {1:(0,4), 2:(4, 32)}. (0, 4) is the range
        cluster = {}
        current_cluster = 0
        start = 0
        for i in range(1, len(sorted_indices)):
            if serialized_heads[sorted_indices[i]] != serialized_heads[sorted_indices[i-1]]:
                cluster[current_cluster] = (start, i)
                start = i
                current_cluster += 1

        cluster[current_cluster] = (start, len(sorted_indices))

        sorted_indices_list.append(sorted_indices)
        cluster_list.append(cluster)

    # check those cluster
    two_pattern_idx = []
    for i in range(0, len(cluster_list)):
        if len(cluster_list[i]) == 2:
            two_pattern_idx.append(i)
        elif len(cluster_list[i]) > 2:
            print(f"cluster_list[{i}] = {cluster_list[i]}")
            raise NotImplementedError
                
    # all lut have just one pattern, return arbitrary one
    if len(two_pattern_idx) == 0:
        # manually make it two
        return_cluster = {0: (0, num_heads // 2), 1: (num_heads // 2, num_heads)}
        # return torch.tensor(sorted_indices_list[0]), cluster_list[0]
        return torch.tensor(sorted_indices_list[0]), return_cluster

    cluster = cluster_list[two_pattern_idx[0]]
    sorted_indices = sorted_indices_list[two_pattern_idx[0]]

    for i in two_pattern_idx:
        checked_lut = lut_list[i]
        checked_serialized_heads = [tuple(head.reshape(-1).tolist()) for head in checked_lut]
        # permute the head with sorted_indices
        permuted_serialized_heads = [checked_serialized_heads[j] for j in sorted_indices]

        # cluster the permuted heads
        current_cluster = 0
        start = 0
        checked_cluster = {}

        for j in range(1, len(permuted_serialized_heads)):
            if permuted_serialized_heads[j] != permuted_serialized_heads[j-1]:
                checked_cluster[current_cluster] = (start, j)
                start = j
                current_cluster += 1
        
        checked_cluster[current_cluster] = (start, len(permuted_serialized_heads))

        # check after applying the permuation, whether
        if checked_cluster != cluster:
            raise NotImplementedError

    # return any of the two pattern stuff    
    return torch.tensor(sorted_indices_list[two_pattern_idx[0]]), cluster_list[two_pattern_idx[0]]
    
def moa_config_to_permutation(moa_config) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[int, int]]]]:
    '''
    Find a permutation, such that after applying the permutation, the heads with same hyperparameters are clustered together

    Parameters:
        moa_config: (`Dict`):
            A dictionary containing the MoA config. Two keys are used:
                'alphas': (`Union[List[torch.Tensor], List[List[int]]]`): A list of alpha values for each layer
                'betas': (`Union[List[torch.Tensor], List[List[int]]]`): A list of beta values for each layer
    
    Return:
        Permutation: (`List[torch.Tensor]`):
            Each element the permutation of a layer. Each element is a 1D Tensor of shape `(num_heads,)` representing the original index of the heads before permutation
        Cluster: (`List[Dict[int, Tuple[int, int]]]`):
            Each element is a dictionary of the form `{cluster_index: (start, end)}` where `start` and `end` are the range of the cluster after permutation
    '''
    
    permutations = []
    clusters = []
    
    # Iterate over each layer
    for layer_index, (layer_alphas, layer_betas) in enumerate(zip(moa_config['alphas'], moa_config['betas'])):
        num_heads = len(layer_alphas)
        
        # Pair each alpha and beta with its index
        combined = list(zip(layer_alphas, layer_betas, range(num_heads)))
        
        # Sort based on alpha and beta values
        sorted_combined = sorted(combined, key=lambda x: (x[0], x[1]))
        
        # Extract the sorted indices
        sorted_indices = [x[2] for x in sorted_combined]
        
        # Create permutation tensor
        permutation_tensor = torch.tensor(sorted_indices)
        permutations.append(permutation_tensor)
        
        # Track clusters
        cluster_dict = {}
        current_start = 0
        
        # Iterate through sorted heads to determine clusters
        for i in range(1, num_heads):
            if sorted_combined[i-1][:2] != sorted_combined[i][:2]:  # Check if current and previous differ
                cluster_dict[len(cluster_dict)] = (current_start, i - 1)
                current_start = i
        cluster_dict[len(cluster_dict)] = (current_start, num_heads - 1)  # Last cluster
        
        clusters.append(cluster_dict)
    
    return permutations, clusters


def lut_to_permutation_single_layer(
    lut: torch.Tensor,
    num_heads: int = 32,
):
    '''
    find a permutation, such that after applying the permutation, same heads clustered together
    also return the clustered index
    '''

    assert lut.shape[0] == num_heads
    assert lut.dim() == 3

    # return a permutation, such that after applying the permutation, same heads clustered together
    serialized_heads = [tuple(head.reshape(-1).tolist()) for head in lut]

    sorted_indices = sorted(range(len(serialized_heads)), key=lambda i: (serialized_heads[i], i))
    
    # get the cluster, in the form of dictionary. like {1:(0,4), 2:(4, 32)}. (0, 4) is the range
    cluster = {}
    current_cluster = 0
    start = 0
    for i in range(1, len(sorted_indices)):
        if serialized_heads[sorted_indices[i]] != serialized_heads[sorted_indices[i-1]]:
            cluster[current_cluster] = (start, i)
            start = i
            current_cluster += 1

    cluster[current_cluster] = (start, len(sorted_indices))

    # check those cluster
    two_pattern_idx = []

    if len(cluster) == 2:
        two_pattern_idx.append(0)
    elif len(cluster) > 2:
        print(f"cluster = {cluster}")
        raise NotImplementedError
                
    # all lut have just one pattern, return arbitrary one
    if len(two_pattern_idx) == 0:
        return torch.tensor(sorted_indices), cluster


    # return any of the two pattern stuff    
    return torch.tensor(sorted_indices), cluster



def get_lut_global_size(lut: torch.Tensor, block_size: int = 64):
    # !!! assume global size is just one block!!
    return block_size
    assert lut.dim() == 2

    last_line_lut = lut[-1]
    
    global_block_num = 0
    for i in range(len(last_line_lut)):
        if last_line_lut[i] == i:
            global_block_num += 1
            continue
        else:
            break
    
    return global_block_num * block_size

def get_lut_band_size(lut: torch.Tensor, block_size: int = 64):
    assert lut.dim() == 2

    last_line_lut = lut[-1]
        
    last_line_lut_unique = set(last_line_lut.tolist())

    return block_size * len(last_line_lut_unique) - get_lut_global_size(lut, block_size)

def evaluate_model(model, test_loader, device, output_dir=None):
    """
    Evaluate model performance on test set
    
    Parameters:
        model: Trained UNet model
        test_loader: Test data loader
        device: Device (CPU or GPU)
        output_dir: Directory to save prediction results
    
    Returns:
        evaluation_metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    # Create output directory (if provided)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    dice_scores = []
    iou_scores = []
    miou_scores = []
    sensitivity_scores = []
    specificity_scores = []
    accuracy_scores = []
    mpa_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            names = batch['name']
            
            # Forward pass
            outputs = model(images)
            
            # Handle potential tuple output
            if isinstance(outputs, tuple):
                seg_output = outputs[0]  # Only use segmentation task output
            else:
                seg_output = outputs
            
            probs = torch.sigmoid(seg_output)
            
            # Calculate evaluation metrics
            dice = dice_coef(probs, masks).item()
            iou = iou_score(probs, masks).item()
            miou = mean_iou(probs, masks).item()
            sens = sensitivity(probs, masks).item()
            spec = specificity(probs, masks).item()
            acc = accuracy(probs, masks).item()
            mpa = mean_pixel_accuracy(probs, masks).item()
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            miou_scores.append(miou)
            sensitivity_scores.append(sens)
            specificity_scores.append(spec)
            accuracy_scores.append(acc)
            mpa_scores.append(mpa)
            
            # If output directory is provided, save prediction results
            if output_dir is not None:
                for i in range(images.size(0)):
                    # Get prediction mask
                    pred_mask = (probs[i].cpu().numpy().squeeze() > 0.5).astype(np.uint8) * 255
                    
                    # Save prediction mask
                    pred_mask_img = Image.fromarray(pred_mask)
                    pred_mask_img.save(os.path.join(output_dir, f"{names[i]}_pred.png"))
                    
                    # Visualize results (original image, true label, prediction label)
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize image
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    
                    true_mask = masks[i].cpu().numpy().squeeze()
                    
                    # Create visualization
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.title('Original Image')
                    plt.imshow(img)
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.title('True Label')
                    plt.imshow(true_mask, cmap='gray')
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.title('Predicted Label')
                    plt.imshow(pred_mask, cmap='gray')
                    plt.axis('off')
                    
                    plt.savefig(os.path.join(output_dir, f"{names[i]}_visualization.png"), bbox_inches='tight')
                    plt.close()
    
    # Calculate average metrics
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_miou = np.mean(miou_scores)
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_mpa = np.mean(mpa_scores)
    
    # Prepare evaluation results
    evaluation_metrics = {
        'dice': avg_dice,
        'iou': avg_iou,
        'miou': avg_miou,
        'sensitivity': avg_sensitivity,
        'specificity': avg_specificity,
        'accuracy': avg_accuracy,
        'mpa': avg_mpa
    }
    
    # Print evaluation results
    print_metrics(evaluation_metrics)
    
    return evaluation_metrics

def predict_image(model, image_path, device, output_dir=None):
    """
    Predict segmentation for a single image
    
    Parameters:
        model: Trained UNet model
        image_path: Path to input image
        device: Device (CPU or GPU)
        output_dir: Directory to save prediction results
    
    Returns:
        pred_mask: Predicted mask
    """
    from torchvision import transforms
    
    model.eval()
    
    # Create output directory (if provided)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        
        # Handle potential tuple output
        if isinstance(output, tuple):
            seg_output = output[0]  # Only use segmentation task output
        else:
            seg_output = output
            
        prob = torch.sigmoid(seg_output)
    
    # Get prediction mask
    pred_mask = (prob.cpu().numpy().squeeze() > 0.5).astype(np.uint8) * 255
    
    # If output directory is provided, save prediction results
    if output_dir is not None:
        # Get original filename
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save prediction mask
        pred_mask_img = Image.fromarray(pred_mask)
        pred_mask_img.save(os.path.join(output_dir, f"{image_name}_pred.png"))
        
        # Visualize results (original image and prediction label)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Predicted Label')
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        
        plt.savefig(os.path.join(output_dir, f"{image_name}_visualization.png"), bbox_inches='tight')
        plt.close()
    
    return pred_mask

def main():
    parser = argparse.ArgumentParser(description='Evaluate UNet model and predict retinal vessel segmentation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save prediction results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_path', type=str, help='Path to single image for prediction mode')
    parser.add_argument('--use_attention', action='store_true', help='Use channel attention mechanism')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = UNetWithAttention(n_channels=3, n_classes=1, enable_attention=args.use_attention)
    
    # Load trained model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different model keys in older checkpoints
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try direct loading for older models
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    print(f"Loaded model: {args.model_path}")
    if 'epoch' in checkpoint:
        print(f"Training Epoch: {checkpoint['epoch']}")
    if 'train_dice' in checkpoint:
        print(f"Training Dice: {checkpoint['train_dice']:.2f}%")
    if 'val_dice' in checkpoint:
        print(f"Validation Dice: {checkpoint['val_dice']:.2f}%")
    
    # Enable attention if specified
    if args.use_attention:
        print("Using channel attention mechanism")
    
    # Single image prediction mode
    if args.image_path:
        print(f"Predicting image: {args.image_path}")
        pred_mask = predict_image(model, args.image_path, device, args.output_dir)
    # Test set evaluation mode
    else:
        print("Evaluating model on test set...")
        test_loader = get_test_loader(args.data_dir, args.batch_size)
        evaluation_metrics = evaluate_model(model, test_loader, device, args.output_dir)

if __name__ == '__main__':
    main() 