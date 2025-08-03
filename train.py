import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from model import UNetWithAttention
from dataset import get_data_loaders
from utils import dice_coef, dice_loss, MultiTaskLoss, print_metrics

def train_model(model, data_dir, batch_size, epochs, learning_rate, device, save_dir, use_multitask=True):
    """
    Train UNet model with attention and multi-task learning
    
    Parameters:
        model: UNet model with attention
        data_dir: Data directory
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device (CPU or GPU)
        save_dir: Directory to save models and logs
        use_multitask: Whether to use multi-task learning
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create TensorBoard summary writer
    log_dir = os.path.join(save_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    
    # Define loss function and optimizer
    if use_multitask:
        criterion = MultiTaskLoss(task_weights=[1.0, 0.5, 0.5])
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            if use_multitask and isinstance(outputs, tuple):
                # Multi-task learning loss
                loss, seg_loss, _, _ = criterion(outputs, masks)
                # For Dice calculation we use only segmentation output
                seg_output = outputs[0]
            else:
                # Standard segmentation loss or single-task output
                seg_output = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(seg_output, masks)
                seg_loss = loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print shapes for debugging (only in the first batch of the first epoch)
            if epoch == 0 and train_loss == 0:
                print(f"Debug - Training shapes: seg_output: {seg_output.shape}, masks: {masks.shape}")
                if isinstance(outputs, tuple):
                    print(f"Debug - Training outputs: tuple of {len(outputs)} elements")
                    for i, out in enumerate(outputs):
                        print(f"  Output {i}: shape {out.shape}")
            
            # Ensure shape matching before calculating Dice coefficient
            seg_pred = torch.sigmoid(seg_output)
            
            # If shapes don't match, adjust prediction size to match target
            if seg_pred.shape != masks.shape:
                # Adjust prediction size to match target
                seg_pred = torch.nn.functional.interpolate(
                    seg_pred, 
                    size=masks.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Calculate Dice coefficient
            train_loss += loss.item() * images.size(0)
            train_dice += dice_coef(seg_pred, masks).item() * images.size(0)
        
        # Calculate average loss and Dice coefficient
        train_loss /= len(train_loader.dataset)
        train_dice /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                if use_multitask and isinstance(outputs, tuple):
                    # Multi-task learning loss
                    loss, seg_loss, _, _ = criterion(outputs, masks)
                    # For Dice calculation we use only segmentation output
                    seg_output = outputs[0]
                else:
                    # Standard segmentation loss or single-task output
                    seg_output = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = criterion(seg_output, masks)
                    seg_loss = loss
                
                # Ensure input and target shapes match
                # Print shapes for debugging
                if epoch == 0 and val_dice == 0:  # Only print in the first batch of the first epoch
                    print(f"Debug - Validation shapes: seg_output: {seg_output.shape}, masks: {masks.shape}")
                    if isinstance(outputs, tuple):
                        print(f"Debug - Validation outputs: tuple of {len(outputs)} elements")
                        for i, out in enumerate(outputs):
                            print(f"  Output {i}: shape {out.shape}")
                
                # Check if batch dimension is missing and add it if necessary
                if len(seg_output.shape) < len(masks.shape):
                    # If dimensions don't match, output format might have changed in model.eval() mode
                    print(f"Warning: Output dimension count ({len(seg_output.shape)}) is less than target dimension count ({len(masks.shape)}), adding batch dimension")
                    # Add batch dimension
                    seg_output = seg_output.unsqueeze(0)
                    print(f"Adjusted output shape: {seg_output.shape}")
                
                seg_pred = torch.sigmoid(seg_output)
                
                # If shapes don't match, adjust prediction size to match target
                if seg_pred.shape != masks.shape:
                    # Ensure dimensions match before interpolation
                    if len(seg_pred.shape) != len(masks.shape):
                        # If dimension counts don't match, interpolation is not possible
                        print(f"Error: Cannot adjust shape, output shape {seg_pred.shape} doesn't match target shape {masks.shape} in dimension count")
                        # Emergency measure: create a zero tensor with the same shape as masks to avoid crash
                        seg_pred = torch.zeros_like(masks)
                    else:
                        # Normal interpolation
                        seg_pred = torch.nn.functional.interpolate(
                            seg_pred, 
                            size=masks.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                
                # Calculate Dice coefficient
                val_loss += loss.item() * images.size(0)
                val_dice += dice_coef(seg_pred, masks).item() * images.size(0)
        
        # Calculate average loss and Dice coefficient
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print results
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.2f}%')
        
        # Record to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved best model with val_loss: {val_loss:.4f}')
        
        # Save last epoch model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
        }, os.path.join(save_dir, 'last_model.pth'))
    
    writer.close()
    return model

def main():
    parser = argparse.ArgumentParser(description='Train UNet model for retinal vessel segmentation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save models and logs')
    parser.add_argument('--use_attention', action='store_true', help='Use channel attention mechanism')
    parser.add_argument('--multitask', action='store_true', help='Use multi-task learning')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = UNetWithAttention(n_channels=3, n_classes=1, enable_attention=args.use_attention)
    model = model.to(device)
    
    # Print model configuration
    print(f"Model configuration:")
    print(f"- Channel Attention: {'Enabled' if args.use_attention else 'Disabled'}")
    print(f"- Multi-task Learning: {'Enabled' if args.multitask else 'Disabled'}")
    
    # Train model
    train_model(
        model=model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        use_multitask=args.multitask
    )

if __name__ == '__main__':
    main() 