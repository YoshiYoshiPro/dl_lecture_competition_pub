import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider, efficient_train_collate
from src.utils import compute_epe_error, compute_multiscale_loss, total_loss
from enum import Enum, auto
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional
import os
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from concurrent.futures import ThreadPoolExecutor

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataloader
    try:
        loader = DatasetProvider(
            dataset_path=Path(args.dataset_path),
            representation_type=RepresentationType.VOXEL,
            delta_t_ms=100,
            num_bins=4,
            val_split=args.val_split
        )
        train_set = loader.get_train_dataset()
        val_set = loader.get_val_dataset()
        test_set = loader.get_test_dataset()
    except Exception as e:
        logger.error(f"Error in data loading: {str(e)}")
        return

    train_data = DataLoader(train_set, batch_size=args.data_loader.train.batch_size,
                            shuffle=args.data_loader.train.shuffle, collate_fn=efficient_train_collate)
    val_data = DataLoader(val_set, batch_size=args.data_loader.train.batch_size,
                          shuffle=False, collate_fn=efficient_train_collate) if val_set else None
    test_data = DataLoader(test_set, batch_size=args.data_loader.test.batch_size,
                           shuffle=False, collate_fn=efficient_train_collate)

    # Model
    model = EVFlowNet(args.train).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.train.lr_factor, patience=args.train.lr_patience, verbose=True)

    # Training
    epoch_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = args.train.patience
    no_improve_epochs = 0
    best_model_path = None

    for epoch in range(args.train.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}/{args.train.epochs}"):
            event_images = [img.to(device) for img in batch['event_volume_multi']]
            pred_flows = model(event_images)
            ground_truth_flow = batch["flow_gt"].to(device)
            ground_truth_valid_mask = batch["flow_gt_valid_mask"].to(device)
            
            optimizer.zero_grad()
            pred_flows = model(event_images)
            loss = total_loss(pred_flows, ground_truth_flow, ground_truth_valid_mask, smooth_weight=args.train.smooth_weight)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_data)
        epoch_losses.append(train_loss)
        
        # Validation
        if val_data:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_data, desc="Validation"):
                    event_images = [img.to(device) for img in batch['event_volume_multi']]
                    ground_truth_flow = batch["flow_gt"].to(device)
                    ground_truth_valid_mask = batch["flow_gt_valid_mask"].to(device)
                    
                    pred_flows = model(event_images)
                    loss = total_loss(pred_flows, ground_truth_flow, ground_truth_valid_mask, smooth_weight=args.train.smooth_weight)
                    val_loss += loss.item()
            
            val_loss /= len(val_data)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')              
                best_model_path = f"checkpoints/best_model_{time.strftime('%Y%m%d%H%M%S')}.pth"
                torch.save(model.state_dict(), best_model_path)

            else:
                no_improve_epochs += 1
            
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        else:
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    # Testing
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss = 0
    flow_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing"):
            event_images = [img.to(device) for img in batch['event_volume_multi']]
            pred_flows = model(event_images)
            flow_predictions.append(pred_flows[-1].cpu())  # 最終スケールの予測を使用

            if 'flow_gt' in batch:
                ground_truth_flow = batch["flow_gt"].to(device)
                ground_truth_valid_mask = batch["flow_gt_valid_mask"].to(device)
                loss = compute_epe_error(pred_flows[-1], ground_truth_flow, ground_truth_valid_mask)
                test_loss += loss.item()

    if 'flow_gt' in batch:
        test_loss /= len(test_data)
        logger.info(f"Test Loss: {test_loss:.4f}")

    # Save predictions
    flow_predictions = torch.cat(flow_predictions, dim=0)
    save_optical_flow_to_npy(flow_predictions, "submission")

    # Plot and save training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Train Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_loss.png')
    plt.close()

    # Save loss values to text file
    with open('training_validation_loss.txt', 'w') as f:
        for epoch, train_loss in enumerate(epoch_losses, 1):
            line = f'Epoch {epoch}: Train Loss: {train_loss}'
            if val_losses:
                line += f', Val Loss: {val_losses[epoch-1]}'
            f.write(line + '\n')

if __name__ == "__main__":
    main()