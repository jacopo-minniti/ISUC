"""
Utility functions for uncertainty classification experiments.
"""
import os
import torch
from datetime import datetime
from typing import Dict, Any


def get_model_version(models_dir: str = "./models", prefix: str = "isuc-v") -> str:
    """Get the latest model version based on modification time."""
    import glob
    
    pattern = os.path.join(models_dir, f"{prefix}*.pt")
    model_files = glob.glob(pattern)
    
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        return latest_model.split(prefix)[-1].split(".pt")[0]
    else:
        return datetime.now().strftime("%m-%d")


def create_checkpoint_paths(version: str = None, models_dir: str = "./models", prefix: str = "isuc-v") -> Dict[str, str]:
    """Create checkpoint paths for current and new model versions."""
    if version is None:
        current_version = get_model_version(models_dir, prefix)
        new_version = datetime.now().strftime("%m-%d")
    else:
        current_version = version
        new_version = version
    
    return {
        "current": os.path.join(models_dir, f"{prefix}{current_version}.pt"),
        "new": os.path.join(models_dir, f"{prefix}{new_version}.pt")
    }


def setup_directories(dataset_name: str, base_data_dir: str = "./data") -> Dict[str, str]:
    """Setup and ensure data directories exist."""
    dataset_dir = os.path.join(base_data_dir, dataset_name)
    prepared_dir = os.path.join(dataset_dir, "prepared")
    
    os.makedirs(prepared_dir, exist_ok=True)
    
    return {
        "dataset": dataset_dir,
        "prepared": prepared_dir,
        "original": os.path.join(dataset_dir, "original")
    }


def build_training_config(args) -> Dict[str, Any]:
    """Build training configuration from command line arguments."""
    checkpoint_paths = create_checkpoint_paths()
    
    return {
        "model_id": args.model,
        "dataset_name": args.dataset,
        "dataset_path": f"./data/{args.dataset}/prepared",
        "classifier_type": getattr(args, "classifier_type", "isuc"),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": getattr(args, "weight_decay", 1e-5),
        "num_workers": getattr(args, "num_workers", 4),
        "clip_grad": getattr(args, "clip_grad", 1.0),
        "use_lr_scheduler": getattr(args, "use_lr_scheduler", False),
        "layer_idx": args.layer_idx,
        "current_checkpoint_path": getattr(args, "current_checkpoint_path", checkpoint_paths["current"]),
        "new_checkpoint_path": getattr(args, "new_checkpoint_path", checkpoint_paths["new"]),
        "model_kwargs": getattr(args, "model_kwargs", {})
    }


def print_experiment_info(config: Dict[str, Any]):
    """Print experiment configuration."""
    print("=" * 50)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 50)
    for key, value in config.items():
        if key != "model_kwargs":  # Skip complex nested dict
            print(f"{key}: {value}")
    print("=" * 50)