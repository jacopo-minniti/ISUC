"""
Training utilities for uncertainty classifiers.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, Any

from src.base import ActivationsDataset
from src.model import load_classifier
from src.evaluate import evaluate


class Trainer:
    """General trainer for uncertainty classifiers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_data(self) -> tuple:
        """Setup train and test dataloaders."""
        dataset_path = self.config["dataset_path"]
        
        train_dataset = ActivationsDataset(os.path.join(dataset_path, "train.json"))
        test_dataset = ActivationsDataset(os.path.join(dataset_path, "test.json"))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 4)
        )
        
        return train_loader, test_loader
    
    def setup_model(self, input_size: int):
        """Setup model, optimizer, and scheduler."""
        # Load or create model
        self.model = load_classifier(
            self.config["classifier_type"],
            input_size,
            self.config["current_checkpoint_path"],
            **self.config.get("model_kwargs", {})
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 1e-5)
        )
        
        # Setup scheduler if requested
        self.scheduler = None
        if self.config.get("use_lr_scheduler", False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        
        # Setup loss function
        self.criterion = nn.BCELoss()
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        for activations, labels in progress_bar:
            activations, labels = activations.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(activations)
            loss = self.criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get("clip_grad"):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(train_loader)
    
    def train(self) -> tuple:
        """Full training loop."""
        # Setup data
        train_loader, test_loader = self.setup_data()
        
        # Get input size from first batch
        sample_activations, _ = next(iter(train_loader))
        input_size = sample_activations.shape[1]
        
        # Setup model
        self.setup_model(input_size)
        
        best_f1 = 0.0
        
        for epoch in range(self.config["epochs"]):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            test_metrics = evaluate(self.model, test_loader, self.device)
            
            # Log to wandb if enabled
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_accuracy": test_metrics['accuracy'],
                    "test_precision": test_metrics['precision'],
                    "test_recall": test_metrics['recall'],
                    "test_f1": test_metrics['f1'],
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
                  f"Test F1: {test_metrics['f1']:.4f}")
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(train_loss)
            
            # Save best model
            if test_metrics['f1'] > best_f1:
                best_f1 = test_metrics['f1']
                torch.save(self.model.state_dict(), self.config["new_checkpoint_path"])
                if wandb.run is not None:
                    wandb.save(self.config["new_checkpoint_path"])
                print(f"New best model saved with F1: {best_f1:.4f}")
        
        # Final evaluation with best model
        try:
            self.model.load_state_dict(torch.load(self.config["new_checkpoint_path"]))
        except Exception as e:
            print(f"Could not load best model: {e}")
        
        final_metrics = evaluate(self.model, test_loader, self.device)
        
        print(f"Final Test Metrics - Accuracy: {final_metrics['accuracy']:.4f}, "
              f"Precision: {final_metrics['precision']:.4f}, "
              f"Recall: {final_metrics['recall']:.4f}, "
              f"F1: {final_metrics['f1']:.4f}")
        
        return self.model, final_metrics


def train_classifier(config: Dict[str, Any]) -> tuple:
    """Train a classifier with the given configuration."""
    trainer = Trainer(config)
    return trainer.train()