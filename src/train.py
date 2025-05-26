import os
import json

import pandas as pd
import numpy as np
import wandb

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from evaluate import evaluate
from utils import load_model
from model import ActivationsDataset


def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    DATASET_PATH = f"./data/{args.dataset}/prepared"
    train_dataset = ActivationsDataset(os.path.join(DATASET_PATH, f"train.json"))
    test_dataset = ActivationsDataset(os.path.join(DATASET_PATH, f"test.json"))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Sample the first batch to determine input size
    sample_activations, _ = next(iter(train_loader))
    input_size = sample_activations.shape[1]
    
    # Initialize model or load from last checkpoint
    model = load_model(input_size, args.current_checkpoint_path)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    # Training loop
    best_test_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for activations, labels in progress_bar:
            activations, labels = activations.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(activations)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, device)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_accuracy": test_metrics['accuracy'],
            "test_precision": test_metrics['precision'],
            "test_recall": test_metrics['recall'],
            "test_f1": test_metrics['f1'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, "
              f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
              f"Test F1: {test_metrics['f1']:.4f}")
        
        # Update learning rate if using scheduler
        if args.use_lr_scheduler:
            scheduler.step(train_loss)
        
        # Save best model
        if test_metrics['f1'] > best_test_f1:
            best_test_f1 = test_metrics['f1']
            torch.save(model.state_dict(), args.new_checkpoint_path)
            wandb.save(args.new_checkpoint_path)
            print(f"New best model saved with F1: {best_test_f1:.4f}")

    
    # Final evaluation
    try:
        model.load_state_dict(torch.load(args.new_checkpoint_path))
    except Exception as e:
        print("Model could not be loaded")
    final_metrics = evaluate(model, test_loader, device)
    
    print(f"Final Test Metrics - Accuracy: {final_metrics['accuracy']:.4f}, "
          f"Precision: {final_metrics['precision']:.4f}, "
          f"Recall: {final_metrics['recall']:.4f}, "
          f"F1: {final_metrics['f1']:.4f}")
    
    return model, final_metrics