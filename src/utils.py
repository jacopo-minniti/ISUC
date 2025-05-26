import torch
import os

from src.model import ISUC


def load_model(input_size: int, checkpoint_path: str) -> ISUC:
    # Initialize your model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISUC(input_size).to(device)

    # Check if the checkpoint file exists and load the model if it does
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            # Load the state_dict from the checkpoint
            # Use map_location to ensure it loads correctly regardless of original device
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load the state_dict into the model
            model.load_state_dict(checkpoint)

            print("Model loaded successfully from checkpoint.")
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            print("Starting training with a fresh model.")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training with a fresh model.")
    
    return model 