"""
Model architectures for uncertainty classification.
"""
import torch
import torch.nn as nn
from src.base import BaseUncertaintyClassifier


class SAPLMA(BaseUncertaintyClassifier):
    """
    Internal State Uncertainty Classifier - based on Azaria & Mitchell, 2023.
    Arxiv Paper: https://arxiv.org/abs/2304.13734
    """
    
    def __init__(self, input_size: int):
        super().__init__(input_size)
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_classifier(classifier_type: str, input_size: int, **kwargs) -> BaseUncertaintyClassifier:
    """Factory function to create classifiers."""
    if classifier_type == "SAPLMA":
        return SAPLMA(input_size)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def load_classifier(classifier_type: str, input_size: int, checkpoint_path: str, **kwargs) -> BaseUncertaintyClassifier:
    """Load a classifier from checkpoint."""
    import os
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_classifier(classifier_type, input_size, **kwargs).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint)
            print("Model loaded successfully from checkpoint.")
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            print("Using fresh model.")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Using fresh model.")
    
    return model