"""
Base classes and interfaces for uncertainty classifiers using LLM internal states.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import torch
from torch.utils.data import Dataset


class BaseDatasetProcessor(ABC):
    """Abstract base class for dataset processors."""
    
    @abstractmethod
    def process_dataset(self, model_id: str, layer_idx: int, output_dir: str) -> bool:
        """Process the dataset and save prepared data."""
        pass


class BaseUncertaintyClassifier(torch.nn.Module):
    """Abstract base class for uncertainty classifiers."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier."""
        pass


class BaseActivationExtractor(ABC):
    """Abstract base class for activation extraction from LLMs."""
    
    @abstractmethod
    def extract_activations(self, text: str, layer_idx: int) -> torch.Tensor:
        """Extract activations from specified layer."""
        pass


class ActivationsDataset(Dataset):
    """Dataset for loading activations and labels from JSON files."""
    
    def __init__(self, path: str) -> None:
        import json
        try:
            with open(path, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded dataset from {path}. Found {len(self.data)} entries.")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {path}")
            self.data = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {path}")
            self.data = []
        except Exception as e:
            print(f"Unexpected error loading dataset: {e}")
            self.data = []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.data)}")
        
        row = self.data[idx]
        try:
            activations = torch.tensor(row['activations'], dtype=torch.float32)
            label = torch.tensor(int(row['label']), dtype=torch.float32)
            return activations, label
        except KeyError as e:
            print(f"Missing key in data entry at index {idx}: {e}")
            raise
        except (TypeError, ValueError) as e:
            print(f"Type conversion error at index {idx}: {e}")
            raise