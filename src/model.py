import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ActivationsDataset(Dataset):
    """
    A custom PyTorch Dataset for loading activations and labels from a JSON file.
    """
    def __init__(self, path: str) -> None:
        """
        Initializes the dataset by loading data from a specified JSON file.

        Args:
            path (str): The file path to the JSON dataset.
            activations_layer (int): The layer index from which activations were collected.
                                     (Note: This parameter is currently not directly used
                                     for data parsing but kept for consistency with original signature).
        """

        try:
            with open(path, 'r') as f:
                self.data = json.load(f)
            print(f"Successfully loaded dataset from {path}. Found {len(self.data)} entries.")
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {path}")
            self.data = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {path}. Please ensure it's a valid JSON file.")
            self.data = []
        except Exception as e:
            print(f"An unexpected error occurred while loading the dataset: {e}")
            self.data = []
        
    def __len__(self) -> int:
        """
        Returns the total number of items in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """
        Retrieves an item (activations and its corresponding label) by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - activations (torch.Tensor): The activations as a float32 tensor.
                - label (torch.Tensor): The label as a float32 tensor.
        
        Raises:
            IndexError: If the index is out of bounds.
            KeyError: If 'activations' or 'label' keys are missing in the data entry.
            TypeError: If data types are not as expected during conversion.
        """
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.data)}")

        row = self.data[idx]
        
        try:
            # The 'activations' field in your JSON is already a list of floats.
            # No need for json.loads() here, as it's already parsed when loading the file.
            activations = torch.tensor(row['activations'], dtype=torch.float32)
            
            # Get the hallucination label
            label = torch.tensor(int(row['label']), dtype=torch.float32)
        except KeyError as e:
            print(f"Error: Missing key in data entry at index {idx}: {e}. Entry: {row}")
            # Depending on your error handling strategy, you might want to return None,
            # raise the error, or skip this item. For now, re-raising to indicate a problem.
            raise
        except (TypeError, ValueError) as e:
            print(f"Error: Type or value conversion issue at index {idx}: {e}. Entry: {row}")
            raise

        return activations, label


class ISUC(nn.Module):
    def __init__(self, input_size: int):
        super(ISUC, self).__init__()
        
        ### CURRENT ARCHITECTURE INSPIRED FROM ### 
        ### Azaria & Mitchell, 2023 ### 
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
        
    def forward(self, x):
        return self.model(x)