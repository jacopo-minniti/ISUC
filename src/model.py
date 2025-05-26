import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ActivationsDataset(Dataset):
    def __init__(self, path: str, activations_layer: int) -> None:
        self.data = pd.read_csv(path)
        self.activations_layer = activations_layer
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Parse the activations string as a list of floats
        activations = json.loads(row[f'activations_{self.activations_layer}'])
        activations = torch.tensor(activations, dtype=torch.float32)
        
        # Get the hallucination label
        label = torch.tensor(row['hallucinated'], dtype=torch.float32)
        
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