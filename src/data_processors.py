"""
Dataset processors for different uncertainty classification datasets.
"""
import os
import json
import pandas as pd
from typing import Dict, Any
from src.base import BaseDatasetProcessor
from src.activations import HuggingFaceActivationExtractor


class TrueFalseDatasetProcessor(BaseDatasetProcessor):
    """Processor for true/false statement datasets."""
    
    def __init__(self, data_path: str, test_file: str = "inventions_true_false.csv"):
        self.data_path = data_path
        self.test_file = test_file
    
    def process_dataset(self, model_id: str, layer_idx: int, output_dir: str) -> bool:
        """Process true/false dataset and extract activations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize activation extractor
        extractor = HuggingFaceActivationExtractor(model_id)
        
        original_data_path = os.path.join(self.data_path, "original")
        train_data = []
        test_data = []
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(original_data_path) if f.endswith(".csv")]
        if not csv_files:
            print(f"No CSV files found in {original_data_path}")
            return False
        
        # Process each file
        for filename in csv_files:
            file_path = os.path.join(original_data_path, filename)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            print(f"Processing {filename}...")
            for _, row in df.iterrows():
                statement = str(row["statement"])
                label = int(row["label"])
                
                # Extract activations
                activations = extractor.extract_activations(statement, layer_idx)
                if activations is None:
                    print(f"Failed to extract activations for: {statement[:50]}...")
                    continue
                
                data_entry = {
                    "statement": statement,
                    "label": label,
                    "model": model_id,
                    "activations": activations.tolist(),
                    "source_file": filename
                }
                
                # Split into train/test based on filename
                if filename == self.test_file:
                    test_data.append(data_entry)
                else:
                    train_data.append(data_entry)
        
        # Save processed data
        train_path = os.path.join(output_dir, "train.json")
        test_path = os.path.join(output_dir, "test.json")
        
        try:
            with open(train_path, "w") as f:
                json.dump(train_data, f, indent=4)
            print(f"Training data saved to {train_path}")
            
            with open(test_path, "w") as f:
                json.dump(test_data, f, indent=4)
            print(f"Test data saved to {test_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False


def get_dataset_processor(dataset_name: str, **kwargs) -> BaseDatasetProcessor:
    """Factory function to get dataset processors."""
    if dataset_name == "true-false":
        data_path = kwargs.get("data_path", "./data/true-false")
        test_file = kwargs.get("test_file", "inventions_true_false.csv")
        return TrueFalseDatasetProcessor(data_path, test_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def process_dataset(dataset_name: str, model_id: str, layer_idx: int, **kwargs) -> bool:
    """Process a dataset and save prepared data."""
    processor = get_dataset_processor(dataset_name, **kwargs)
    output_dir = kwargs.get("output_dir", f"./data/{dataset_name}/prepared")
    return processor.process_dataset(model_id, layer_idx, output_dir)