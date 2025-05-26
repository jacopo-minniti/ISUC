import os
import json
import random
import argparse
import sys

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to sys.path to allow importing from src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import ActivationsDataset, ISUC

DEFAULT_MODEL_PATH = "./models/isuc-v05-26.pt"
DEFAULT_DATASET_NAME = "true-false"
DEFAULT_DATA_BASE_PATH = "../data"
DEFAULT_NUM_SAMPLES = 15

def run_sampling(args):
    """
    Loads data, samples, predicts, and evaluates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Construct path to test.json
    test_json_path = os.path.join(args.data_base_path, args.dataset_name, "prepared", "test.json")
    print(f"Loading test data from: {test_json_path}")

    if not os.path.exists(test_json_path):
        print(f"Error: Test data file not found at {test_json_path}")
        return

    # Load the test dataset
    test_dataset = ActivationsDataset(path=test_json_path)

    if not test_dataset or len(test_dataset) == 0:
        print("Error: Test dataset is empty or could not be loaded.")
        return

    print(f"Successfully loaded {len(test_dataset)} items from test dataset.")

    # Determine number of samples
    num_to_sample = min(args.num_samples, len(test_dataset))
    if num_to_sample <= 0:
        print("Number of samples must be positive.")
        return
    
    print(f"Sampling {num_to_sample} items randomly.")
    
    # Randomly select indices
    # Ensure population is not empty before sampling
    if len(test_dataset) == 0:
        print("Cannot sample from an empty dataset.")
        return
    
    try:
        sampled_indices = random.sample(range(len(test_dataset)), num_to_sample)
    except ValueError as e:
        print(f"Error during sampling: {e}. This might happen if num_to_sample > len(dataset).")
        print(f"Attempting to sample all available items: {len(test_dataset)}")
        sampled_indices = list(range(len(test_dataset)))
        num_to_sample = len(test_dataset)
        if num_to_sample == 0:
            print("No items to sample.")
            return


    # Get input_size from the first sample's activations
    # ActivationsDataset.__getitem__ returns (activations_tensor, label_tensor)
    if not test_dataset.data: # Check if self.data was populated
        print("Error: Dataset has no data entries after loading.")
        return
        
    try:
        # Try to get activations from the first item in the raw loaded data
        first_item_activations = test_dataset.data[0].get('activations')
        if first_item_activations is None or not isinstance(first_item_activations, list):
            raise ValueError("Activations not found or not a list in the first data item.")
        input_size = len(first_item_activations)
    except (IndexError, TypeError, ValueError) as e:
        print(f"Error determining input size from dataset: {e}")
        print("Attempting to get input size from the first processed item if available...")
        if len(test_dataset) > 0:
            try:
                sample_activations, _ = test_dataset[0]
                input_size = sample_activations.shape[0]
            except Exception as e_inner:
                print(f"Could not determine input size from processed item: {e_inner}")
                return
        else:
            print("Dataset is empty, cannot determine input size.")
            return
            
    print(f"Determined input size for the model: {input_size}")

    # Initialize model
    model = ISUC(input_size=input_size)
    
    # Load model checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading model checkpoint from: {args.checkpoint_path}")
        try:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        except Exception as e:
            print(f"Error loading model checkpoint: {e}. Predictions will be from an uninitialized model.")
    else:
        print(f"Warning: Model checkpoint not found at {args.checkpoint_path}. Predictions will be from an uninitialized model.")

    model.to(device)
    model.eval()

    true_labels_list = []
    predicted_labels_list = []

    print("\n--- Sampled Results ---")
    for i, idx in enumerate(sampled_indices):
        # ActivationsDataset.data contains the raw dicts, useful for 'statement'
        # ActivationsDataset[idx] gives (activations_tensor, label_tensor)
        
        raw_item = test_dataset.data[idx]
        statement = raw_item.get("statement", "N/A")
        true_label_from_raw = int(raw_item.get("label", -1)) # Get label from raw data for printing

        activations_tensor, label_tensor = test_dataset[idx]
        
        activations_tensor = activations_tensor.to(device)

        with torch.no_grad():
            # Model expects batch dimension, add it if activations_tensor is 1D
            if activations_tensor.ndim == 1:
                activations_tensor = activations_tensor.unsqueeze(0)
            
            output = model(activations_tensor)
            predicted_prob = output.item()
            predicted_label = 1 if predicted_prob > 0.5 else 0

        print(f"\nSample {i+1}/{num_to_sample} (Original Index: {idx}):")
        print(f"  Statement: {statement[:100]}..." if len(statement) > 100 else f"  Statement: {statement}")
        print(f"  True Label: {true_label_from_raw}")
        print(f"  Predicted Probability: {predicted_prob:.4f}")
        print(f"  Predicted Label: {predicted_label}")

        true_labels_list.append(true_label_from_raw)
        predicted_labels_list.append(predicted_label)

    # Calculate and print metrics
    if not true_labels_list or not predicted_labels_list:
        print("\nNo samples processed to calculate metrics.")
        return

    print("\n--- Overall Metrics for Sampled Data ---")
    accuracy = accuracy_score(true_labels_list, predicted_labels_list)
    # For binary classification, average='binary' is appropriate.
    # zero_division handles cases where precision/recall might be undefined (e.g., no positive predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_list, 
        predicted_labels_list, 
        average='binary', 
        zero_division=0
    )

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from test data and evaluate a model.")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default=DEFAULT_DATASET_NAME,
        help=f"Name of the dataset (default: {DEFAULT_DATASET_NAME})"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the model checkpoint (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of random samples to draw (default: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--data_base_path", 
        type=str, 
        default=DEFAULT_DATA_BASE_PATH,
        help=f"Base path for the data directory (default: {DEFAULT_DATA_BASE_PATH})"
    )
    
    args = parser.parse_args()
    run_sampling(args)