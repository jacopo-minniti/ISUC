"""
Sample test set generator for evaluating trained models.
"""
import sys
import os
import torch
import json
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.base import ActivationsDataset
from src.model import load_classifier
from src.evaluate import evaluate
from torch.utils.data import DataLoader


def create_sample_test_set(dataset_path: str, sample_size: int = 50):
    """
    Create a smaller sample from the test set for quick evaluation.
    
    Args:
        dataset_path: Path to the test dataset JSON file
        sample_size: Number of samples to include
    
    Returns:
        Path to the sample test set file
    """
    # Load full dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Sample data
    if len(data) > sample_size:
        sampled_data = random.sample(data, sample_size)
    else:
        sampled_data = data
    
    # Save sample
    sample_path = dataset_path.replace('.json', f'_sample_{sample_size}.json')
    with open(sample_path, 'w') as f:
        json.dump(sampled_data, f, indent=2)
    
    print(f"Created sample test set with {len(sampled_data)} samples: {sample_path}")
    return sample_path


def evaluate_on_sample(model_path: str, model_type: str, test_data_path: str, 
                      sample_size: int = 50, batch_size: int = 32):
    """
    Evaluate a trained model on a sample of the test set.
    
    Args:
        model_path: Path to the trained model checkpoint
        model_type: Type of classifier ("isuc" or "simple_mlp")
        test_data_path: Path to the test dataset
        sample_size: Size of the sample to evaluate on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create sample test set
    sample_path = create_sample_test_set(test_data_path, sample_size)
    
    try:
        # Load sample dataset
        dataset = ActivationsDataset(sample_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Get input size from first batch
        sample_activations, _ = next(iter(dataloader))
        input_size = sample_activations.shape[1]
        
        # Load model
        model = load_classifier(model_type, input_size, model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Evaluate
        metrics = evaluate(model, dataloader, device)
        
        print(f"\nEvaluation Results on {len(dataset)} samples:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        
        return metrics
        
    finally:
        # Clean up sample file
        if os.path.exists(sample_path):
            os.unlink(sample_path)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python sample_test_set.py <model_path> <model_type> <test_data_path> [sample_size]")
        print("Example: python sample_test_set.py ./models/isuc-v05-26.pt isuc ./data/true-false/prepared/test.json 100")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    test_data_path = sys.argv[3]
    sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    evaluate_on_sample(model_path, model_type, test_data_path, sample_size)