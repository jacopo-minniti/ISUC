"""
Simple classification utility for testing trained models.
"""
import sys
import os
import torch
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import load_classifier
from src.activations import HuggingFaceActivationExtractor


def classify_query(model_path: str, model_type: str, input_size: int, 
                  model_id: str, layer_idx: int, query: str) -> dict:
    """
    Classify a single query using a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        model_type: Type of classifier ("isuc" or "simple_mlp")
        input_size: Input size for the model
        model_id: HuggingFace model ID for activation extraction
        layer_idx: Layer index for activation extraction
        query: Text query to classify
    
    Returns:
        Dictionary with classification results
    """
    # Load the trained classifier
    classifier = load_classifier(model_type, input_size, model_path)
    classifier.eval()
    
    # Extract activations
    extractor = HuggingFaceActivationExtractor(model_id)
    activations = extractor.extract_activations(query, layer_idx)
    
    if activations is None:
        return {"error": "Failed to extract activations"}
    
    # Make prediction
    with torch.no_grad():
        activations_tensor = activations.unsqueeze(0)  # Add batch dimension
        output = classifier(activations_tensor)
        probability = output.item()
        prediction = int(probability > 0.5)
    
    return {
        "query": query,
        "prediction": prediction,
        "probability": probability,
        "confidence": abs(probability - 0.5) * 2  # Convert to 0-1 confidence
    }


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python classify_query.py <query>")
        print("Example: python classify_query.py 'The Earth is flat'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    # Default parameters (adjust these based on your setup)
    result = classify_query(
        model_path="./models/isuc-v05-26.pt",
        model_type="isuc",
        input_size=4096,  # Adjust based on your model
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        layer_idx=16,
        query=query
    )
    
    print(json.dumps(result, indent=2))