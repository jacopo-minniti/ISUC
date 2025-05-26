import os
import sys
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from src.model import ISUC

DEFAULT_MODEL_PATH = "./models/isuc-v05-26.pt"
DEFAULT_LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" # Example, choose one appropriate for your ISUC training
DEFAULT_ACTIVATION_LAYER = 16 # 1-based indexing

def get_llm_activations(text, tokenizer, llm_model, layer_1based, device):
    """
    Gets activations from a specific layer of the LLM for the [CLS] token.
    The hidden_states output from Hugging Face models includes:
    - hidden_states[0]: initial embedding outputs
    - hidden_states[1]: output of the first layer
    - ...
    - hidden_states[N]: output of the Nth layer
    So, for the 16th layer (1-based), we need hidden_states[16].
    """
    if layer_1based < 1 or layer_1based > llm_model.config.num_hidden_layers + 1: # +1 for embedding layer
        raise ValueError(
            f"Activation layer must be between 1 and {llm_model.config.num_hidden_layers +1}. "
            f"Layer {layer_1based} requested."
        )
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    llm_model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        outputs = llm_model(**inputs, output_hidden_states=True)
    
    # hidden_states is a tuple.
    # hidden_states[0] are the embedding outputs.
    # hidden_states[i] is the output of the (i-1)-th layer for i > 0.
    # So, for the 16th layer (1-based), we access hidden_states[16].
    # If layer_1based refers to the output of the Nth transformer block, then it's hidden_states[layer_1based].
    # If it refers to the embedding layer as layer 1, then it's hidden_states[layer_1based-1].
    # The prompt "16th (1based) layer" usually means the output of the 16th transformer block.
    # Let's assume layer_1based means the output of the Nth transformer block.
    # The tuple `outputs.hidden_states` has `num_hidden_layers + 1` elements.
    # `outputs.hidden_states[0]` is the embedding layer.
    # `outputs.hidden_states[1]` is the output of the 1st hidden layer.
    # `outputs.hidden_states[layer_1based]` is the output of the `layer_1based`-th hidden layer.
    
    if layer_1based > len(outputs.hidden_states) -1 :
         raise ValueError(
            f"Requested layer {layer_1based} is out of bounds. "
            f"Model has {len(outputs.hidden_states)-1} hidden layers (plus embeddings)."
            f"Max layer index is {len(outputs.hidden_states)-1}."
        )

    # We want the activations from the specified layer.
    # outputs.hidden_states[0] is embeddings
    # outputs.hidden_states[1] is 1st layer output
    # ...
    # outputs.hidden_states[N] is Nth layer output
    # So, for 16th layer (1-based), we use index 16.
    layer_index_0_based_for_tuple = layer_1based 
    
    activations_at_layer = outputs.hidden_states[layer_index_0_based_for_tuple]
    
    # We'll take the activations of the last token
    cls_activations = activations_at_layer[:, -1, :].squeeze() # Shape: (hidden_size,)
    
    return cls_activations

def run_query_classification(args):
    """
    Processes one or more queries, extracts activations, and classifies using ISUC.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load LLM and Tokenizer
    print(f"Loading LLM tokenizer: {args.llm_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
        print(f"Loading LLM model: {args.llm_model_name}")
        llm_model = AutoModel.from_pretrained(args.llm_model_name).to(device)
    except Exception as e:
        print(f"Error loading LLM or tokenizer '{args.llm_model_name}': {e}")
        return

    # Determine ISUC input size from LLM's hidden size
    isuc_input_size = llm_model.config.hidden_size
    print(f"LLM hidden size (ISUC input_size): {isuc_input_size}")

    # 2. Initialize and Load ISUC model
    print("Initializing ISUC model...")
    isuc_model = ISUC(input_size=isuc_input_size)
    
    if os.path.exists(args.isuc_model_path):
        print(f"Loading ISUC model checkpoint from: {args.isuc_model_path}")
        try:
            isuc_model.load_state_dict(torch.load(args.isuc_model_path, map_location=device))
        except Exception as e:
            print(f"Error loading ISUC model checkpoint: {e}. Classification will use an uninitialized ISUC model.")
    else:
        print(f"Warning: ISUC model checkpoint not found at {args.isuc_model_path}. Classification will use an uninitialized ISUC model.")
    
    isuc_model.to(device)
    isuc_model.eval()

    for query_text in args.queries: # Iterate over each query
        # 3. Get Activations for the query
        print(f"\nProcessing query: \"{query_text}\"")
        print(f"Extracting activations from layer {args.activation_layer_1based} of {args.llm_model_name}...")
        try:
            activations = get_llm_activations(
                query_text, 
                tokenizer, 
                llm_model, 
                args.activation_layer_1based, 
                device
            )
        except ValueError as e:
            print(f"Error getting activations for query \"{query_text}\": {e}")
            continue # Move to the next query
        except Exception as e:
            print(f"An unexpected error occurred while getting activations for query \"{query_text}\": {e}")
            continue # Move to the next query

        if activations is None or activations.nelement() == 0:
            print(f"Failed to extract valid activations for query \"{query_text}\".")
            continue # Move to the next query
            
        # Ensure activations tensor is 2D (batch_size, features) for ISUC model
        if activations.ndim == 1:
            activations = activations.unsqueeze(0) # Add batch dimension

        # 4. Classify with ISUC
        print(f"Classifying query \"{query_text}\" with ISUC model...")
        with torch.no_grad():
            output_prob = isuc_model(activations) # ISUC model returns a probability
            predicted_label = 1 if output_prob.item() > 0.5 else 0
            
        result = "TRUE" if predicted_label == 1 else "FALSE"

        print("\n--- Classification Result ---")
        print(f"  Query: \"{query_text}\"")
        print(f"  Predicted Probability (from ISUC): {output_prob.item():.4f}")
        print(f"  Predicted Label: {predicted_label} ({result})")
        print("---------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify one or more queries using LLM activations and an ISUC model.")
    parser.add_argument(
        "queries", 
        type=str,
        nargs='+', # Accept one or more queries
        help="The natural language query(s) to classify."
    )
    parser.add_argument(
        "--llm_model_name", 
        type=str, 
        default=DEFAULT_LLM_MODEL_NAME,
        help=f"Name of the Hugging Face transformer model to use for activations (default: {DEFAULT_LLM_MODEL_NAME})."
    )
    parser.add_argument(
        "--activation_layer_1based", 
        type=int, 
        default=DEFAULT_ACTIVATION_LAYER,
        help="The 1-based index of the LLM layer from which to extract activations (e.g., 16 for the 16th layer's output). "
             "Note: Layer 1 is the output of the first transformer block."
             f"(default: {DEFAULT_ACTIVATION_LAYER})"
    )
    parser.add_argument(
        "--isuc_model_path", 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained ISUC model checkpoint (default: {DEFAULT_MODEL_PATH})."
    )
    
    args = parser.parse_args()
    run_query_classification(args)