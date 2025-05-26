import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def collect_internal_states(model_id: str, dataset: str, layer_idx: int) -> None:
    """
    Collects internal states (activations) from a specified layer of a language model
    for a given dataset, processes the data, and saves it in JSON format.

    Args:
        model_id (str): The Hugging Face model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf").
        dataset (str): The name of the dataset to process (e.g., "true-false").
        layer_idx (int): The index of the layer from which to extract activations (0-indexed for transformer layers).
    """
    base_data_path = "./data"
    output_dir = os.path.join(base_data_path, dataset, "prepared")
    os.makedirs(output_dir, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval() # Set model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Add padding token if it doesn't exist (common for some models like Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Dataset dispatcher
    if dataset == "true-false":
        success = process_true_false_dataset(
            tokenizer, model, model_id, device, layer_idx,
            base_data_path, output_dir
        )
        if not success:
            print("Failed to process true-false dataset.")
    else:
        print(f"Dataset '{dataset}' is not supported yet.")

    print("Finished collecting internal states.")



# Dataset handler functions
def process_true_false_dataset(tokenizer, model, model_id, device, layer_idx, base_data_path, output_dir): # Added layer_idx, removed hook_key, activations_cache
    """Process the true-false dataset specifically."""
    original_data_path = os.path.join(base_data_path, "true-false", "original")
    train_data = []
    test_data = []
    test_file_name = "inventions_true_false.csv"

    files_to_process = [f for f in os.listdir(original_data_path) if f.endswith(".csv")]
    if not files_to_process:
        print(f"No CSV files found in {original_data_path}")
        return False

    for filename in files_to_process:
        file_path = os.path.join(original_data_path, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            continue

        print(f"Processing {filename}...")
        for i, (_, row) in enumerate(df.iterrows()):
            statement = str(row["statement"])
            label = int(row["label"])
            
            try:
                inputs = tokenizer(statement, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
                
                with torch.no_grad():
                    # Request hidden states directly from the model output
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                
                layer_activations_list = None # Initialize
                # outputs.hidden_states is a tuple: (embeddings, layer1_hidden, layer2_hidden, ...)
                # So, for layer_idx (0-indexed transformer layer), we need index layer_idx + 1
                target_hidden_state_index = layer_idx + 1
                if target_hidden_state_index < len(outputs.hidden_states):
                    layer_activations_tensor = outputs.hidden_states[target_hidden_state_index].detach().cpu()
                    # Activations tensor shape: (batch_size, seq_len, hidden_dim). Squeeze batch_size (1).
                    # Get activations for the last token only.
                    layer_activations_list = layer_activations_tensor.squeeze(0)[-1, :].tolist()
                else:
                    print(f"Warning: Layer index {layer_idx} (target_hidden_state_index {target_hidden_state_index}) is out of bounds for hidden_states (length {len(outputs.hidden_states)}). Statement: '{statement[:50]}...'.")

                data_entry = {
                    "statement": statement,
                    "label": label,
                    "model": model_id,
                    "activations": layer_activations_list,
                    "source_file": filename
                }

                if filename == test_file_name:
                    test_data.append(data_entry)
                else:
                    train_data.append(data_entry)

            except Exception as e_inner:
                print(f"Error processing statement '{statement[:50]}...' from {filename}: {e_inner}")
                continue
        print(f"Finished processing {filename}.")

    # Save the processed data
    train_output_path = os.path.join(output_dir, "train_data.json")
    test_output_path = os.path.join(output_dir, "test_data.json")

    try:
        with open(train_output_path, "w") as f:
            json.dump(train_data, f, indent=4)
        print(f"Training data saved to {train_output_path}")

        with open(test_output_path, "w") as f:
            json.dump(test_data, f, indent=4)
        print(f"Test data saved to {test_output_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON files: {e}")
        return False