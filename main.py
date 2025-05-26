import os
import argparse
import glob
from datetime import datetime

import wandb
from dotenv import load_dotenv

from src.train import train
from prepare_data import collect_internal_states

load_dotenv()


def main():
    # Retrieve the latest version based on the timestamp in the filenames
    model_files = glob.glob("./models/isuc-v*.pt")
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        CURRENT_VERSION = latest_model.split("isuc-v")[-1].split(".pt")[0]
    else:
        CURRENT_VERSION = datetime.now().strftime("%m-%d")

    # Set the new version to today's date
    NEW_VERSION = datetime.now().strftime("%m-%d")
    CURRENT_CHECKPOINT_PATH = f"./models/isuc-v{CURRENT_VERSION}.pt"
    NEW_CHECKPOINT_PATH = f"./models/isuc-v{NEW_VERSION}.pt"

    parser = argparse.ArgumentParser(description="Train a hallucination classifier")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--layer_idx", type=int, default=16)
    parser.add_argument("--current_checkpoint_path", type=str, default=CURRENT_CHECKPOINT_PATH)
    parser.add_argument("--new_checkpoint_path", type=str, default=NEW_CHECKPOINT_PATH)
    parser.add_argument("--dataset", type=str, default=f"true-false")
    
    args = parser.parse_args()

    # login to wandb if not already
    wandb.login()
    wandb.init(project="ISUC")
    
    # prepare data if not already prepared
    DATASET_PATH = f"./data/{args.dataset}/prepared"
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        print(f"Created dataset directory at {DATASET_PATH}")
        collect_internal_states(args.model, args.dataset, args.layer_idx)
    
    # Start training
    model, metrics = train(args)
    
    # terminate run
    wandb.finish()
    print("Final Metrics\n", metrics)
    print("##### RUN TERMINATED #####")


if __name__ == "__main__":
    main()
