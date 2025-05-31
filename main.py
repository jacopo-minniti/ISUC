"""
Main entry point for uncertainty classifier experiments using LLM internal states.
"""
import os
import argparse
import wandb
from dotenv import load_dotenv

from src.data_processors import process_dataset
from src.train import train_classifier
from src.utils import build_training_config, print_experiment_info, setup_directories

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Train uncertainty classifiers using LLM internal states")
    
    # Model and data arguments
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default="true-false",
                       help="Dataset name")
    parser.add_argument("--layer_idx", type=int, default=16,
                       help="Layer index for activation extraction")
    
    # Training arguments
    parser.add_argument("--classifier_type", type=str, default="SAPLMA", 
                       choices=["SAPLMA"],
                       help="Type of classifier to train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    
    # Paths
    parser.add_argument("--current_checkpoint_path", type=str, default=None,
                       help="Path to current model checkpoint")
    parser.add_argument("--new_checkpoint_path", type=str, default=None,
                       help="Path to save new model checkpoint")
    
    # Experiment control
    parser.add_argument("--skip_data_processing", action="store_true",
                       help="Skip data processing if already done")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="ISUC",
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Setup directories
    dirs = setup_directories(args.dataset)
    
    # Prepare data if needed
    if not args.skip_data_processing:
        prepared_path = dirs["prepared"]
        train_file = os.path.join(prepared_path, "train.json")
        test_file = os.path.join(prepared_path, "test.json")
        
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Preparing dataset: {args.dataset}")
            success = process_dataset(
                dataset_name=args.dataset,
                model_id=args.model,
                layer_idx=args.layer_idx,
                output_dir=prepared_path
            )
            if not success:
                print("Failed to process dataset. Exiting.")
                return
        else:
            print(f"Dataset already prepared at {prepared_path}")
    
    # Build training configuration
    config = build_training_config(args)
    print_experiment_info(config)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=f"{args.dataset}_{args.classifier_type}_{args.model.split('/')[-1]}"
        )
    
    try:
        # Train the classifier
        print("Starting training...")
        model, final_metrics = train_classifier(config)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print("Final Metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Clean up wandb
        if args.use_wandb and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
