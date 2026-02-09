#!/usr/bin/env python3
"""Main training script for RL Computer Vision tasks.

This script provides a command-line interface for training RL agents on
computer vision tasks with various algorithms and configurations.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.training import Trainer, TrainingConfig
from src.utils.training import Evaluator
from src.environments.vision_env import VisionEnvironment


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_config_from_args(args) -> TrainingConfig:
    """Create TrainingConfig from command line arguments."""
    config_dict = load_config(args.config)
    
    # Override with command line arguments
    if args.algorithm:
        config_dict['algorithm'] = args.algorithm
    if args.episodes:
        config_dict['num_episodes'] = args.episodes
    if args.device:
        config_dict['device'] = args.device
    if args.seed:
        config_dict['seed'] = args.seed
    if args.wandb:
        config_dict['use_wandb'] = args.wandb
    
    return TrainingConfig(**config_dict)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train RL agents on computer vision tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        choices=["dqn", "double_dqn", "prioritized_dqn", "dueling_dqn", "rainbow_dqn"],
        help="RL algorithm to use (overrides config)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        help="Number of training episodes (overrides config)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--eval-only", 
        action="store_true",
        help="Only run evaluation (requires trained model)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str,
        help="Path to checkpoint file for evaluation or resuming training"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs",
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration
        config = create_config_from_args(args)
        
        print(f"Training Configuration:")
        print(f"  Algorithm: {config.algorithm}")
        print(f"  Task: {config.task_type}")
        print(f"  Dataset: {config.dataset}")
        print(f"  Episodes: {config.num_episodes}")
        print(f"  Device: {config.device}")
        print(f"  Seed: {config.seed}")
        print()
        
        # Initialize trainer
        trainer = Trainer(config)
        
        if args.eval_only:
            # Evaluation only mode
            if not args.checkpoint:
                print("Error: --checkpoint required for evaluation mode")
                sys.exit(1)
            
            print(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
            
            print("Running evaluation...")
            evaluator = Evaluator(trainer.agent, trainer.env)
            eval_results = evaluator.evaluate_performance(num_episodes=100)
            
            print("\nEvaluation Results:")
            print(f"  Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
            print(f"  Mean Accuracy: {eval_results['mean_accuracy']:.3f} ± {eval_results['std_accuracy']:.3f}")
            print(f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            
            # Save evaluation results
            import json
            eval_file = os.path.join(args.output_dir, "evaluation_results.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"Evaluation results saved to: {eval_file}")
            
        else:
            # Training mode
            print("Starting training...")
            training_stats = trainer.train()
            
            print("\nTraining completed!")
            print(f"Final training reward: {training_stats['episode_rewards'][-1]:.3f}")
            if training_stats['eval_rewards']:
                print(f"Final evaluation reward: {training_stats['eval_rewards'][-1]:.3f}")
            if training_stats['eval_accuracies']:
                print(f"Final evaluation accuracy: {training_stats['eval_accuracies'][-1]:.3f}")
            
            # Run final evaluation
            print("\nRunning final evaluation...")
            evaluator = Evaluator(trainer.agent, trainer.env)
            eval_results = evaluator.evaluate_performance(num_episodes=100)
            
            print("\nFinal Evaluation Results:")
            print(f"  Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
            print(f"  Mean Accuracy: {eval_results['mean_accuracy']:.3f} ± {eval_results['std_accuracy']:.3f}")
            print(f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
            
            # Save training statistics and evaluation results
            import json
            stats_file = os.path.join(args.output_dir, "training_stats.json")
            eval_file = os.path.join(args.output_dir, "evaluation_results.json")
            
            with open(stats_file, 'w') as f:
                json.dump(training_stats, f, indent=2)
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            print(f"Training statistics saved to: {stats_file}")
            print(f"Evaluation results saved to: {eval_file}")
            
            # Plot learning curves
            try:
                evaluator.plot_learning_curves(training_stats, 
                                             save_path=os.path.join(args.output_dir, "learning_curves.png"))
                print(f"Learning curves saved to: {os.path.join(args.output_dir, 'learning_curves.png')}")
            except Exception as e:
                print(f"Warning: Could not save learning curves: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
