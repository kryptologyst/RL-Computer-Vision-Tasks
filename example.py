#!/usr/bin/env python3
"""Example script demonstrating RL Computer Vision training.

This script shows how to train a DQN agent on CIFAR-10 classification
with minimal configuration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.training import TrainingConfig, Trainer
from src.environments.vision_env import make_vision_env


def main():
    """Run a simple training example."""
    print("🤖 RL Computer Vision - Simple Training Example")
    print("=" * 50)
    
    # Create a simple configuration
    config = TrainingConfig(
        task_type="classification",
        dataset="cifar10",
        algorithm="dqn",
        num_episodes=100,  # Small number for quick demo
        eval_freq=25,
        num_eval_episodes=5,
        save_freq=50,
        seed=42,
        device="auto"
    )
    
    print(f"Configuration:")
    print(f"  Task: {config.task_type}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Device: {config.device}")
    print()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(config)
    
    # Run training
    print("Starting training...")
    training_stats = trainer.train()
    
    # Print results
    print("\nTraining completed!")
    print(f"Final training reward: {training_stats['episode_rewards'][-1]:.3f}")
    
    if training_stats['eval_rewards']:
        print(f"Final evaluation reward: {training_stats['eval_rewards'][-1]:.3f}")
    if training_stats['eval_accuracies']:
        print(f"Final evaluation accuracy: {training_stats['eval_accuracies'][-1]:.3f}")
    
    # Run final evaluation
    print("\nRunning final evaluation...")
    from src.utils.training import Evaluator
    evaluator = Evaluator(trainer.agent, trainer.env)
    eval_results = evaluator.evaluate_performance(num_episodes=10)
    
    print(f"\nFinal Performance:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    print(f"  Mean Accuracy: {eval_results['mean_accuracy']:.3f} ± {eval_results['std_accuracy']:.3f}")
    print(f"  Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}")
    
    print("\n✅ Example completed successfully!")
    print("Check the 'checkpoints' and 'logs' directories for saved models and logs.")


if __name__ == "__main__":
    main()
