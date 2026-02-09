"""Training and evaluation utilities for computer vision RL tasks."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

from ..environments.vision_env import VisionEnvironment
from ..algorithms.dqn_variants import DQNAgent, DoubleDQNAgent, PrioritizedReplayDQNAgent, DuelingDQNAgent, RainbowDQNAgent


@dataclass
class TrainingConfig:
    """Configuration for training RL agents."""
    
    # Environment settings
    task_type: str = "classification"
    dataset: str = "cifar10"
    image_size: Tuple[int, int] = (32, 32)
    max_steps: int = 100
    reward_type: str = "dense"
    
    # Agent settings
    algorithm: str = "dqn"  # dqn, double_dqn, prioritized_dqn, dueling_dqn, rainbow_dqn
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    
    # Training settings
    num_episodes: int = 1000
    eval_freq: int = 100
    num_eval_episodes: int = 10
    save_freq: int = 500
    
    # Device and reproducibility
    device: str = "auto"
    seed: Optional[int] = 42
    
    # Logging and saving
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    use_wandb: bool = False
    project_name: str = "rl-computer-vision"


class Trainer:
    """Trainer class for RL agents on computer vision tasks."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_seeds()
        
        # Initialize environment
        self.env = VisionEnvironment(
            task_type=config.task_type,
            dataset=config.dataset,
            image_size=config.image_size,
            max_steps=config.max_steps,
            reward_type=config.reward_type,
            seed=config.seed,
        )
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_accuracies = []
        self.losses = []
        
        # Initialize wandb if requested
        if config.use_wandb:
            self._init_wandb()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories."""
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        Path("./assets").mkdir(parents=True, exist_ok=True)
    
    def setup_seeds(self):
        """Set random seeds for reproducibility."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_agent(self):
        """Create the appropriate agent based on configuration."""
        state_shape = (1, *self.config.image_size)
        num_actions = self.env.action_space.n
        
        agent_config = {
            'state_shape': state_shape,
            'num_actions': num_actions,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'epsilon_start': self.config.epsilon_start,
            'epsilon_end': self.config.epsilon_end,
            'epsilon_decay': self.config.epsilon_decay,
            'buffer_size': self.config.buffer_size,
            'batch_size': self.config.batch_size,
            'target_update_freq': self.config.target_update_freq,
            'device': self.config.device,
        }
        
        if self.config.algorithm == "dqn":
            return DQNAgent(**agent_config)
        elif self.config.algorithm == "double_dqn":
            return DoubleDQNAgent(**agent_config)
        elif self.config.algorithm == "prioritized_dqn":
            return PrioritizedReplayDQNAgent(**agent_config)
        elif self.config.algorithm == "dueling_dqn":
            return DuelingDQNAgent(**agent_config)
        elif self.config.algorithm == "rainbow_dqn":
            return RainbowDQNAgent(**agent_config)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            wandb.init(
                project=self.config.project_name,
                config=asdict(self.config),
                name=f"{self.config.algorithm}_{self.config.task_type}_{self.config.dataset}",
            )
        except ImportError:
            self.logger.warning("wandb not installed, skipping wandb logging")
    
    def train(self) -> Dict[str, List[float]]:
        """Train the agent."""
        self.logger.info(f"Starting training with {self.config.algorithm} on {self.config.task_type} task")
        
        start_time = time.time()
        
        for episode in tqdm(range(self.config.num_episodes), desc="Training"):
            episode_reward, episode_length = self._train_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log training progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                self.logger.info(
                    f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                    f"Avg Length = {avg_length:.2f}, Epsilon = {self.agent.epsilon:.3f}"
                )
            
            # Evaluation
            if episode % self.config.eval_freq == 0:
                eval_reward, eval_accuracy = self.evaluate()
                self.eval_rewards.append(eval_reward)
                self.eval_accuracies.append(eval_accuracy)
                
                self.logger.info(
                    f"Evaluation at episode {episode}: "
                    f"Reward = {eval_reward:.2f}, Accuracy = {eval_accuracy:.2f}"
                )
            
            # Save checkpoint
            if episode % self.config.save_freq == 0:
                self.save_checkpoint(episode)
            
            # Log to wandb
            if self.config.use_wandb:
                self._log_to_wandb(episode)
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_eval_reward, final_eval_accuracy = self.evaluate()
        self.logger.info(f"Final evaluation: Reward = {final_eval_reward:.2f}, Accuracy = {final_eval_accuracy:.2f}")
        
        # Save final model
        self.save_checkpoint(self.config.num_episodes, is_final=True)
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_rewards': self.eval_rewards,
            'eval_accuracies': self.eval_accuracies,
            'losses': self.losses,
        }
    
    def _train_episode(self) -> Tuple[float, int]:
        """Train for one episode."""
        state, _ = self.env.reset()
        state = torch.tensor(state['image'], dtype=torch.float32)
        
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = torch.tensor(next_state['image'], dtype=torch.float32)
            done = terminated or truncated
            
            # Store transition
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                loss = self.agent.update()
                if loss is not None:
                    self.losses.append(loss)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        return episode_reward, episode_length
    
    def evaluate(self, num_episodes: Optional[int] = None) -> Tuple[float, float]:
        """Evaluate the agent."""
        if num_episodes is None:
            num_episodes = self.config.num_eval_episodes
        
        eval_rewards = []
        eval_accuracies = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state['image'], dtype=torch.float32)
            
            episode_reward = 0
            correct_predictions = 0
            total_predictions = 0
            
            while True:
                # Select action (no exploration)
                action = self.agent.select_action(state, training=False)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = torch.tensor(next_state['image'], dtype=torch.float32)
                done = terminated or truncated
                
                episode_reward += reward
                
                # Track accuracy for classification tasks
                if self.config.task_type == "classification":
                    if action == info['current_label']:
                        correct_predictions += 1
                    total_predictions += 1
                
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            if total_predictions > 0:
                eval_accuracies.append(correct_predictions / total_predictions)
            else:
                eval_accuracies.append(0.0)
        
        return np.mean(eval_rewards), np.mean(eval_accuracies)
    
    def save_checkpoint(self, episode: int, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.q_network.state_dict(),
            'target_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': asdict(self.config),
            'training_stats': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'eval_rewards': self.eval_rewards,
                'eval_accuracies': self.eval_accuracies,
                'losses': self.losses,
            }
        }
        
        if is_final:
            filename = f"final_model.pth"
        else:
            filename = f"checkpoint_episode_{episode}.pth"
        
        filepath = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.agent.device)
        
        self.agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.episode_rewards = checkpoint['training_stats']['episode_rewards']
        self.episode_lengths = checkpoint['training_stats']['episode_lengths']
        self.eval_rewards = checkpoint['training_stats']['eval_rewards']
        self.eval_accuracies = checkpoint['training_stats']['eval_accuracies']
        self.losses = checkpoint['training_stats']['losses']
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
    
    def _log_to_wandb(self, episode: int):
        """Log metrics to wandb."""
        if not self.config.use_wandb:
            return
        
        try:
            import wandb
            
            log_dict = {
                'episode': episode,
                'epsilon': self.agent.epsilon,
                'avg_reward_10': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
                'avg_length_10': np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else 0,
            }
            
            if self.eval_rewards:
                log_dict['eval_reward'] = self.eval_rewards[-1]
            if self.eval_accuracies:
                log_dict['eval_accuracy'] = self.eval_accuracies[-1]
            if self.losses:
                log_dict['avg_loss_10'] = np.mean(self.losses[-10:]) if len(self.losses) >= 10 else 0
            
            wandb.log(log_dict)
        except ImportError:
            pass


class Evaluator:
    """Evaluation utilities for trained RL agents."""
    
    def __init__(self, agent, env: VisionEnvironment):
        self.agent = agent
        self.env = env
    
    def evaluate_performance(self, num_episodes: int = 100) -> Dict[str, float]:
        """Comprehensive evaluation of agent performance."""
        rewards = []
        accuracies = []
        episode_lengths = []
        
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            state, _ = self.env.reset()
            state = torch.tensor(state['image'], dtype=torch.float32)
            
            episode_reward = 0
            episode_length = 0
            correct_predictions = 0
            total_predictions = 0
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = torch.tensor(next_state['image'], dtype=torch.float32)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if action == info['current_label']:
                    correct_predictions += 1
                total_predictions += 1
                
                state = next_state
                
                if done:
                    break
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if total_predictions > 0:
                accuracies.append(correct_predictions / total_predictions)
            else:
                accuracies.append(0.0)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'rewards': rewards,
            'accuracies': accuracies,
            'episode_lengths': episode_lengths,
        }
    
    def plot_learning_curves(self, training_stats: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Evaluation rewards
        if training_stats['eval_rewards']:
            eval_episodes = np.arange(0, len(training_stats['eval_rewards'])) * self.config.eval_freq
            axes[0, 1].plot(eval_episodes, training_stats['eval_rewards'])
            axes[0, 1].set_title('Evaluation Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True)
        
        # Evaluation accuracy
        if training_stats['eval_accuracies']:
            eval_episodes = np.arange(0, len(training_stats['eval_accuracies'])) * self.config.eval_freq
            axes[1, 0].plot(eval_episodes, training_stats['eval_accuracies'])
            axes[1, 0].set_title('Evaluation Accuracy')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].grid(True)
        
        # Losses
        if training_stats['losses']:
            axes[1, 1].plot(training_stats['losses'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_report(self, eval_results: Dict[str, float]) -> str:
        """Create a performance report."""
        report = f"""
Performance Report
==================

Mean Reward: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}
Mean Accuracy: {eval_results['mean_accuracy']:.3f} ± {eval_results['std_accuracy']:.3f}
Mean Episode Length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f}

Confidence Intervals (95%):
- Reward: [{eval_results['mean_reward'] - 1.96 * eval_results['std_reward']:.3f}, 
           {eval_results['mean_reward'] + 1.96 * eval_results['std_reward']:.3f}]
- Accuracy: [{eval_results['mean_accuracy'] - 1.96 * eval_results['std_accuracy']:.3f}, 
            {eval_results['mean_accuracy'] + 1.96 * eval_results['std_accuracy']:.3f}]
"""
        return report
