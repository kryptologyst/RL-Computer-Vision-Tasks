"""Tests for RL Computer Vision project."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.environments.vision_env import VisionEnvironment, make_vision_env
from src.models.vision_networks import VisionQNetwork, DuelingVisionQNetwork, VisionPolicyNetwork, VisionValueNetwork
from src.algorithms.dqn_variants import DQNAgent, DoubleDQNAgent, PrioritizedReplayDQNAgent, DuelingDQNAgent, RainbowDQNAgent
from src.utils.training import TrainingConfig, Trainer, Evaluator


class TestVisionEnvironment:
    """Test the vision environment."""
    
    def test_environment_creation(self):
        """Test environment creation."""
        env = VisionEnvironment(task_type="classification", dataset="synthetic", seed=42)
        assert env.task_type == "classification"
        assert env.dataset == "synthetic"
        assert env.seed == 42
    
    def test_environment_reset(self):
        """Test environment reset."""
        env = VisionEnvironment(task_type="classification", dataset="synthetic", seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, dict)
        assert 'image' in obs
        assert obs['image'].shape == (1, 32, 32)
        assert isinstance(info, dict)
    
    def test_environment_step(self):
        """Test environment step."""
        env = VisionEnvironment(task_type="classification", dataset="synthetic", seed=42)
        obs, info = env.reset()
        
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_environment_spaces(self):
        """Test action and observation spaces."""
        env = VisionEnvironment(task_type="classification", dataset="synthetic", seed=42)
        
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert env.action_space.n == 10  # Default number of classes


class TestVisionNetworks:
    """Test neural network models."""
    
    def test_vision_q_network(self):
        """Test VisionQNetwork."""
        network = VisionQNetwork(
            input_channels=1,
            image_size=(32, 32),
            num_actions=10
        )
        
        # Test forward pass
        x = torch.randn(1, 1, 32, 32)
        output = network(x)
        
        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()
    
    def test_dueling_q_network(self):
        """Test DuelingVisionQNetwork."""
        network = DuelingVisionQNetwork(
            input_channels=1,
            image_size=(32, 32),
            num_actions=10
        )
        
        # Test forward pass
        x = torch.randn(1, 1, 32, 32)
        output = network(x)
        
        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()
    
    def test_policy_network(self):
        """Test VisionPolicyNetwork."""
        network = VisionPolicyNetwork(
            input_channels=1,
            image_size=(32, 32),
            num_actions=10
        )
        
        # Test forward pass
        x = torch.randn(1, 1, 32, 32)
        logits = network(x)
        probs = network.get_action_probs(x)
        log_probs = network.get_action_log_probs(x)
        
        assert logits.shape == (1, 10)
        assert probs.shape == (1, 10)
        assert log_probs.shape == (1, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1))
    
    def test_value_network(self):
        """Test VisionValueNetwork."""
        network = VisionValueNetwork(
            input_channels=1,
            image_size=(32, 32)
        )
        
        # Test forward pass
        x = torch.randn(1, 1, 32, 32)
        value = network(x)
        
        assert value.shape == (1, 1)
        assert not torch.isnan(value).any()


class TestDQNAgents:
    """Test DQN agent implementations."""
    
    def test_dqn_agent_creation(self):
        """Test DQN agent creation."""
        agent = DQNAgent(
            state_shape=(1, 32, 32),
            num_actions=10,
            device="cpu"
        )
        
        assert agent.state_shape == (1, 32, 32)
        assert agent.num_actions == 10
        assert agent.device.type == "cpu"
    
    def test_dqn_action_selection(self):
        """Test DQN action selection."""
        agent = DQNAgent(
            state_shape=(1, 32, 32),
            num_actions=10,
            device="cpu"
        )
        
        state = torch.randn(1, 32, 32)
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 10
    
    def test_dqn_experience_replay(self):
        """Test DQN experience replay."""
        agent = DQNAgent(
            state_shape=(1, 32, 32),
            num_actions=10,
            buffer_size=100,
            batch_size=16,
            device="cpu"
        )
        
        # Add some transitions
        for _ in range(20):
            state = torch.randn(1, 32, 32)
            action = np.random.randint(0, 10)
            reward = np.random.random()
            next_state = torch.randn(1, 32, 32)
            done = np.random.random() > 0.9
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Test update
        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_double_dqn_agent(self):
        """Test Double DQN agent."""
        agent = DoubleDQNAgent(
            state_shape=(1, 32, 32),
            num_actions=10,
            device="cpu"
        )
        
        assert isinstance(agent, DoubleDQNAgent)
        
        # Test update
        for _ in range(20):
            state = torch.randn(1, 32, 32)
            action = np.random.randint(0, 10)
            reward = np.random.random()
            next_state = torch.randn(1, 32, 32)
            done = np.random.random() > 0.9
            
            agent.store_transition(state, action, reward, next_state, done)
        
        loss = agent.update()
        assert loss is not None
    
    def test_prioritized_replay_agent(self):
        """Test Prioritized Replay DQN agent."""
        agent = PrioritizedReplayDQNAgent(
            state_shape=(1, 32, 32),
            num_actions=10,
            device="cpu"
        )
        
        assert isinstance(agent, PrioritizedReplayDQNAgent)
        assert agent.replay_buffer.priorities is not None
    
    def test_dueling_dqn_agent(self):
        """Test Dueling DQN agent."""
        agent = DuelingDQNAgent(
            state_shape=(1, 32, 32),
            num_actions=10,
            device="cpu"
        )
        
        assert isinstance(agent, DuelingDQNAgent)
    
    def test_rainbow_dqn_agent(self):
        """Test Rainbow DQN agent."""
        agent = RainbowDQNAgent(
            state_shape=(1, 32, 32),
            num_actions=10,
            device="cpu"
        )
        
        assert isinstance(agent, RainbowDQNAgent)
        assert agent.replay_buffer.priorities is not None


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_config_creation(self):
        """Test TrainingConfig creation."""
        config = TrainingConfig(
            task_type="classification",
            dataset="cifar10",
            algorithm="dqn",
            num_episodes=100,
            seed=42
        )
        
        assert config.task_type == "classification"
        assert config.dataset == "cifar10"
        assert config.algorithm == "dqn"
        assert config.num_episodes == 100
        assert config.seed == 42


class TestTrainer:
    """Test trainer functionality."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        config = TrainingConfig(
            task_type="classification",
            dataset="synthetic",
            algorithm="dqn",
            num_episodes=10,
            seed=42
        )
        
        trainer = Trainer(config)
        
        assert trainer.config == config
        assert trainer.env is not None
        assert trainer.agent is not None
    
    def test_trainer_evaluation(self):
        """Test trainer evaluation."""
        config = TrainingConfig(
            task_type="classification",
            dataset="synthetic",
            algorithm="dqn",
            num_episodes=10,
            seed=42
        )
        
        trainer = Trainer(config)
        
        eval_reward, eval_accuracy = trainer.evaluate(num_episodes=5)
        
        assert isinstance(eval_reward, float)
        assert isinstance(eval_accuracy, float)
        assert 0 <= eval_accuracy <= 1


class TestEvaluator:
    """Test evaluator functionality."""
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        config = TrainingConfig(
            task_type="classification",
            dataset="synthetic",
            algorithm="dqn",
            num_episodes=10,
            seed=42
        )
        
        trainer = Trainer(config)
        evaluator = Evaluator(trainer.agent, trainer.env)
        
        assert evaluator.agent is not None
        assert evaluator.env is not None
    
    def test_evaluator_performance(self):
        """Test evaluator performance evaluation."""
        config = TrainingConfig(
            task_type="classification",
            dataset="synthetic",
            algorithm="dqn",
            num_episodes=10,
            seed=42
        )
        
        trainer = Trainer(config)
        evaluator = Evaluator(trainer.agent, trainer.env)
        
        results = evaluator.evaluate_performance(num_episodes=5)
        
        assert 'mean_reward' in results
        assert 'mean_accuracy' in results
        assert 'mean_length' in results
        assert isinstance(results['mean_reward'], float)
        assert isinstance(results['mean_accuracy'], float)
        assert isinstance(results['mean_length'], float)


if __name__ == "__main__":
    pytest.main([__file__])
