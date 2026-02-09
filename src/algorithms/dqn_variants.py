"""Advanced Reinforcement Learning algorithms for computer vision tasks."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
from dataclasses import dataclass
import math

from ..models.vision_networks import VisionQNetwork, DuelingVisionQNetwork, VisionPolicyNetwork, VisionValueNetwork


@dataclass
class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    capacity: int
    states: deque
    actions: deque
    rewards: deque
    next_states: deque
    dones: deque
    priorities: Optional[deque] = None
    
    def __init__(self, capacity: int, use_prioritized: bool = False):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        
        if use_prioritized:
            self.priorities = deque(maxlen=capacity)
        else:
            self.priorities = None
    
    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool, priority: float = 1.0):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if self.priorities is not None:
            self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List, Optional[List]]:
        """Sample a batch of transitions."""
        if len(self.states) < batch_size:
            batch_size = len(self.states)
        
        if self.priorities is not None:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.states), batch_size, p=probs)
        else:
            # Uniform sampling
            indices = random.sample(range(len(self.states)), batch_size)
        
        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_rewards = [self.rewards[i] for i in indices]
        batch_next_states = [self.next_states[i] for i in indices]
        batch_dones = [self.dones[i] for i in indices]
        
        if self.priorities is not None:
            batch_priorities = [self.priorities[i] for i in indices]
            return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_priorities
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, None
    
    def __len__(self) -> int:
        return len(self.states)


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        num_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = "auto",
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.q_network = VisionQNetwork(
            input_channels=state_shape[0],
            image_size=state_shape[1:],
            num_actions=num_actions,
        ).to(self.device)
        
        self.target_network = VisionQNetwork(
            input_channels=state_shape[0],
            image_size=state_shape[1:],
            num_actions=num_actions,
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.step_count = 0
        self.losses = []
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state.to(self.device))
            return q_values.argmax().item()
    
    def store_transition(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """Update the Q-network using a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, _ = batch
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent to reduce overestimation bias."""
    
    def update(self) -> Optional[float]:
        """Update using Double DQN algorithm."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, _ = batch
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()


class PrioritizedReplayDQNAgent(DQNAgent):
    """DQN agent with prioritized experience replay."""
    
    def __init__(self, *args, alpha: float = 0.6, beta_start: float = 0.4, 
                 beta_end: float = 1.0, beta_decay: float = 0.995, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_decay = beta_decay
        
        # Use prioritized replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer.capacity, use_prioritized=True)
    
    def store_transition(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool, td_error: float = 1.0):
        """Store transition with priority based on TD error."""
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.replay_buffer.push(state, action, reward, next_state, done, priority)
    
    def update(self) -> Optional[float]:
        """Update using prioritized experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch with priorities
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, priorities = batch
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        priorities = torch.tensor(priorities, dtype=torch.float32).to(self.device)
        
        # Compute importance sampling weights
        weights = (len(self.replay_buffer) * priorities) ** (-self.beta)
        weights = weights / weights.max()
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss with importance sampling
        td_errors = current_q_values.squeeze() - target_q_values
        loss = (weights * td_errors ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities with new TD errors
        new_priorities = (abs(td_errors.detach().cpu().numpy()) + 1e-6) ** self.alpha
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon and beta
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        if self.beta < self.beta_end:
            self.beta *= self.beta_decay
        
        self.losses.append(loss.item())
        return loss.item()


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN agent using dueling architecture."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace networks with dueling versions
        self.q_network = DuelingVisionQNetwork(
            input_channels=self.state_shape[0],
            image_size=self.state_shape[1:],
            num_actions=self.num_actions,
        ).to(self.device)
        
        self.target_network = DuelingVisionQNetwork(
            input_channels=self.state_shape[0],
            image_size=self.state_shape[1:],
            num_actions=self.num_actions,
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reinitialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)


class RainbowDQNAgent(DQNAgent):
    """Rainbow DQN combining multiple improvements."""
    
    def __init__(self, *args, num_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.z = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        
        # Use dueling architecture
        self.q_network = DuelingVisionQNetwork(
            input_channels=self.state_shape[0],
            image_size=self.state_shape[1:],
            num_actions=self.num_actions,
        ).to(self.device)
        
        self.target_network = DuelingVisionQNetwork(
            input_channels=self.state_shape[0],
            image_size=self.state_shape[1:],
            num_actions=self.num_actions,
        ).to(self.device)
        
        # Add distributional head
        self.q_network.fc3 = nn.Linear(self.q_network.fc2.out_features, self.num_actions * num_atoms)
        self.target_network.fc3 = nn.Linear(self.target_network.fc2.out_features, self.num_actions * num_atoms)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reinitialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Use prioritized replay
        self.replay_buffer = ReplayBuffer(self.replay_buffer.capacity, use_prioritized=True)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using distributional Q-values."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            q_dist = self.q_network(state.to(self.device))
            q_dist = q_dist.view(-1, self.num_actions, self.num_atoms)
            q_values = (q_dist * self.z).sum(dim=2)
            return q_values.argmax().item()
    
    def update(self) -> Optional[float]:
        """Update using distributional DQN with prioritized replay."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch with priorities
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, priorities = batch
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        priorities = torch.tensor(priorities, dtype=torch.float32).to(self.device)
        
        # Compute importance sampling weights
        weights = (len(self.replay_buffer) * priorities) ** (-self.beta)
        weights = weights / weights.max()
        
        # Current distribution
        current_dist = self.q_network(states).view(-1, self.num_actions, self.num_atoms)
        current_dist = current_dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_atoms))
        
        # Target distribution
        with torch.no_grad():
            next_dist = self.target_network(next_states).view(-1, self.num_actions, self.num_atoms)
            next_actions = next_dist.mean(dim=2).argmax(dim=1)
            next_dist = next_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_atoms))
            
            # Project distribution
            target_z = rewards.unsqueeze(1) + self.gamma * self.z.unsqueeze(0) * ~dones.unsqueeze(1)
            target_z = torch.clamp(target_z, self.v_min, self.v_max)
            b = (target_z - self.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()
            
            target_dist = torch.zeros_like(next_dist)
            target_dist.scatter_add_(2, l, next_dist * (u.float() - b))
            target_dist.scatter_add_(2, u, next_dist * (b - l.float()))
        
        # Compute loss
        loss = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=2)
        loss = (weights * loss.squeeze()).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            td_errors = torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=2).squeeze()
            new_priorities = (abs(td_errors.cpu().numpy()) + 1e-6) ** self.alpha
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon and beta
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        if self.beta < self.beta_end:
            self.beta *= self.beta_decay
        
        self.losses.append(loss.item())
        return loss.item()
