"""Computer Vision Reinforcement Learning Environment.

This module implements a computer vision environment where an RL agent learns to
perform vision-based tasks like image classification and object detection.
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import random


class VisionEnvironment(gym.Env):
    """Computer Vision Environment for RL training.
    
    This environment simulates computer vision tasks where an agent learns to:
    1. Classify images correctly
    2. Detect objects in images
    3. Navigate through image-based decision making
    
    Args:
        task_type: Type of vision task ('classification', 'detection', 'navigation')
        dataset: Dataset to use ('cifar10', 'mnist', 'synthetic')
        image_size: Size of input images (height, width)
        max_steps: Maximum steps per episode
        reward_type: Type of reward function ('sparse', 'dense', 'shaped')
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        dataset: str = "cifar10",
        image_size: Tuple[int, int] = (32, 32),
        max_steps: int = 100,
        reward_type: str = "dense",
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.task_type = task_type
        self.dataset = dataset
        self.image_size = image_size
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.seed = seed
        
        # Set up random seeds
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        else:
            self.np_random = np.random.RandomState()
        
        # Load dataset
        self._load_dataset()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Initialize episode state
        self.current_step = 0
        self.current_image_idx = 0
        self.current_image = None
        self.current_label = None
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    
    def _load_dataset(self) -> None:
        """Load the specified dataset."""
        if self.dataset == "cifar10":
            # Use CIFAR-10 for classification tasks
            self.dataset_obj = CIFAR10(root='./data', train=True, download=True)
            self.num_classes = 10
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                              'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset == "mnist":
            # Use MNIST for simpler classification
            self.dataset_obj = MNIST(root='./data', train=True, download=True)
            self.num_classes = 10
            self.class_names = [str(i) for i in range(10)]
        else:
            # Synthetic dataset
            self.num_classes = 5
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]
            self.dataset_obj = None
    
    def _setup_spaces(self) -> None:
        """Set up action and observation spaces."""
        if self.task_type == "classification":
            # Action space: select class (discrete)
            self.action_space = spaces.Discrete(self.num_classes)
            
            # Observation space: image + metadata
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(1, *self.image_size),
                    dtype=np.float32
                ),
                'step': spaces.Box(
                    low=0, high=self.max_steps,
                    shape=(1,), dtype=np.int32
                ),
                'correct_predictions': spaces.Box(
                    low=0, high=self.max_steps,
                    shape=(1,), dtype=np.int32
                )
            })
        
        elif self.task_type == "detection":
            # Action space: bounding box coordinates + class
            self.action_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(5,),  # [x, y, width, height, class]
                dtype=np.float32
            )
            
            # Observation space: image
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(1, *self.image_size),
                dtype=np.float32
            )
        
        else:  # navigation
            # Action space: movement directions
            self.action_space = spaces.Discrete(4)  # up, down, left, right
            
            # Observation space: image + position
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(1, *self.image_size),
                    dtype=np.float32
                ),
                'position': spaces.Box(
                    low=0.0, high=1.0,
                    shape=(2,), dtype=np.float32
                )
            })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            random.seed(seed)
        
        self.current_step = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Sample new image
        self._sample_new_image()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Calculate reward based on action
        reward = self._calculate_reward(action)
        
        # Update statistics
        if self.task_type == "classification":
            self.total_predictions += 1
            if action == self.current_label:
                self.correct_predictions += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Sample new image for next step (if not done)
        if not terminated:
            self._sample_new_image()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _sample_new_image(self) -> None:
        """Sample a new image from the dataset."""
        if self.dataset_obj is not None:
            # Use real dataset
            idx = self.np_random.randint(0, len(self.dataset_obj))
            image, label = self.dataset_obj[idx]
            
            # Convert PIL image to tensor
            if isinstance(image, Image.Image):
                self.current_image = self.transform(image)
            else:
                self.current_image = torch.tensor(image).float()
                if self.current_image.dim() == 3:
                    self.current_image = self.current_image.unsqueeze(0)
            
            self.current_label = label
        else:
            # Generate synthetic image
            self.current_image = torch.randn(1, *self.image_size)
            self.current_label = self.np_random.randint(0, self.num_classes)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        if self.task_type == "classification":
            return {
                'image': self.current_image.numpy(),
                'step': np.array([self.current_step], dtype=np.int32),
                'correct_predictions': np.array([self.correct_predictions], dtype=np.int32)
            }
        elif self.task_type == "detection":
            return {'image': self.current_image.numpy()}
        else:  # navigation
            position = np.array([0.5, 0.5], dtype=np.float32)  # Center position
            return {
                'image': self.current_image.numpy(),
                'position': position
            }
    
    def _calculate_reward(self, action: Any) -> float:
        """Calculate reward based on action and task type."""
        if self.task_type == "classification":
            if action == self.current_label:
                if self.reward_type == "sparse":
                    return 1.0
                elif self.reward_type == "dense":
                    return 1.0
                else:  # shaped
                    return 1.0 + 0.1 * (self.correct_predictions / max(1, self.total_predictions))
            else:
                if self.reward_type == "sparse":
                    return 0.0
                elif self.reward_type == "dense":
                    return -0.1
                else:  # shaped
                    return -0.1
        
        elif self.task_type == "detection":
            # Simplified detection reward
            return self.np_random.uniform(-0.1, 0.1)
        
        else:  # navigation
            # Simple navigation reward
            return self.np_random.uniform(-0.1, 0.1)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        info = {
            'step': self.current_step,
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,
            'accuracy': self.correct_predictions / max(1, self.total_predictions),
            'current_label': self.current_label,
            'class_name': self.class_names[self.current_label] if self.current_label is not None else None
        }
        return info
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "rgb_array":
            # Convert tensor back to image format
            image = self.current_image.squeeze(0).numpy()
            image = (image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            image = np.clip(image, 0, 1)
            return (image * 255).astype(np.uint8)
        return None
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass


def make_vision_env(
    task_type: str = "classification",
    dataset: str = "cifar10",
    image_size: Tuple[int, int] = (32, 32),
    max_steps: int = 100,
    reward_type: str = "dense",
    seed: Optional[int] = None,
) -> VisionEnvironment:
    """Create a VisionEnvironment instance.
    
    Args:
        task_type: Type of vision task
        dataset: Dataset to use
        image_size: Size of input images
        max_steps: Maximum steps per episode
        reward_type: Type of reward function
        seed: Random seed for reproducibility
        
    Returns:
        Configured VisionEnvironment instance
    """
    return VisionEnvironment(
        task_type=task_type,
        dataset=dataset,
        image_size=image_size,
        max_steps=max_steps,
        reward_type=reward_type,
        seed=seed,
    )
