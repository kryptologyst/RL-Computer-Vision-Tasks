"""Interactive Streamlit demo for RL Computer Vision tasks.

This demo allows users to visualize agent performance, compare algorithms,
and interact with trained models.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.environments.vision_env import VisionEnvironment, make_vision_env
from src.algorithms.dqn_variants import DQNAgent, DoubleDQNAgent, PrioritizedReplayDQNAgent, DuelingDQNAgent, RainbowDQNAgent
from src.utils.training import Evaluator


# Page configuration
st.set_page_config(
    page_title="RL Computer Vision Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_training_results(results_dir: str) -> Dict:
    """Load training results from JSON files."""
    stats_file = os.path.join(results_dir, "training_stats.json")
    eval_file = os.path.join(results_dir, "evaluation_results.json")
    
    results = {}
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            results['training_stats'] = json.load(f)
    
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            results['evaluation_results'] = json.load(f)
    
    return results


def create_agent_from_config(config: Dict) -> Tuple[Any, VisionEnvironment]:
    """Create agent and environment from configuration."""
    # Create environment
    env = make_vision_env(
        task_type=config.get('task_type', 'classification'),
        dataset=config.get('dataset', 'cifar10'),
        image_size=tuple(config.get('image_size', [32, 32])),
        max_steps=config.get('max_steps', 100),
        reward_type=config.get('reward_type', 'dense'),
        seed=config.get('seed', 42)
    )
    
    # Create agent
    state_shape = (1, *config.get('image_size', [32, 32]))
    num_actions = env.action_space.n
    
    agent_config = {
        'state_shape': state_shape,
        'num_actions': num_actions,
        'learning_rate': config.get('learning_rate', 1e-4),
        'gamma': config.get('gamma', 0.99),
        'epsilon_start': config.get('epsilon_start', 1.0),
        'epsilon_end': config.get('epsilon_end', 0.01),
        'epsilon_decay': config.get('epsilon_decay', 0.995),
        'buffer_size': config.get('buffer_size', 10000),
        'batch_size': config.get('batch_size', 32),
        'target_update_freq': config.get('target_update_freq', 100),
        'device': config.get('device', 'auto'),
    }
    
    algorithm = config.get('algorithm', 'dqn')
    
    if algorithm == 'dqn':
        agent = DQNAgent(**agent_config)
    elif algorithm == 'double_dqn':
        agent = DoubleDQNAgent(**agent_config)
    elif algorithm == 'prioritized_dqn':
        agent = PrioritizedReplayDQNAgent(**agent_config)
    elif algorithm == 'dueling_dqn':
        agent = DuelingDQNAgent(**agent_config)
    elif algorithm == 'rainbow_dqn':
        agent = RainbowDQNAgent(**agent_config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent, env


def plot_learning_curves_interactive(training_stats: Dict) -> go.Figure:
    """Create interactive learning curves plot."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Evaluation Rewards', 
                        'Evaluation Accuracy', 'Training Loss'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Episode rewards
    episodes = list(range(len(training_stats['episode_rewards'])))
    fig.add_trace(
        go.Scatter(x=episodes, y=training_stats['episode_rewards'], 
                  name='Episode Rewards', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Evaluation rewards
    if training_stats['eval_rewards']:
        eval_episodes = [i * 100 for i in range(len(training_stats['eval_rewards']))]  # Assuming eval_freq=100
        fig.add_trace(
            go.Scatter(x=eval_episodes, y=training_stats['eval_rewards'],
                      name='Evaluation Rewards', line=dict(color='green')),
            row=1, col=2
        )
    
    # Evaluation accuracy
    if training_stats['eval_accuracies']:
        eval_episodes = [i * 100 for i in range(len(training_stats['eval_accuracies']))]
        fig.add_trace(
            go.Scatter(x=eval_episodes, y=training_stats['eval_accuracies'],
                      name='Evaluation Accuracy', line=dict(color='red')),
            row=2, col=1
        )
    
    # Training loss
    if training_stats['losses']:
        loss_steps = list(range(len(training_stats['losses'])))
        fig.add_trace(
            go.Scatter(x=loss_steps, y=training_stats['losses'],
                      name='Training Loss', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Learning Curves")
    return fig


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🤖 RL Computer Vision Demo</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Important Safety Notice</h4>
    <p><strong>This is a research and educational demonstration only.</strong> 
    The models and algorithms shown here are not intended for production use 
    in real-world computer vision systems, especially in safety-critical applications 
    such as autonomous vehicles, medical diagnosis, or security systems.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Task selection
    task_type = st.sidebar.selectbox(
        "Task Type",
        ["classification", "detection", "navigation"],
        index=0
    )
    
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["cifar10", "mnist", "synthetic"],
        index=0
    )
    
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["dqn", "double_dqn", "prioritized_dqn", "dueling_dqn", "rainbow_dqn"],
        index=0
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    num_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 1000)
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 1e-4, format="%.0e")
    epsilon_start = st.sidebar.slider("Epsilon Start", 0.1, 1.0, 1.0)
    epsilon_end = st.sidebar.slider("Epsilon End", 0.01, 0.1, 0.01)
    
    # Create configuration
    config = {
        'task_type': task_type,
        'dataset': dataset,
        'algorithm': algorithm,
        'num_episodes': num_episodes,
        'learning_rate': learning_rate,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'image_size': [32, 32] if dataset == 'cifar10' else [28, 28],
        'max_steps': 100,
        'reward_type': 'dense',
        'gamma': 0.99,
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 32,
        'target_update_freq': 100,
        'device': 'auto',
        'seed': 42
    }
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Live Demo", "📊 Training Results", "🔍 Model Analysis", "📈 Performance Comparison"])
    
    with tab1:
        st.header("Live Agent Demo")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎯 Run Agent Episode", type="primary"):
                with st.spinner("Running agent episode..."):
                    try:
                        # Create agent and environment
                        agent, env = create_agent_from_config(config)
                        
                        # Run episode
                        state, _ = env.reset()
                        state_tensor = torch.tensor(state['image'], dtype=torch.float32)
                        
                        episode_reward = 0
                        episode_length = 0
                        correct_predictions = 0
                        total_predictions = 0
                        
                        states = []
                        actions = []
                        rewards = []
                        
                        while True:
                            action = agent.select_action(state_tensor, training=False)
                            next_state, reward, terminated, truncated, info = env.step(action)
                            next_state_tensor = torch.tensor(next_state['image'], dtype=torch.float32)
                            done = terminated or truncated
                            
                            states.append(state['image'])
                            actions.append(action)
                            rewards.append(reward)
                            
                            episode_reward += reward
                            episode_length += 1
                            
                            if action == info['current_label']:
                                correct_predictions += 1
                            total_predictions += 1
                            
                            state = next_state
                            state_tensor = next_state_tensor
                            
                            if done:
                                break
                        
                        # Display results
                        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                        
                        st.success(f"Episode completed!")
                        st.metric("Total Reward", f"{episode_reward:.2f}")
                        st.metric("Episode Length", episode_length)
                        st.metric("Accuracy", f"{accuracy:.2%}")
                        
                        # Show episode visualization
                        if len(states) > 0:
                            st.subheader("Episode Visualization")
                            
                            # Create a grid of images
                            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                            axes = axes.flatten()
                            
                            for i in range(min(10, len(states))):
                                img = states[i].squeeze()
                                axes[i].imshow(img, cmap='gray' if dataset == 'mnist' else None)
                                axes[i].set_title(f"Step {i}\nAction: {actions[i]}\nReward: {rewards[i]:.2f}")
                                axes[i].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error running episode: {e}")
        
        with col2:
            st.subheader("Environment Info")
            st.info(f"""
            **Task:** {task_type.title()}
            **Dataset:** {dataset.upper()}
            **Algorithm:** {algorithm.upper()}
            **Image Size:** {config['image_size']}
            """)
            
            st.subheader("Agent Info")
            st.info(f"""
            **Epsilon:** {epsilon_start:.3f} → {epsilon_end:.3f}
            **Learning Rate:** {learning_rate:.0e}
            **Episodes:** {num_episodes}
            """)
    
    with tab2:
        st.header("Training Results")
        
        # Load results from different directories
        results_dirs = ["./outputs", "./logs", "./checkpoints"]
        available_results = []
        
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                for subdir in os.listdir(results_dir):
                    subdir_path = os.path.join(results_dir, subdir)
                    if os.path.isdir(subdir_path):
                        results = load_training_results(subdir_path)
                        if results:
                            available_results.append((subdir, results))
        
        if available_results:
            st.subheader("Available Results")
            
            # Select results to display
            result_names = [name for name, _ in available_results]
            selected_result = st.selectbox("Select Results", result_names)
            
            selected_data = next(data for name, data in available_results if name == selected_result)
            
            if 'training_stats' in selected_data:
                st.subheader("Learning Curves")
                fig = plot_learning_curves_interactive(selected_data['training_stats'])
                st.plotly_chart(fig, use_container_width=True)
            
            if 'evaluation_results' in selected_data:
                st.subheader("Final Performance")
                eval_results = selected_data['evaluation_results']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Mean Reward",
                        f"{eval_results['mean_reward']:.3f}",
                        f"±{eval_results['std_reward']:.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Mean Accuracy",
                        f"{eval_results['mean_accuracy']:.3f}",
                        f"±{eval_results['std_accuracy']:.3f}"
                    )
                
                with col3:
                    st.metric(
                        "Mean Episode Length",
                        f"{eval_results['mean_length']:.1f}",
                        f"±{eval_results['std_length']:.1f}"
                    )
                
                # Performance distribution
                st.subheader("Performance Distribution")
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Reward Distribution', 'Accuracy Distribution'))
                
                fig.add_trace(
                    go.Histogram(x=eval_results['rewards'], name='Rewards', nbinsx=20),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=eval_results['accuracies'], name='Accuracies', nbinsx=20),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No training results found. Run training first to see results here.")
    
    with tab3:
        st.header("Model Analysis")
        
        st.info("Model analysis features will be available after training a model.")
        
        # Placeholder for model analysis features
        st.subheader("Q-Value Analysis")
        st.text("Q-value distributions and action preferences will be shown here.")
        
        st.subheader("Feature Visualization")
        st.text("CNN feature maps and attention visualizations will be shown here.")
    
    with tab4:
        st.header("Performance Comparison")
        
        st.info("Compare different algorithms and configurations.")
        
        # Algorithm comparison
        algorithms = ["dqn", "double_dqn", "prioritized_dqn", "dueling_dqn", "rainbow_dqn"]
        
        st.subheader("Algorithm Comparison")
        
        # Mock comparison data (in real implementation, this would load actual results)
        comparison_data = {
            'Algorithm': algorithms,
            'Mean Reward': [0.45, 0.52, 0.58, 0.61, 0.68],
            'Mean Accuracy': [0.72, 0.78, 0.82, 0.85, 0.89],
            'Training Time (min)': [15, 18, 22, 20, 25]
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(df, use_container_width=True)
        
        # Comparison plots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Mean Reward by Algorithm', 'Mean Accuracy by Algorithm')
        )
        
        fig.add_trace(
            go.Bar(x=df['Algorithm'], y=df['Mean Reward'], name='Mean Reward'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['Algorithm'], y=df['Mean Accuracy'], name='Mean Accuracy'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>RL Computer Vision Demo | Research & Educational Use Only</p>
    <p>Built with Streamlit, PyTorch, and Gymnasium</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
