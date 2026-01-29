"""
Day 20: AI Agent - Autonomous Agent Creation and Orchestration
Demonstrates AI agent concepts, decision making, and orchestration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Define AI Agent Framework
# ========================================
print("Creating AI Agent Framework...")

class SimpleAIAgent:
    def __init__(self, name, initial_state=0):
        self.name = name
        self.state = initial_state
        self.history = [initial_state]
        self.actions_taken = []
        self.rewards = []
        self.decision_log = []
    
    def perceive(self, environment_state):
        """Agent perceives the environment"""
        return environment_state
    
    def think(self, observation):
        """Agent processes observation and makes decision"""
        # Simple decision making based on observation
        if observation > 5:
            action = "explore"
        elif observation < -5:
            action = "retreat"
        else:
            action = "maintain"
        return action
    
    def act(self, action, environment):
        """Agent performs action and receives reward"""
        if action == "explore":
            reward = np.random.uniform(1, 3)
            new_state = self.state + reward
        elif action == "retreat":
            reward = np.random.uniform(-2, -0.5)
            new_state = self.state + reward
        else:  # maintain
            reward = np.random.uniform(-0.5, 0.5)
            new_state = self.state + reward
        
        self.state = new_state
        self.actions_taken.append(action)
        self.rewards.append(reward)
        self.history.append(new_state)
        
        return new_state, reward
    
    def learn(self, reward, next_state):
        """Agent learns from experience"""
        self.decision_log.append({
            'state': self.state,
            'reward': reward,
            'next_state': next_state
        })

# ========================================
# Step 2: Create Multi-Agent Environment
# ========================================
print("Creating multi-agent environment...")

np.random.seed(42)

agents = {
    'Explorer': SimpleAIAgent('Explorer', initial_state=0),
    'Planner': SimpleAIAgent('Planner', initial_state=5),
    'Optimizer': SimpleAIAgent('Optimizer', initial_state=10),
    'Monitor': SimpleAIAgent('Monitor', initial_state=15),
}

# Simulate agent interactions
num_steps = 50

for step in range(num_steps):
    env_state = np.sin(step / 10) * 5
    
    for agent_name, agent in agents.items():
        # Perceive
        observation = agent.perceive(env_state)
        
        # Think
        action = agent.think(observation)
        
        # Act
        new_state, reward = agent.act(action, None)
        
        # Learn
        agent.learn(reward, new_state)

print("Multi-agent simulation completed!")

# ========================================
# Step 3: Visualize Agent Trajectories
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors_agents = {'Explorer': 'red', 'Planner': 'blue', 'Optimizer': 'green', 'Monitor': 'orange'}

# 1. State trajectories
for agent_name, agent in agents.items():
    axes[0, 0].plot(agent.history, label=agent_name, linewidth=2, alpha=0.7, color=colors_agents[agent_name])
axes[0, 0].set_xlabel('Time Step', fontweight='bold')
axes[0, 0].set_ylabel('Agent State', fontweight='bold')
axes[0, 0].set_title('Agent State Evolution', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Cumulative rewards
for agent_name, agent in agents.items():
    cumulative_rewards = np.cumsum(agent.rewards)
    axes[0, 1].plot(cumulative_rewards, label=agent_name, linewidth=2, alpha=0.7, color=colors_agents[agent_name])
axes[0, 1].set_xlabel('Time Step', fontweight='bold')
axes[0, 1].set_ylabel('Cumulative Reward', fontweight='bold')
axes[0, 1].set_title('Cumulative Rewards Over Time', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Action distribution
action_counts = defaultdict(lambda: defaultdict(int))
for agent_name, agent in agents.items():
    for action in agent.actions_taken:
        action_counts[agent_name][action] += 1

actions = set()
for agent_actions in action_counts.values():
    actions.update(agent_actions.keys())
actions = sorted(list(actions))

x_pos = np.arange(len(agents))
width = 0.25

for i, action in enumerate(actions):
    counts = [action_counts[agent_name].get(action, 0) for agent_name in agents.keys()]
    axes[1, 0].bar(x_pos + i*width, counts, width, label=action, alpha=0.8)

axes[1, 0].set_ylabel('Action Count', fontweight='bold')
axes[1, 0].set_title('Action Distribution by Agent', fontweight='bold')
axes[1, 0].set_xticks(x_pos + width)
axes[1, 0].set_xticklabels(agents.keys())
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Performance metrics
metrics_data = {
    'Agent': list(agents.keys()),
    'Final State': [agent.history[-1] for agent in agents.values()],
    'Total Reward': [sum(agent.rewards) for agent in agents.values()],
    'Avg Reward': [np.mean(agent.rewards) for agent in agents.values()],
    'Reward Std': [np.std(agent.rewards) for agent in agents.values()],
}

metrics_text = f"""
AGENT PERFORMANCE METRICS
{'='*50}

"""

for i, agent_name in enumerate(agents.keys()):
    agent = agents[agent_name]
    metrics_text += f"\n{agent_name}:\n"
    metrics_text += f"  Final State: {agent.history[-1]:.2f}\n"
    metrics_text += f"  Total Reward: {sum(agent.rewards):.2f}\n"
    metrics_text += f"  Avg Reward/Step: {np.mean(agent.rewards):.3f}\n"
    metrics_text += f"  Max State: {max(agent.history):.2f}\n"
    metrics_text += f"  Min State: {min(agent.history):.2f}\n"

axes[1, 1].text(0.05, 0.95, metrics_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/agent_trajectories.png', dpi=300, bbox_inches='tight')
plt.close()

print("Agent trajectories plot saved!")

# ========================================
# Step 4: Agent Decision Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Action sequences for one agent
explorer = agents['Explorer']
actions_numeric = {'explore': 2, 'maintain': 1, 'retreat': 0}
action_sequence = [actions_numeric.get(a, 1) for a in explorer.actions_taken]

axes[0, 0].bar(range(len(action_sequence)), action_sequence, color='coral', edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('Action Code', fontweight='bold')
axes[0, 0].set_xlabel('Time Step', fontweight='bold')
axes[0, 0].set_title('Explorer Agent - Action Sequence', fontweight='bold')
axes[0, 0].set_yticks([0, 1, 2])
axes[0, 0].set_yticklabels(['Retreat', 'Maintain', 'Explore'])
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Reward distribution for all agents
reward_data = [agent.rewards for agent in agents.values()]
bp = axes[0, 1].boxplot(reward_data, labels=agents.keys(), patch_artist=True)
for patch, color in zip(bp['boxes'], colors_agents.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0, 1].set_ylabel('Reward', fontweight='bold')
axes[0, 1].set_title('Reward Distribution by Agent', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. State vs Reward correlation
for agent_name, agent in agents.items():
    states = agent.history[:-1]
    rewards = agent.rewards
    axes[1, 0].scatter(states, rewards, s=50, label=agent_name, alpha=0.6, color=colors_agents[agent_name])
axes[1, 0].set_xlabel('State', fontweight='bold')
axes[1, 0].set_ylabel('Reward', fontweight='bold')
axes[1, 0].set_title('State-Reward Relationship', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Agent comparison heatmap
comparison_data = []
for agent_name, agent in agents.items():
    comparison_data.append([
        np.mean(agent.rewards),
        np.std(agent.rewards),
        len([a for a in agent.actions_taken if a == 'explore']) / len(agent.actions_taken),
        agent.history[-1]
    ])

comparison_matrix = np.array(comparison_data).T
im = axes[1, 1].imshow(comparison_matrix, cmap='RdYlGn', aspect='auto')
axes[1, 1].set_xticks(range(len(agents)))
axes[1, 1].set_xticklabels(agents.keys())
axes[1, 1].set_yticks(range(4))
axes[1, 1].set_yticklabels(['Avg Reward', 'Std Dev', 'Explore %', 'Final State'])
axes[1, 1].set_title('Agent Comparison Matrix', fontweight='bold')

# Add text annotations
for i in range(comparison_matrix.shape[0]):
    for j in range(comparison_matrix.shape[1]):
        text = axes[1, 1].text(j, i, f'{comparison_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=10, fontweight='bold')

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.savefig('outputs/agent_decision_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Agent decision analysis plot saved!")

# ========================================
# Step 5: Agent Collaboration
# ========================================
# Simulate collaborative task
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Simulated task progress with collaborative agents
task_progress = []
individual_progress = defaultdict(list)

for step in range(50):
    # Collaborative progress (agents help each other)
    collaborative = sum([agent.history[step] if step < len(agent.history) else agent.history[-1] for agent in agents.values()]) / len(agents)
    task_progress.append(collaborative)
    
    for agent_name, agent in agents.items():
        agent_state = agent.history[step] if step < len(agent.history) else agent.history[-1]
        individual_progress[agent_name].append(agent_state)

# 1. Collaborative task progress
axes[0, 0].plot(task_progress, linewidth=3, color='purple', label='Collaborative Task')
for agent_name, progress in individual_progress.items():
    axes[0, 0].plot(progress, linewidth=1, alpha=0.5, color=colors_agents[agent_name], label=agent_name)
axes[0, 0].fill_between(range(len(task_progress)), task_progress, alpha=0.2, color='purple')
axes[0, 0].set_xlabel('Time Step', fontweight='bold')
axes[0, 0].set_ylabel('Progress', fontweight='bold')
axes[0, 0].set_title('Individual vs Collaborative Progress', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Efficiency gains
efficiency_gain = []
for step in range(50):
    individual_avg = np.mean([individual_progress[agent][step] for agent in agents.keys()])
    collaborative_val = task_progress[step]
    efficiency_gain.append(collaborative_val - individual_avg)

axes[0, 1].fill_between(range(len(efficiency_gain)), efficiency_gain, alpha=0.5, color='green')
axes[0, 1].plot(efficiency_gain, linewidth=2, color='darkgreen')
axes[0, 1].set_xlabel('Time Step', fontweight='bold')
axes[0, 1].set_ylabel('Efficiency Gain', fontweight='bold')
axes[0, 1].set_title('Collaboration Efficiency Gain', fontweight='bold')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 1].grid(True, alpha=0.3)

# 3. Agent coordination matrix
coordination_matrix = np.random.uniform(0, 1, (len(agents), len(agents)))
for i in range(len(agents)):
    coordination_matrix[i, i] = 1.0

agent_list = list(agents.keys())
im = axes[1, 0].imshow(coordination_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
axes[1, 0].set_xticks(range(len(agent_list)))
axes[1, 0].set_yticks(range(len(agent_list)))
axes[1, 0].set_xticklabels(agent_list)
axes[1, 0].set_yticklabels(agent_list)
axes[1, 0].set_title('Agent Coordination Matrix', fontweight='bold')

for i in range(len(agent_list)):
    for j in range(len(agent_list)):
        text = axes[1, 0].text(j, i, f'{coordination_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if coordination_matrix[i, j] < 0.5 else "white",
                              fontsize=9, fontweight='bold')

plt.colorbar(im, ax=axes[1, 0], label='Coordination Level')

# 4. Summary statistics
summary_text = f"""
MULTI-AGENT SYSTEM SUMMARY
{'='*50}

AGENTS: {len(agents)}
STEPS: {len(task_progress)}

COLLECTIVE METRICS:
  Average Task Progress: {np.mean(task_progress):.2f}
  Max Task Progress: {max(task_progress):.2f}
  Min Task Progress: {min(task_progress):.2f}
  
  Avg Efficiency Gain: {np.mean(efficiency_gain):.3f}
  Cumulative Collaboration Benefit: {sum(efficiency_gain):.2f}

INDIVIDUAL METRICS:
"""

for agent_name, agent in agents.items():
    final_state = agent.history[-1]
    total_reward = sum(agent.rewards)
    summary_text += f"\n  {agent_name}:\n"
    summary_text += f"    Final State: {final_state:.2f}\n"
    summary_text += f"    Total Reward: {total_reward:.2f}\n"

axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/agent_collaboration.png', dpi=300, bbox_inches='tight')
plt.close()

print("Agent collaboration plot saved!")

print("\n✅ AI Agent Framework Complete!")
print("Generated outputs:")
print("  - outputs/agent_trajectories.png")
print("  - outputs/agent_decision_analysis.png")
print("  - outputs/agent_collaboration.png")
