import torch
import torch.nn as nn
import torch.optim as optim
from cart import Cart, Physics
from torch.distributions import Categorical
from torch.nn import functional as F

# Define a simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Define the PPO algorithm
class PPO:
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99, epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01):
        self.policy = PolicyNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train_step(self, states, actions, old_probs, rewards):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)
        returns = self.compute_returns(rewards)

        # Compute advantages
        with torch.no_grad():
            values = self.policy(states)
            advantages = returns - values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # PPO loss function
        logits = self.policy(states)
        probs = F.softmax(logits, dim=1)
        new_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = new_probs / old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss function
        value_loss = F.smooth_l1_loss(values, returns.unsqueeze(1))

        # Entropy regularization
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        # Total loss
        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Environment setup
env = Cart(Physics()) # TODO
state_size = env.observation_space
action_size = env.action_space

# PPO setup
ppo = PPO(state_size, action_size)

# Training loop
num_episodes = 1000
max_steps = 500

for episode in range(num_episodes):
    state = env.reset()
    states, actions, old_probs, rewards = [], [], [], []
    total_reward = 0

    for step in range(max_steps):
        # Collect data
        states.append(state)
        action_probs = ppo.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        actions.append(action.item())
        old_probs.append(action_dist.log_prob(action))
        
        state, reward, done = env.step(action.item())
        rewards.append(reward)

        if done:
            break

    # Train the policy network
    ppo.train_step(states, actions, old_probs, rewards)

    # Print the total reward for the episode
    total_reward = sum(rewards)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Evaluate the trained policy
state = env.reset()
for _ in range(200):
    env.render()
    action_probs = ppo.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    action = torch.argmax(action_probs).item()
    state, r, done  = env.step(action)
    if done:
        print("reward: ", r)
        break

env.close()