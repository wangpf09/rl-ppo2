import torch
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical
from model import agent
from torch.utils.data import DataLoader, Dataset


class PPO2:
    def __init__(self, state_dim, action_dim, hidden_dim, lr=0.001):
        self.actor = agent.ActorNetwork(state_dim, action_dim, hidden_dim)
        self.old_actor = agent.ActorNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = agent.CriticNetwork(state_dim, hidden_dim)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.all_rewards = []
        self.values = None

    def get_action(self, state):
        action_probs = self.old_actor(state)
        action = Categorical(action_probs)
        return action.sample()

    def collect_data(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.all_rewards.append(reward)

    def clean_collect_data(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def compute_advantages(self, gamma=0.99, tau=0.95):
        self.values = self.critic(torch.cat(self.states))
        deltas = [r + gamma * v_next - v for r, v, v_next in
                  zip(self.rewards, self.values, self.values[1:] + torch.FloatTensor([0]))]
        advantages = []
        adv = 0
        for delta in reversed(deltas):
            adv = delta + gamma * tau * adv
            advantages.append(adv)
        advantages = list(reversed(advantages))
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def train(self, epochs=10):
        advantages = self.compute_advantages()
        for _ in range(epochs):
            dataset = list(zip(self.states, self.actions, advantages))
            batch_size = len(self.actions)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch_states, batch_actions, batch_advantages in dataloader:
                self.update_actor(batch_states, batch_actions, batch_advantages)
        self.update_old_actor()
        self.update_critic()

    def update_old_actor(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def update_actor(self, states, actions, advantages):
        action_probs = self.old_actor(states)
        old_action_probs = torch.gather(action_probs.squeeze(), 1, actions)
        loss = self.surrogate_objective(states, actions, old_action_probs.detach(), advantages)

        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()

    def surrogate_objective(self, states, actions, old_action_probs, advantages, epsilon=0.2):
        action_probs = self.actor(states)
        new_action_probs = torch.gather(action_probs.squeeze(), 1, actions)
        ratio = torch.exp(new_action_probs - old_action_probs)
        surrogate = ratio * advantages.unsqueeze(1)
        clipped_surrogate = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages.unsqueeze(1)

        return -torch.min(surrogate, clipped_surrogate).mean()

    def update_critic(self):
        critics = self.critic(torch.cat(self.states))
        targets = torch.tensor(self.rewards) + torch.cat((critics[1:], torch.tensor([[0.0]])))
        criterion = nn.MSELoss()
        self.optimizer_critic.zero_grad()
        critics = critics.expand_as(targets)
        loss = criterion(critics, targets)
        loss.backward()
        self.optimizer_critic.step()
