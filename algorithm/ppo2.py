import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from model import agent


class PPO2:
    def __init__(self, state_dim, action_dim, hidden_dim, device, lr=0.001):
        self.device = device
        self.actor = agent.ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.old_actor = agent.ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = agent.CriticNetwork(state_dim, hidden_dim).to(self.device)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.writer = SummaryWriter()

        self.states = []
        self.actions = []
        self.rewards = []
        self.total_rewards = []
        self.values = None

    def get_action(self, state):
        action_probs = self.actor(state)
        action = Categorical(action_probs)
        return action.sample()

    def collect_data(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clean_collect_data(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def compute_advantages(self, gamma=0.99, tau=0.95):
        self.values = self.critic(torch.cat(self.states))
        deltas = [r + gamma * v_next - v for r, v, v_next in
                  zip(self.rewards, self.values, self.values[1:] + torch.FloatTensor([0]).to(self.device))]
        advantages = []
        adv = 0
        for delta in reversed(deltas):
            adv = delta + gamma * tau * adv
            advantages.append(adv)
        advantages = list(reversed(advantages))
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.to(self.device)

    def train(self, current_round, epochs=10):
        advantages = self.compute_advantages()
        for i in range(epochs):
            dataset = list(zip(self.states, self.actions, advantages))
            batch_size = len(self.actions)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            cur_epoch = current_round * epochs + i
            for batch_states, batch_actions, batch_advantages in dataloader:
                self.update_actor(cur_epoch, batch_states, batch_actions, batch_advantages)
        self.update_old_actor()
        self.update_critic(current_round)

    def update_old_actor(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def update_actor(self, current_epoch, states, actions, advantages):
        action_probs = self.old_actor(states)
        old_action_probs = torch.gather(action_probs.squeeze(), 1, actions)
        loss = self.surrogate_objective(states, actions, old_action_probs.detach(), advantages)
        self.writer.add_scalar('loss/actor', loss, current_epoch)
        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()

    def surrogate_objective(self, states, actions, old_action_probs, advantages, epsilon=0.2):
        action_probs = self.actor(states)
        new_action_probs = torch.gather(action_probs.squeeze(), 1, actions)
        ratio = torch.exp(new_action_probs - old_action_probs)
        surrogate = ratio * advantages.unsqueeze(1).to(self.device)
        clipped_surrogate = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages.unsqueeze(1)

        return -torch.min(surrogate, clipped_surrogate).mean()

    def update_critic(self, current_round):
        critics = self.critic(torch.cat(self.states))
        targets = torch.tensor(self.rewards).to(self.device) + torch.cat(
            (critics[1:], torch.tensor([[0.0]]).to(self.device)))
        criterion = nn.MSELoss()
        self.optimizer_critic.zero_grad()
        critics = critics.expand_as(targets)
        loss = criterion(critics, targets)
        self.writer.add_scalar('loss/critic', loss, current_round)
        loss.backward()
        self.optimizer_critic.step()

    def save_model(self, current_round, best=False):
        print("Saving model...")
        if best:
            # 保存最优模型状态
            torch.save(self.actor.state_dict(), f'./models/best_actor.pth')
            torch.save(self.critic.state_dict(), f'./models/best_critic.pth')
        else:
            torch.save(self.actor.state_dict(), f'./models/actor_{current_round}.pth')
            torch.save(self.critic.state_dict(), f'./models/critic_{current_round}.pth')
