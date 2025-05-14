import torch
import torch.nn as nn
from model_1 import PolicyNetwork
from primal_env_trainer import PRIMALEnvironment

class PRIMALTrainer:
    def __init__(self, lr=1e-3):
        self.policy_net = PolicyNetwork()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def train_one_stage(self, env_yaml, actor_yaml, stage, episodes=300, max_steps=200):
        env = PRIMALEnvironment(env_yaml, actor_yaml)

        print("\n🔧 Initial FC1 weight mean:", self.policy_net.fc[1].weight.data.mean().item())

        for ep in range(episodes):
            obs_list = env.reset()
            dones = [False] * env.num_agents
            total_reward = 0

            for step in range(max_steps):
                obs_tensor = torch.stack([
                    torch.tensor(obs, dtype=torch.float32) for obs in obs_list
                ])

                logits = self.policy_net(obs_tensor)             # [N, 5]
                actions = torch.argmax(logits, dim=1)            # [N]
                log_probs = torch.log_softmax(logits, dim=1)     # [N, 5]
                selected_log_probs = log_probs[range(env.num_agents), actions]  # [N]

                obs_list, rewards, dones, _ = env.step(actions.tolist())
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)     # [N]

                # Policy gradient style loss (REINFORCE-like)
                loss = -(selected_log_probs * rewards_tensor).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_reward += sum(rewards)

                if all(dones):
                    break

            print(f"[EP {ep}] Avg Total Reward: {total_reward / env.num_agents:.2f}, Steps: {step + 1}")
            print(f"[DEBUG TRAIN] env class = {type(env).__name__}, dones = {dones}")

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
