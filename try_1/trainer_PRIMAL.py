import torch
from model_1 import PolicyNetwork
from primal_training_loop_1 import train_primal_on_env  # 自定义PRIMAL训练入口

class PRIMALTrainer:
    def __init__(self):
        self.policy_net = PolicyNetwork()

    def train_one_stage(self, env_yaml, actor_yaml, stage):
        print(f"Training on Stage {stage}...")
        train_primal_on_env(env_yaml, actor_yaml, self.policy_net, episodes=300)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
