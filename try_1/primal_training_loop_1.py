import torch
import torch.nn as nn
import torch.optim as optim
from rl_env import PRIMALEnvironment
from model_1 import PolicyNetwork
from evaluator import Evaluator
def train_primal_on_env(env_yaml, actor_yaml, policy_net, episodes=300):
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    gamma = 0.99

    for episode in range(episodes):
        env = PRIMALEnvironment(env_yaml, actor_yaml)
        obs_list = env.reset()
        num_agents = env.num_agents

        log_probs = [[] for _ in range(num_agents)]
        rewards = [[] for _ in range(num_agents)]
        dones = [False] * num_agents

        max_steps = 200
        step_count = 0

        while not all(dones) and step_count < max_steps:
            obs_tensor = torch.stack([
                torch.tensor(o, dtype=torch.float32) for o in obs_list
            ])  # [N, C, H, W]

            logits = policy_net(obs_tensor)  # [N, num_actions]
            dists = torch.distributions.Categorical(logits=logits)
            actions = dists.sample()  # [N]

            prev_dones = dones.copy()
            obs_list, reward_list, dones, _ = env.step(actions.tolist())
            # print(f"[DEBUG TRAIN] Agent0 pos={env.agent_positions[0]}, goal={env.agent_goals[0]}")

            for i in range(num_agents):
                if not prev_dones[i]:
                    log_probs[i].append(dists.log_prob(actions)[i])
                    rewards[i].append(torch.tensor([reward_list[i]], dtype=torch.float32))
                if reward_list[i] >= 1.0 and not prev_dones[i]:
                    print(f"[EP {episode}] ✅ AGENT-{i} SUCCESS at step {step_count+1}, pos: {env.agent_positions[i]}")

            step_count += 1

        # 更新策略网络
        losses = []
        for i in range(num_agents):
            if not log_probs[i]:
                continue
            R = torch.tensor([0.0])
            returns = []
            for r in reversed(rewards[i]):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.cat(returns).detach()
            if returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-6)
            else:
                returns = returns - returns.mean()

            lp = torch.stack(log_probs[i])
            losses.append(-(lp * returns).mean())

        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        total_rewards = [sum([r.item() for r in rewards[i]]) for i in range(num_agents)]
        avg_total = sum(total_rewards) / num_agents
        print(f"[EP {episode}] Avg Total Reward: {avg_total:.2f}, Steps: {step_count}")

        if episode % 10 == 0:
            print(f"[EP {episode}] Rewards: {[f'{r:.2f}' for r in total_rewards]}")

