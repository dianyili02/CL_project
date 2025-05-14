import torch
from tqdm import tqdm
from primal_env_trainer import PRIMALEnvironment

class Evaluator:
    def __init__(self, eval_episodes=20, max_steps=200):
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps

    def evaluate(self, policy_net, env_file, actor_file, verbose=False):
        print(f"🚀 Starting evaluation with {self.eval_episodes} episodes...")
        
        # ✅ 打印网络权重均值，确认不是 untrained 网络
        print("🔍 PolicyNetwork weight means:")
        print("  Conv1 mean:", policy_net.conv[0].weight.data.mean().item())
        print("  FC1 mean  :", policy_net.fc[1].weight.data.mean().item())

        total_successes = 0
        total_collisions = 0
        total_steps = 0
        makespans = []

        for ep in tqdm(range(self.eval_episodes), desc="Evaluating"):
            env = PRIMALEnvironment(env_file, actor_file)
            obs_list = env.reset()
            num_agents = env.num_agents
            prev_positions = env.agent_positions.copy()
            
            ep_success = False
            ep_collisions = 0

            for step in range(self.max_steps):
                with torch.no_grad():
                    obs_tensor = torch.stack([
                        torch.tensor(obs, dtype=torch.float32) for obs in obs_list
                    ])
                    logits = policy_net(obs_tensor)
                    actions = torch.argmax(logits, dim=1).tolist()

                obs_list, _, _, _ = env.step(actions)
                dones = env.dones  # ✅ 使用环境内部状态

                if all(dones):
                    ep_success = True
                    makespans.append(step + 1)
                    break

                # Same-cell collision
                curr_positions = env.agent_positions
                if len(set(curr_positions)) < len(curr_positions):
                    ep_collisions += 1

                # Swap collision
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        if (prev_positions[i] == curr_positions[j] and
                            prev_positions[j] == curr_positions[i]):
                            ep_collisions += 1

                prev_positions = curr_positions.copy()

            else:
                makespans.append(self.max_steps)

            total_successes += int(ep_success)
            total_collisions += ep_collisions
            total_steps += min(step + 1, self.max_steps)

        metrics = {
            "success_rate": total_successes / self.eval_episodes,
            "collision_rate": total_collisions / (num_agents * total_steps) if total_steps > 0 else 0,
            "avg_makespan": sum(makespans) / len(makespans) if makespans else self.max_steps,
            "avg_collisions_per_episode": total_collisions / self.eval_episodes
        }

        print("\n📊 Evaluation Results:")
        print(f"✅ Success rate: {metrics['success_rate']:.2%} ({total_successes}/{self.eval_episodes})")
        print(f"❌ Collision rate: {metrics['collision_rate']:.4f} per agent-step")
        print(f"🔄 Avg collisions per episode: {metrics['avg_collisions_per_episode']:.2f}")
        print(f"⏱️ Avg makespan: {metrics['avg_makespan']:.1f} steps")
        
        return metrics

    def meets_criteria(self, metrics):
        return metrics["success_rate"] >= 0.5 and metrics["collision_rate"] <= 0.1
