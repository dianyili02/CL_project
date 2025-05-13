import torch
from rl_env import PRIMALEnvironment

class Evaluator:
    def __init__(self, eval_episodes=20, max_steps=200):
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps

    def evaluate(self, policy_net, env_file, actor_file):
        print("✅ Entering evaluator.evaluate()")
    
        success_count = 0
        collision_count = 0
        makespans = []

        for ep in range(self.eval_episodes):
            env = PRIMALEnvironment(env_file, actor_file)
            obs_list = env.reset()
            print(f"[DEBUG] obs[0] shape: {obs_list[0].shape}")

            num_agents = env.num_agents
            prev_positions = env.agent_positions.copy()

            for step in range(self.max_steps):
                obs_tensor = torch.stack([
                    torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW
                    for obs in obs_list
])

                logits = policy_net(obs_tensor)
                actions = torch.argmax(logits, dim=1).tolist()

                obs_list, _, dones, _ = env.step(actions)
                curr_positions = env.agent_positions.copy()

                # --- Collision: same cell ---
                if len(set(curr_positions)) < len(curr_positions):
                    collision_count += 1
                    break

                # --- Collision: edge swap ---
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        if prev_positions[i] == curr_positions[j] and prev_positions[j] == curr_positions[i]:
                            collision_count += 1
                            break

                prev_positions = curr_positions.copy()

                if all(dones):  # or use: if sum(dones) == num_agents:
                    success_count += 1
                    makespans.append(step + 1)
                    break
            else:
                makespans.append(self.max_steps)  # timeout

        metrics = {
            "success_rate": success_count / self.eval_episodes,
            "collision_rate": collision_count / self.eval_episodes,
            "avg_makespan": sum(makespans) / len(makespans)
        }

        print(f"[EVAL] Success rate: {metrics['success_rate']:.2f}, "
              f"Collision rate: {metrics['collision_rate']:.2f}, "
              f"Makespan: {metrics['avg_makespan']:.1f}")
        return metrics

    def meets_criteria(self, metrics):
        return metrics["success_rate"] >= 0.5 and metrics["collision_rate"] <= 0.1

