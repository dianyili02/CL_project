# import torch
# from rl_env_1 import PRIMALEnvironment

# def tuple_eq(a, b):
#     return tuple(a) == tuple(b)

# class Evaluator:
#     def __init__(self, eval_episodes=20, max_steps=200):
#         self.eval_episodes = eval_episodes
#         self.max_steps = max_steps

#     def evaluate(self, policy_net, env_file, actor_file):
#         print("✅ Entering evaluator.evaluate()")

#         success_count = 0
#         collision_count = 0
#         makespans = []

#         for ep in range(self.eval_episodes):
#             env = PRIMALEnvironment(env_file, actor_file)
#             obs_list = env.reset()
#             num_agents = env.num_agents
#             prev_positions = env.agent_positions.copy()

#             for step in range(self.max_steps):
#                 obs_tensor = torch.stack([
#     torch.tensor(obs if obs.shape[0] == 4 else obs.transpose(2, 0, 1), dtype=torch.float32)
#     for obs in obs_list
# ])
#                 if 'logits' in locals():
#                     print(f"[Eval EP {ep}, Step {step}] logits: {logits.detach().cpu().numpy()}")

#                 for i in range(num_agents):
#                     print(f"[Eval EP {ep}] Agent {i} pos: {env.agent_positions[i]}, goal: {env.agent_goals[i]}, equal?: {tuple(env.agent_positions[i]) == tuple(env.agent_goals[i])}")





#                 logits = policy_net(obs_tensor)
#                 actions = torch.argmax(logits, dim=1).tolist()

#                 # print(f"[Eval EP {ep}, Step {step}] logits: {logits.detach().cpu().numpy()}")
#                 # print(f"[Eval EP {ep}, Step {step}] Actions taken: {actions}")

#                 obs_list, _, dones, _ = env.step(actions)

#                 for i in range(env.num_agents):
#                     pos = env.agent_positions[i]
#                     goal = env.agent_goals[i]
#                     print(f"[Eval EP {ep}] Agent {i} pos: {pos}, goal: {goal}, equal?: {pos == goal}")

#                 # === 碰撞检测 ===
#                 current_positions = env.agent_positions
#                 position_set = set()
#                 edge_set = set()
#                 collision_detected = False

#                 for i, pos in enumerate(current_positions):
#                     if pos in position_set:
#                         collision_detected = True
#                     position_set.add(pos)

#                     edge = (prev_positions[i], pos)
#                     if edge[::-1] in edge_set:
#                         collision_detected = True
#                     edge_set.add(edge)

#                 if collision_detected:
#                     collision_count += 1
#                     print(f"[⚠️ COLLISION] Episode {ep}, Step {step}")

#                 prev_positions = current_positions.copy()

#                 # === 成功条件：所有 agent 到达目标 ===
#                 all_reached = all(
#                     tuple_eq(env.agent_positions[i], env.agent_goals[i])
#                     for i in range(num_agents)
#                 )

#                 if all(tuple(map(int, env.agent_positions[i])) == tuple(map(int, env.agent_goals[i])) for i in range(env.num_agents)):
#                     print(f"✅ Eval EP {ep}: All agents reached goal at step {step}")
#                     success_count += 1
#             break

#         print("✅ Evaluation done.")
#         print(f"⭐️ Success rate: {success_count / self.eval_episodes:.2f}")
#         print(f"💥 Collision rate: {collision_count / (self.eval_episodes * self.max_steps):.2f}")
#         if makespans:
#             print(f"⏱ Avg Makespan: {sum(makespans) / len(makespans):.2f}")

#         return {
#             "success_rate": success_count / self.eval_episodes,
#             "collision_rate": collision_count / (self.eval_episodes * self.max_steps),
#             "avg_makespan": sum(makespans) / len(makespans) if makespans else None
#         }

#     def meets_criteria(self, metrics):
#         return metrics["success_rate"] >= 0.8 and metrics["collision_rate"] <= 0.1

import torch
from rl_env_1 import PRIMALEnvironment
from model_1 import PolicyNetwork

def tuple_eq(a, b):
    return tuple(a) == tuple(b)

class Evaluator:
    def __init__(self, eval_episodes=20, max_steps=200):
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.policy_net = PolicyNetwork()
    def evaluate(self, policy_net, env_file, actor_file):
        print("✅ Entering evaluator.evaluate()")

        success_count = 0
        collision_count = 0
        makespans = []

        for ep in range(self.eval_episodes):
            env = PRIMALEnvironment(env_file, actor_file)
            obs_list = env.reset()
            dones = [False] * env.num_agents
            step = 0

            while not all(dones) and step < self.max_steps:
                obs_tensor = torch.stack([
                    torch.tensor(o if o.shape[0] == 4 else o.transpose(2, 0, 1), dtype=torch.float32)
                    for o in obs_list
                ])

                logits = policy_net(obs_tensor)
                dists = torch.distributions.Categorical(logits=logits)
                actions = dists.sample()

                obs_list, _, dones, _ = env.step(actions.tolist())

                for i in range(env.num_agents):
                    pos = tuple(map(int, env.agent_positions[i]))
                    goal = tuple(map(int, env.agent_goals[i]))
                    print(f"[Eval EP {ep}] Agent {i} pos: {pos}, goal: {goal}, equal?: {pos == goal}")

                if all(
                    tuple(map(int, env.agent_positions[i])) == tuple(map(int, env.agent_goals[i]))
                    for i in range(env.num_agents)
                ):
                    print(f"✅ Eval EP {ep}: All agents reached goal at step {step}")
                    success_count += 1
                    break

                step += 1

        print("✅ Evaluation done.")
        print(f"⭐️ Success rate: {success_count / self.eval_episodes:.2f}")
        print(f"💥 Collision rate: {collision_count / (self.eval_episodes * self.max_steps):.2f}")
        if makespans:
            print(f"⏱ Avg Makespan: {sum(makespans) / len(makespans):.2f}")

        return {
            "success_rate": success_count / self.eval_episodes,
            "collision_rate": collision_count / (self.eval_episodes * self.max_steps),
            "avg_makespan": sum(makespans) / len(makespans) if makespans else None
        }

    def meets_criteria(self, metrics):
        return metrics["success_rate"] >= 0.8 and metrics["collision_rate"] <= 0.1