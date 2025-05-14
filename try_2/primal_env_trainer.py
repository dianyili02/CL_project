import yaml
import numpy as np
from typing import List, Tuple, Dict, Any

class PRIMALEnvironment:
    def __init__(self, env_yaml_path: str, actor_yaml_path: str, obs_size: int = 10):
        self.obs_size = obs_size
        self.load_env(env_yaml_path, actor_yaml_path)
        self.total_collisions = 0
        self.total_steps = 0

    def load_env(self, env_file: str, actor_file: str) -> None:
        with open(env_file, "r") as f:
            env_data = yaml.safe_load(f)
        with open(actor_file, "r") as f:
            actor_data = yaml.safe_load(f)

        self.map_dim = env_data["map"]["dimensions"]
        self.obstacles = env_data["map"]["obstacles"]
        self.grid = np.zeros(self.map_dim, dtype=np.int32)
        for ox, oy in self.obstacles:
            self.grid[ox][oy] = 1

        self.agent_starts = [tuple(agent["start"]) for agent in actor_data["agents"]]
        self.agent_goals = [tuple(agent["goal"]) for agent in actor_data["agents"]]
        self.num_agents = len(self.agent_starts)

    def reset(self) -> List[np.ndarray]:
        self.agent_positions = list(self.agent_starts)
        self.dones = [False] * self.num_agents
        self.collisions = [False] * self.num_agents
        self.current_episode_collisions = 0

        for i in range(self.num_agents):
            if self.agent_positions[i] == self.agent_goals[i]:
                self.dones[i] = True
                print(f"[RESET] Agent {i} starts at goal {self.agent_goals[i]}")

        return [self.get_observation(i) for i in range(self.num_agents)]

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        move_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay
        }

        proposed_positions = []
        rewards = [0.0] * self.num_agents
        new_dones = self.dones.copy()
        new_collisions = [False] * self.num_agents
        step_collisions = 0

        for i, (r, c) in enumerate(self.agent_positions):
            if self.dones[i]:
                proposed_positions.append((r, c))
                continue
            dr, dc = move_map.get(actions[i], (0, 0))
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.map_dim[0] and 0 <= nc < self.map_dim[1] and self.grid[nr][nc] == 0:
                proposed_positions.append((nr, nc))
            else:
                proposed_positions.append((r, c))

        pos_count = {}
        for pos in proposed_positions:
            pos_count[pos] = pos_count.get(pos, 0) + 1

        final_positions = []
        for i, pos in enumerate(proposed_positions):
            if pos_count[pos] > 1 and not self.dones[i]:
                final_positions.append(self.agent_positions[i])
                new_collisions[i] = True
                step_collisions += 1
            else:
                final_positions.append(pos)

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (self.agent_positions[i] == final_positions[j] and 
                    self.agent_positions[j] == final_positions[i]):
                    final_positions[i] = self.agent_positions[i]
                    final_positions[j] = self.agent_positions[j]
                    new_collisions[i] = True
                    new_collisions[j] = True
                    step_collisions += 2

        current_successes = 0
        for i in range(self.num_agents):
            if self.dones[i]:
                rewards[i] = 0.0
                continue

            old_pos = self.agent_positions[i]
            new_pos = final_positions[i]
            goal_pos = self.agent_goals[i]

            reward = -0.01  # base penalty

            if new_pos == goal_pos:
                new_dones[i] = True
                reward = 1.0
                current_successes += 1
            else:
                if self._manhattan(new_pos, goal_pos) < self._manhattan(old_pos, goal_pos):
                    reward += 0.05
                if actions[i] == 4:
                    reward -= 0.05
                if new_collisions[i]:
                    reward -= 0.1

            rewards[i] = reward

        self.agent_positions = final_positions
        self.dones = new_dones
        self.collisions = new_collisions
        self.total_collisions += step_collisions
        self.total_steps += 1
        self.current_episode_collisions += step_collisions

        info = {
            "collisions": step_collisions,
            "successes": current_successes,
            "episode_collisions": self.current_episode_collisions,
            "all_dones": all(self.dones)
        }

        return [self.get_observation(i) for i in range(self.num_agents)], rewards, self.dones.copy(), info

    def get_observation(self, agent_id: int) -> np.ndarray:
        obs = np.zeros((4, self.obs_size, self.obs_size), dtype=np.float32)
        half = self.obs_size // 2
        cx, cy = self.agent_positions[agent_id]
        gx, gy = self.agent_goals[agent_id]

        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                gx_abs, gy_abs = cx + dx, cy + dy
                ix, iy = dx + half, dy + half
                if (0 <= gx_abs < self.map_dim[0] and 
                    0 <= gy_abs < self.map_dim[1] and
                    0 <= ix < self.obs_size and 
                    0 <= iy < self.obs_size):

                    obs[0, ix, iy] = self.grid[gx_abs, gy_abs]

                    if (gx_abs, gy_abs) == (gx, gy):
                        obs[2, ix, iy] = 1.0

                    for j, (ax, ay) in enumerate(self.agent_positions):
                        if (gx_abs, gy_abs) == (ax, ay):
                            if j == agent_id:
                                obs[1, ix, iy] = 1.0
                            else:
                                obs[3, ix, iy] = 1.0
        return obs

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_metrics(self) -> Dict[str, float]:
        """Returns step-based collision rate only. Use evaluator for success rate."""
        return {
            "collision_rate": self.total_collisions / (self.num_agents * self.total_steps) 
                if self.total_steps > 0 else 0
        }
