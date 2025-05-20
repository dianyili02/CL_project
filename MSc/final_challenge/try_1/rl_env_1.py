
import yaml
import numpy as np

class PRIMALEnvironment:
    def __init__(self, env_yaml_path, actor_yaml_path, obs_size=10):
        self.obs_size = obs_size
        self.load_env(env_yaml_path, actor_yaml_path)

    def load_env(self, env_file, actor_file):
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
        self.agent_positions = list(self.agent_starts)
        self.dones = [False] * self.num_agents
        self.collisions = [False] * self.num_agents

    def reset(self):
        self.agent_positions = list(self.agent_starts)
        self.dones = [False] * self.num_agents
        self.collisions = [False] * self.num_agents

        for i in range(self.num_agents):
            if self.agent_positions[i] == self.agent_goals[i]:
                self.dones[i] = True
        return [self.get_observation(i) for i in range(self.num_agents)]

    def step(self, actions):
        move_map = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)
        }

        assert len(actions) == self.num_agents
        proposed_positions = []
        rewards = [0.0] * self.num_agents
        self.collisions[:] = [False] * self.num_agents
        new_dones = self.dones.copy()
        for i, (r, c) in enumerate(self.agent_positions):
            if self.dones[i]:
                proposed_positions.append((r, c))
                continue
            dr, dc = move_map.get(actions[i], (0, 0))
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.map_dim[0] and 0 <= nc < self.map_dim[1] and self.grid[nr, nc] == 0:
                proposed_positions.append((nr, nc))
            else:
                proposed_positions.append((r, c))

        pos_count = {}
        for pos in proposed_positions:
            pos_count[pos] = pos_count.get(pos, 0) + 1

        final_positions = []
        for i, pos in enumerate(proposed_positions):
            if pos_count[pos] == 1:
                final_positions.append(pos)
            else:
                final_positions.append(self.agent_positions[i])
                self.collisions[i] = True

        for i in range(self.num_agents):
            if self.dones[i]:
                continue

            old_pos = self.agent_positions[i]
            new_pos = final_positions[i]
            goal_pos = self.agent_goals[i]

            if new_pos == goal_pos:
                if not self.dones[i]:
                    print(f"[DEBUG] Agent {i} reached goal at {new_pos}")
                reward = 1.0
                self.dones[i] = True  # ✅ 修复点
            else:
                reward = -0.01
                old_dist = self._manhattan(old_pos, goal_pos)
                new_dist = self._manhattan(new_pos, goal_pos)
                if new_dist < old_dist:
                    reward += 0.05
                if actions[i] == 4:
                    reward -= 0.05

            rewards[i] = reward

        self.agent_positions = final_positions
        # self.dones = new_dones  # ✅ VERY IMPORTANT

        observations = [self.get_observation(i) for i in range(self.num_agents)]
        return observations, rewards, self.dones.copy(), {}

    def get_observation(self, agent_id):
        obs = np.zeros((4, self.obs_size, self.obs_size), dtype=np.float32)
        half = self.obs_size // 2
        cx, cy = self.agent_positions[agent_id]
        gx, gy = self.agent_goals[agent_id]

        for dx in range(-half, half):
            for dy in range(-half, half):
                gx_abs, gy_abs = cx + dx, cy + dy
                ix, iy = dx + half, dy + half
                if 0 <= gx_abs < self.map_dim[0] and 0 <= gy_abs < self.map_dim[1]:
                    obs[0, ix, iy] = self.grid[gx_abs, gy_abs]
                    if (gx_abs, gy_abs) == (gx, gy):
                        obs[2, ix, iy] = 1.0
                    for j, (ax, ay) in enumerate(self.agent_positions):
                        if (gx_abs, gy_abs) == (ax, ay):
                            obs[1 if j == agent_id else 3, ix, iy] = 1.0
        return obs

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
