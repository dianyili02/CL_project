import random
import yaml

def generate_env_yaml(dim, obstacle_density, filename):
    w, h = dim
    total = w * h
    num_obstacles = int(obstacle_density * total)
    cells = [(x, y) for x in range(w) for y in range(h)]
    obstacles = random.sample(cells, num_obstacles)

    with open(filename, "w") as f:
        yaml.dump({"map": {"dimensions": dim, "obstacles": [list(o) for o in obstacles]}}, f)
    
    return obstacles  # ✅ 返回给 generate_actor_yaml 使用


def generate_actor_yaml(dim, num_agents, filename, obstacle_list=None):
    width, height = dim
    all_cells = [(x, y) for x in range(width) for y in range(height)]

    if obstacle_list is None:
        obstacle_list = []

    # Remove obstacles from candidate positions
    candidate_cells = [cell for cell in all_cells if cell not in obstacle_list]

    if len(candidate_cells) < 2 * num_agents:
        raise ValueError("Not enough free cells to assign unique starts and goals.")

    # Keep generating until starts and goals are valid and distinct
    valid = False
    while not valid:
        starts = random.sample(candidate_cells, num_agents)
        remaining_cells = [c for c in candidate_cells if c not in starts]
        goals = random.sample(remaining_cells, num_agents)

        # Make sure no agent's start == goal
        valid = all(start != goal for start, goal in zip(starts, goals))

    agents = []
    for i in range(num_agents):
        agents.append({
            "name": f"agent{i}",
            "start": list(starts[i]),
            "goal": list(goals[i])
        })

    with open(filename, "w") as f:
        yaml.dump({"agents": agents}, f)