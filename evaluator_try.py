def evaluate(self, policy_net, env_file, actor_file):
    print("✅ Entering evaluator.evaluate()")

    success_count = 0
    collision_count = 0
    makespans = []

    for ep in range(self.eval_episodes):
        env = PRIMALEnvironment(env_file, actor_file)
        obs_list = env.reset()
        num_agents = env.num_agents
        prev_positions = env.agent_positions.copy()

        for step in range(self.max_steps):
            obs_tensor = torch.stack([
                torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
                for obs in obs_list
            ])
            logits = policy_net(obs_tensor)
            actions = torch.argmax(logits, dim=1).tolist()

            obs_list, _, dones, _ = env.step(actions)

            # === 碰撞检测 ===
            current_positions = env.agent_positions
            position_set = set()
            edge_set = set()
            collision_detected = False

            for i, pos in enumerate(current_positions):
                if pos in position_set:
                    collision_detected = True
                position_set.add(pos)

                # Edge collision detection
                edge = (prev_positions[i], pos)
                if edge[::-1] in edge_set:
                    collision_detected = True
                edge_set.add(edge)

            if collision_detected:
                collision_count += 1
                print(f"[⚠️ COLLISION] Episode {ep}, Step {step}")

            prev_positions = current_positions.copy()

            if all(dones):
                success_count += 1
                makespans.append(step + 1)
                break

    print(f"✅ Evaluation done.")
    print(f"⭐️ Success rate: {success_count / self.eval_episodes:.2f}")
    print(f"💥 Collision rate: {collision_count / (self.eval_episodes * self.max_steps):.2f}")
    if makespans:
        print(f"⏱ Avg Makespan: {sum(makespans) / len(makespans):.2f}")
