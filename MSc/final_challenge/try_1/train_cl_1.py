# from curriculum import CurriculumScheduler
# from trainer_PRIMAL import PRIMALTrainer
# from evaluator import Evaluator
# from env_generator_1 import generate_env_yaml, generate_actor_yaml
# import random 
# import yaml


# def run_curriculum_training():
#     scheduler = CurriculumScheduler()
#     trainer = PRIMALTrainer()
#     evaluator = Evaluator()
#     while not scheduler.is_done():
#         task = scheduler.get_current_task()
#         print(f"\n🟢 Stage {scheduler.current_stage} - Env Size: {task['dim']}, Agents: {task['num_agents']}")
#         obstacles = generate_env_yaml(task['dim'], task['obstacle_density'], "./env.yaml")
#         generate_actor_yaml(task['dim'], task['num_agents'], "./actors.yaml", obstacle_list=obstacles)

#         # generate_env_yaml(task['dim'], task['obstacle_density'], "./env.yaml")
#         # generate_actor_yaml(task['dim'], task['num_agents'], "./actors.yaml")
#         # 在调用 evaluate 前
#         print("🔍 调用 evaluator.evaluate() 前检查 policy_net 是否存在...")
#         print("policy_net is None?", trainer.policy_net is None)

#         # metrics = evaluator.evaluate(policy_net=trainer.policy_net, env_file="./env.yaml", actor_file="./actors.yaml")

#         trainer.train_one_stage(env_yaml="./env.yaml", actor_yaml="./actors.yaml", stage=scheduler.current_stage)
#         metrics = evaluator.evaluate(policy_net=trainer.policy_net, env_file="./env.yaml", actor_file="./actors.yaml")

#         if evaluator.meets_criteria(metrics):
#             print(f"✅ Stage {scheduler.current_stage} passed. Promoting to next level.")
#             scheduler.promote()
#         else:
#             print(f"🔁 Stage {scheduler.current_stage} not passed. Re-training...")

#     trainer.save_model("/home/dianyili/Desktop/MSc/final_challenge/models/final_policy.pt")
#     print("🏁 Curriculum training complete.")

# if __name__ == "__main__":
#     run_curriculum_training()
from curriculum import CurriculumScheduler
from trainer_PRIMAL import PRIMALTrainer
from evaluator import Evaluator
from env_generator_1 import generate_env_yaml, generate_actor_yaml
import os
import torch


def run_curriculum_training():
    scheduler = CurriculumScheduler()
    trainer = PRIMALTrainer()
    evaluator = Evaluator()

    print("🚀 Starting Curriculum Training...\n")

    while not scheduler.is_done():
        task = scheduler.get_current_task()
        stage = scheduler.current_stage
        env_file = "./env.yaml"
        actor_file = "./actors.yaml"

        print(f"\n🟢 Stage {stage} - Env Size: {task['dim']}, Agents: {task['num_agents']}")
        
        # === Step 1: 生成环境和agent配置 ===
        obstacles = generate_env_yaml(task['dim'], task['obstacle_density'], env_file)
        generate_actor_yaml(task['dim'], task['num_agents'], actor_file, obstacle_list=obstacles)

        # === Step 2: 训练 ===
        print("🎯 Training on generated environment...")
        trainer.train_one_stage(env_yaml=env_file, actor_yaml=actor_file, stage=stage)

        # === Step 3: 评估 ===
        print("🧪 Evaluating trained policy...")
        metrics = evaluator.evaluate(policy_net=trainer.policy_net, env_file=env_file, actor_file=actor_file)

        # === Step 4: 根据评估结果决定是否晋级 ===
        if evaluator.meets_criteria(metrics):
            print(f"✅ Stage {stage} passed! Promoting to next stage.")
            scheduler.promote()
        else:
            print(f"🔁 Stage {stage} not passed. Re-training on the same stage...")

        # # === 可选：保存阶段性模型（方便调试） ===
        # torch.save(trainer.policy_net.state_dict(), f"./models/policy_stage_{stage}.pt")

    # === 最终策略保存 ===
    trainer.save_model("/home/dianyili/Desktop/MSc/final_challenge/models/final_policy.pt")
    print("🏁 Curriculum training complete. Final model saved.")


if __name__ == "__main__":
    run_curriculum_training()


