import os
import time
import traceback
from datetime import datetime
from curriculum import CurriculumScheduler
from trainer_primal import PRIMALTrainer
from evaluator import Evaluator
from env_generator_1 import generate_env_yaml, generate_actor_yaml
import matplotlib.pyplot as plt

def setup_logging():
    log_dir = f"./logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_training_plots(metrics_history, log_dir):
    if not metrics_history:
        return
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot([m['success_rate'] for m in metrics_history], label='Success Rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot([m['collision_rate'] for m in metrics_history], label='Collision Rate')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot([m['avg_makespan'] for m in metrics_history], label='Makespan')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{log_dir}/training_progress.png")
    plt.close()

def run_curriculum_training():
    log_dir = setup_logging()
    scheduler = CurriculumScheduler()
    trainer = PRIMALTrainer()
    evaluator = Evaluator(eval_episodes=10, max_steps=100)
    metrics_history = []

    try:
        while not scheduler.is_done():
            task = scheduler.get_current_task()
            print(f"\n🟢 Stage {scheduler.current_stage} - Size: {task['dim']}, Agents: {task['num_agents']}, Obstacles: {task['obstacle_density']}")

            env_file = f"{log_dir}/stage_{scheduler.current_stage}_env.yaml"
            actor_file = f"{log_dir}/stage_{scheduler.current_stage}_actors.yaml"
            obstacles = generate_env_yaml(task['dim'], task['obstacle_density'], env_file)
            generate_actor_yaml(task['dim'], task['num_agents'], actor_file, obstacle_list=obstacles)

            try:
                trainer.train_one_stage(env_file, actor_file, scheduler.current_stage, episodes=100)
            except Exception as e:
                print(f"❌ Training failed: {e}")
                traceback.print_exc()
                scheduler.adjust_difficulty()
                continue

            # ✅ 保存模型
            model_path = f"{log_dir}/stage_{scheduler.current_stage}_model.pt"
            trainer.save_model(model_path)

            # ✅ 评估时使用同一个 policy_net 实例
            print("🧪 Evaluating...")
            metrics = evaluator.evaluate(
                policy_net=trainer.policy_net,
                env_file=env_file,
                actor_file=actor_file
            )
            metrics_history.append(metrics)

            # ✅ Debug 检查 policy 权重
            print("⚠️ Debug: First conv weight mean =", trainer.policy_net.conv[0].weight.data.mean().item())

            with open(f"{log_dir}/metrics.csv", "a") as f:
                if scheduler.current_stage == 0:
                    f.write("stage,success_rate,collision_rate,makespan\n")
                f.write(f"{scheduler.current_stage},{metrics['success_rate']},{metrics['collision_rate']},{metrics['avg_makespan']}\n")

            if evaluator.meets_criteria(metrics):
                print(f"✅ Stage passed! Success: {metrics['success_rate']:.1%}")
                scheduler.promote()
            else:
                print(f"🔁 Needs improvement. Success: {metrics['success_rate']:.1%}")
                scheduler.repeat_stage()
                if scheduler.current_attempts >= scheduler.max_attempts:
                    scheduler.adjust_difficulty()

            save_training_plots(metrics_history, log_dir)

    except KeyboardInterrupt:
        print("⚠️ Interrupted")

    trainer.save_model(f"{log_dir}/final_model.pt")
    print(f"🏁 Training complete. Logs at: {log_dir}")

if __name__ == "__main__":
    run_curriculum_training()
