class CurriculumScheduler:
    def __init__(self):
        # 原始课程阶段
        self.original_stages = [
            {"dim": [6, 6], "num_agents": 2, "obstacle_density": 0.1},
            {"dim": [8, 8], "num_agents": 3, "obstacle_density": 0.15},
            {"dim": [10, 10], "num_agents": 4, "obstacle_density": 0.2},
            {"dim": [12, 12], "num_agents": 5, "obstacle_density": 0.25},
            {"dim": [16, 16], "num_agents": 6, "obstacle_density": 0.3}
        ]
        self.current_stage = 0
        self.current_attempts = 0
        self.max_attempts = 3  # 最大尝试次数
        self.stages = self.original_stages.copy()  # 工作副本
        
    def get_current_task(self):
        return self.stages[self.current_stage]
    
    def promote(self):
        """推进到下一个课程阶段"""
        self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)
        self.current_attempts = 0
        print(f"Promoted to stage {self.current_stage}")
        
    def repeat_stage(self):
        """重复当前阶段"""
        self.current_attempts += 1
        print(f"Repeating stage {self.current_stage} (attempt {self.current_attempts})")
        
    def adjust_difficulty(self):
        """调整当前阶段难度"""
        if self.current_attempts >= self.max_attempts:
            # 降低难度：减少智能体数量或障碍密度
            current_task = self.stages[self.current_stage]
            current_task["num_agents"] = max(2, current_task["num_agents"] - 1)
            current_task["obstacle_density"] = max(0.05, current_task["obstacle_density"] - 0.05)
            print(f"Adjusted difficulty: agents={current_task['num_agents']}, obstacles={current_task['obstacle_density']}")
            self.current_attempts = 0
    
    def is_done(self):
        """是否完成了所有课程阶段"""
        return self.current_stage >= len(self.stages) - 1
    
    def reset_stages(self):
        """重置为原始课程设置"""
        self.stages = self.original_stages.copy()
        print("Reset to original curriculum")