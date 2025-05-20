
class CurriculumScheduler:
    def __init__(self):
        self.stages = [
            {'dim': [6, 6], 'num_agents': 1, 'obstacle_density': 0.0},
            {'dim': [8, 8], 'num_agents': 3, 'obstacle_density': 0.15},
            {'dim': [10, 10], 'num_agents': 4, 'obstacle_density': 0.2},
            {'dim': [12, 12], 'num_agents': 5, 'obstacle_density': 0.25}
        ]
        self.current_stage = 0

    def get_current_task(self):
        return self.stages[self.current_stage]

    def promote(self):
        self.current_stage += 1
    def is_done(self):
        return self.current_stage >= len(self.stages)
