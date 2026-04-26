import numpy as np

class ContinualMetrics:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.acc_matrix = np.zeros((num_tasks, num_tasks))
        
    def update(self, task_id, accs):
        for t, a in enumerate(accs):
            self.acc_matrix[task_id][t] = a
            
    def average_accuracy(self):
        return np.mean([self.acc_matrix[i][i] for i in range(self.num_tasks)])
    
    def forgetting_measure(self):
        max_past = np.max(self.acc_matrix, axis=0)
        return np.mean(max_past - self.acc_matrix[-1])