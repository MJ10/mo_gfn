import wandb


class BaseAlgorithm():
    def __init__(self, cfg, tokenizer, task_cfg, **kwargs):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.task_cfg = task_cfg

    def optimize(self, task, initial_data=None):
        raise NotImplementedError("Override this method in your class")
    
    def log(self, metrics, commit=True):
        wandb.log(metrics, commit=True)
    
