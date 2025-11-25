from clearml import Task, Logger as ClearMLLogger
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional
import torch

class ExperimentLogger:
    def __init__(self, config: DictConfig):
        self.config = config
        
        self.task = Task.init(
            project_name="Yambda-RecSys",
            task_name=config.experiment_name,
            output_uri=True,
            auto_connect_frameworks={'pytorch': True} 
        )
        
        self.task.connect(OmegaConf.to_container(config, resolve=True))
        self.logger = ClearMLLogger.current_logger()

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Logs scalars (loss, recall). Expects format {'Train/Loss': 0.5}.
        """
        for name, value in metrics.items():
            if '/' in name:
                title, series = name.split('/', 1)
            else:
                title, series = "General", name
                
            self.logger.report_scalar(
                title=title, 
                series=series, 
                value=value, 
                iteration=step
            )
            
    def close(self):
        self.task.close()
