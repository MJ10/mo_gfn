import hydra
import wandb
import warnings
import random
import logging
import os
from pathlib import Path
import sys
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
from torch_seq_moo.utils import flatten_config

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_run(cfg):
    trial_id = cfg.trial_id
    if cfg.job_name is None:
        cfg.job_name = '_'.join(randomname.get_name().lower().split('-') + [str(trial_id)])
    cfg.seed = random.randint(0, 100000) if cfg.seed is None else cfg.seed
    set_seed(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)  # Resolve config interpolations
    cfg = DictConfig(cfg)
    # logger.write_hydra_yaml(cfg)

    print(OmegaConf.to_yaml(cfg))
    with open('hydra_config.txt', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path='./configs', config_name='main')
def main(config):
    random.seed(None)  # make sure random seed resets between multirun jobs for random job-name generation

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
        
    # init_run()

    wandb.init(project='torch_seq_moo', config=log_config, mode=config.wandb_mode,
               group=config.group_name, name=config.exp_name, tags=config.exp_tags)
    config['job_name'] = wandb.run.name
    config = init_run(config)  # random seed is fixed here

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # catch really annoying BioPython warnings

        try:
            logging.info("Initializing Tokenizer")
            tokenizer = hydra.utils.instantiate(config.tokenizer)
            # logger = hydra.utils.instantiate(config.logger)
            logging.info("Initializing Task")
            task = hydra.utils.instantiate(config.task, tokenizer=tokenizer)

            project_root = Path(os.getcwd()).parents[2]  # changing the Hydra run dir will break this.
            all_seqs, all_targets = task.task_setup(config, project_root=project_root)

            logging.info("Initializing Algorithm")
            algorithm = hydra.utils.instantiate(
                config.algorithm,
                task=task,
                tokenizer=tokenizer,
                cfg=config.algorithm,
                task_cfg=config.task
            )

            logging.info("Running Optimizer")
            metrics = algorithm.optimize(task, init_data=None)
            
            metrics = {key.split('/')[-1]: val for key, val in metrics.items()}  # strip prefix
            ret_val = metrics['hypervol_rel']

        except Exception as err:
            logging.exception(err)
            ret_val = float('NaN')

    wandb.finish()
    return ret_val


if __name__ == "__main__":
    main()
    sys.exit()
