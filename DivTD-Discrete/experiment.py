import os
import sys
import copy
import time
import json
import torch
import numpy as np
import pandas as pd

import agents
from utils.helper import *


class Experiment(object):
  '''
  Train the agent to play the game.
  '''
  def __init__(self, cfg):
    self.cfg = copy.deepcopy(cfg)
    if torch.cuda.is_available() and 'cuda' in cfg['device']:
      self.device = cfg['device']
    else:
      self.cfg['device'] = 'cpu'
      self.device = 'cpu'
    self.config_idx = cfg['config_idx']
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    if self.cfg['generate_random_seed']:
      self.cfg['seed'] = np.random.randint(int(1e6))
    self.model_path = self.cfg['model_path']
    self.cfg_path = self.cfg['cfg_path']
    self.save_config()

  def run(self):
    '''
    Run the game for multiple times
    '''
    set_one_thread()
    self.start_time = time.time()
    set_random_seed(self.cfg['seed'])
    self.agent = getattr(agents, self.agent_name)(self.cfg)
    self.agent.env['Train'].seed(self.cfg['seed'])
    self.agent.env['Train'].action_space.np_random.seed(self.cfg['seed'])
    self.agent.env['Test'].seed(self.cfg['seed'])
    self.agent.env['Test'].action_space.np_random.seed(self.cfg['seed'])
    # Train && Test
    self.agent.run_steps(render=self.cfg['render'])
    # Save model
    # self.save_model()
    self.end_time = time.time()
    self.agent.logger.info(f'Memory usage: {rss_memory_usage():.2f} MB')
    self.agent.logger.info(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')
  
  def save_eval_result(self):
    '''
    Print and save the evaluation (验证) results collected during training.
    与 main_atari.py 中 `np.save(result_dir + file_name, evaluations)` 对应:
      * evaluations.npy : 每次验证的平均回报
      * evaluations.csv : 每次验证的详细统计
    训练过程中每次验证都会调用 agent.save_eval_result() 写盘, 这里再打印一次汇总信息。
    '''
    agent = getattr(self, 'agent', None)
    evaluations = getattr(agent, 'evaluations', None)
    if not evaluations:
      return None
    # Save results to files (same format as main_atari.py)
    eval_log_path = self.cfg.get('eval_log_path', None)
    if eval_log_path:
      np.save(eval_log_path, np.array([e['Average Return'] for e in evaluations]))
      pd.DataFrame(evaluations).to_csv(eval_log_path.replace('.npy', '.csv'), index=False)
    # Print a summary
    last, best = evaluations[-1], max(evaluations, key=lambda x: x['Average Return'])
    summary = (f'<{self.config_idx}> [Eval] Summary: Last Average Return={last["Average Return"]:.2f} '
               f'@ Step {last["Step"]}, Best Average Return={best["Average Return"]:.2f} '
               f'@ Step {best["Step"]} (saved to {eval_log_path})')
    print(summary)
    agent.logger.info(summary)
    return np.array([e['Average Return'] for e in evaluations])
  
  def save_model(self):
    self.agent.save_model(self.model_path)
  
  def load_model(self):
    self.agent.load_model(self.model_path)

  def save_config(self):
    cfg_json = json.dumps(self.cfg, indent=2)
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()