import os

# 仅连续控制 (MuJoCo/DMC) 需要；该路径不存在时忽略即可
try:
  os.add_dll_directory(r"C:\Users\Administrator\.mujoco\mujoco210\bin")
except Exception:
  pass
import sys
import argparse

from utils.sweeper import Sweeper
from utils.helper import make_dir
from experiment import Experiment

def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default=r'/home/gaogong/data/diffusion/augmentation_model/Explorer/configs/DIvTD_minatar.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
  # 验证 (evaluation) 相关参数: 不传则使用配置文件中的值(若没有则使用下面的默认值)
  parser.add_argument('--eval_freq', type=float, default=None, help='Run evaluation every N training steps (0 disables evaluation)')
  parser.add_argument('--eval_episodes', type=int, default=None, help='Number of episodes per evaluation')
  parser.add_argument('--eval_epsilon', type=float, default=None, help='Exploration rate used during evaluation (0 -> greedy policy)')
  parser.add_argument('--save_model', action='store_true', help='Save the model at every evaluation')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  
  # Set config dict default value
  cfg.setdefault('network_update_steps', 1)
  cfg['env'].setdefault('max_episode_steps', -1)
  cfg.setdefault('show_tb', False)
  cfg.setdefault('render', False)
  cfg.setdefault('gradient_clip', -1)
  cfg.setdefault('hidden_act', 'ReLU')
  cfg.setdefault('output_act', 'Linear')

  # 验证 (evaluation) 设置, 优先级: 命令行 > 配置文件 > 默认值
  # 与 main_atari.py 的 --eval_freq / --eval_episodes / --eval_epsilon 对应
  cfg.setdefault('eval_freq', 0)
  cfg.setdefault('eval_episodes', 10)
  cfg.setdefault('eval_epsilon', 0.0)
  cfg.setdefault('save_model', False)
  if args.eval_freq is not None:
    cfg['eval_freq'] = args.eval_freq
  if args.eval_episodes is not None:
    cfg['eval_episodes'] = args.eval_episodes
  if args.eval_epsilon is not None:
    cfg['eval_epsilon'] = args.eval_epsilon
  if args.save_model:
    cfg['save_model'] = True
  cfg['eval_freq'] = int(cfg['eval_freq'])
  cfg['eval_episodes'] = int(cfg['eval_episodes'])
  cfg['eval_epsilon'] = float(cfg['eval_epsilon'])

  # Set experiment name and log paths
  # 用 os.path 取文件名主干，兼容 Windows 反斜杠路径。
  # 原先的 split('/')[-1].split('.')[0] 在遇到 'D:\...\xxx.json' 这类路径时
  # 会把整串路径当成实验名，进而 makedirs('./logs/D:') 触发 WinError 123。
  cfg['exp'] = os.path.splitext(os.path.basename(args.config_file))[0]

  # 保存路径改为: 算法名字(cfg['agent']['name']) / 任务名字(cfg['env']['name']) / seed /
  # 随机种子直接取 json 配置文件中的 cfg['seed'], 不再随机生成:
  # 关闭 generate_random_seed, 否则 Experiment 会用随机种子覆盖掉 json 中的 seed,
  # 导致目录名里的 seed 与实际训练使用的 seed 不一致。
  cfg['generate_random_seed'] = False
  cfg['seed'] = int(cfg.get('seed', 0))
  agent_name = str(cfg['agent']['name'])
  env_name = str(cfg['env']['name'])
  seed = str(cfg['seed'])
  # 防止名字里出现路径分隔符等非法字符
  def safe_name(name):
    return str(name).replace('/', '_').replace('\\', '_').replace(os.sep, '_').strip()
  if len(args.slurm_dir) > 0:
    cfg['logs_dir'] = os.path.join(args.slurm_dir, safe_name(agent_name), safe_name(env_name), seed) + '/'
  else:
    cfg['logs_dir'] = os.path.join('./logs', safe_name(agent_name), safe_name(env_name), seed) + '/'
  make_dir(cfg['logs_dir'])
  cfg['train_log_path'] = cfg['logs_dir'] + 'result_Train.feather'
  cfg['test_log_path'] = cfg['logs_dir'] + 'result_Test.feather'
  cfg['model_path'] = cfg['logs_dir'] + 'model.pt'
  cfg['cfg_path'] = cfg['logs_dir'] + 'config.json'
  # 验证结果的保存路径: evaluations.npy (平均回报曲线) / evaluations.csv (详细统计)
  cfg['eval_log_path'] = cfg['logs_dir'] + 'evaluations.npy'

  exp = Experiment(cfg)
  # 打印实验信息 (与 main_atari.py 保持一致)
  print("---------------------------------------")
  print(f"Agent: {cfg['agent']['name']}, Env: {cfg['env']['name']}, Seed: {cfg.get('seed', '?')}")
  print("---------------------------------------")

  exp.run()
  # 打印并保存最终的验证结果
  exp.save_eval_result()

if __name__=='__main__':
  main(sys.argv)