"""gymnasium -> gym 0.25 兼容层

背景
----
本项目 (Explorer-master) 基于 gym 0.25 的 API 编写:
    obs = env.reset()                     # 只返回观测
    obs, reward, done, info = env.step(a) # 4 元组

但下列环境包都是基于 **gymnasium** 实现的，它们只把自己注册进 gymnasium 的
registry，且 step/reset 采用新 API (reset 返回 (obs, info)，step 返回 5 元组):
    gym_minatar      entry_point: 'gym_minatar.envs:BaseEnv'
    gym_pygame       entry_point: 'gym_pygame.envs:*'
    gym_exploration  entry_point: 'gym_exploration.envs:*'

因此 env = gym.make('Asterix-MinAtar-v0') 会抛
    gym.error.NameNotFound: Environment Asterix-MinAtar doesn't exist.

本模块把 gymnasium registry 中属于这些包的 env ID 重新注册到 gym 0.25 的
registry，并用 GymnasiumCompatEnv 做 API 适配。
"""

import gym
import gymnasium

# 需要桥接进 gym 0.25 的环境包（按 entry_point 前缀匹配）
TARGET_PACKAGES = ('gym_minatar', 'gym_pygame', 'gym_exploration')


def _convert_space(space):
  """gymnasium space -> gym 0.25 space"""
  if isinstance(space, gymnasium.spaces.Discrete):
    return gym.spaces.Discrete(space.n)
  if isinstance(space, gymnasium.spaces.Box):
    return gym.spaces.Box(low=space.low, high=space.high,
                          shape=space.shape, dtype=space.dtype)
  if isinstance(space, gymnasium.spaces.MultiBinary):
    return gym.spaces.MultiBinary(space.n)
  if isinstance(space, gymnasium.spaces.MultiDiscrete):
    return gym.spaces.MultiDiscrete(space.nvec)
  return space


class GymnasiumCompatEnv(gym.Env):
  """把 gymnasium 环境包装成 gym 0.25 API 的环境。

  特殊处理:
    * unwrapped 返回 self —— 使得 envs/env.py 的 get_env_group_title() 能通过
      env.unwrapped.spec.entry_point 判定环境分组（需要保留原始的
      'gym_minatar.envs:BaseEnv' 等字符串）。
    * __getattr__ 代理到内部 gymnasium 环境 —— 使得 VanillaDQN.createNN() 中
      env['Train'].game.state_shape() 这类访问仍然可用。
  """

  def __init__(self, env_id, entry_point, **kwargs):
    self._gymnasium_env = gymnasium.make(env_id, **kwargs)
    self.env_id = env_id
    self.observation_space = _convert_space(self._gymnasium_env.observation_space)
    self.action_space = _convert_space(self._gymnasium_env.action_space)
    self.metadata = getattr(self._gymnasium_env, 'metadata', {})
    self.reward_range = getattr(self._gymnasium_env, 'reward_range',
                                (-float('inf'), float('inf')))
    # 保留原 entry_point 字符串，供 get_env_group_title() 做分组判定
    self._spec = gym.envs.registration.EnvSpec(id=env_id, entry_point=entry_point)

  @property
  def unwrapped(self):
    return self

  @property
  def spec(self):
    return self._spec

  @spec.setter
  def spec(self, value):
    # gym.make() 在创建环境后会执行 `env.unwrapped.spec = <注册时的 spec>`，
    # 而注册时用的是工厂函数（entry_point 是 callable，无法用于分组判定）。
    # 这里忽略外部赋值，保留 entry_point 为原始的 'gym_minatar.envs:BaseEnv'。
    pass

  def reset(self, **kwargs):
    obs, _info = self._gymnasium_env.reset(**kwargs)
    return obs

  def step(self, action):
    obs, reward, terminated, truncated, info = self._gymnasium_env.step(action)
    return obs, float(reward), bool(terminated or truncated), info

  def seed(self, seed=None):
    if hasattr(self._gymnasium_env, 'seed'):
      self._gymnasium_env.seed(seed)
    if seed is not None:
      try:
        self.action_space.seed(seed)
      except Exception:
        pass
    return [seed]

  def render(self, *args, **kwargs):
    return self._gymnasium_env.render(*args, **kwargs)

  def close(self):
    return self._gymnasium_env.close()

  def __getattr__(self, name):
    # 仅当常规属性查找失败时调用；下划线开头直接抛错，避免 __init__ 期间递归
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      return getattr(self._gymnasium_env, name)
    except AttributeError:
      # gymnasium.make() 返回的是被 OrderEnforcing / PassiveEnvChecker 包过的对象，
      # 这些 wrapper 不转发自定义属性（如 MinAtar 的 .game），故再向 unwrapped 找一次
      return getattr(self._gymnasium_env.unwrapped, name)


def register_gymnasium_envs(packages=TARGET_PACKAGES):
  """把 gymnasium 中属于 packages 的环境注册进 gym 0.25 registry。

  返回新注册的 env ID 列表（已存在或注册失败的会跳过）。
  """
  # 这里必须是 **gym** 的 registry：上面迭代的是 gymnasium 的 registry，
  # 若拿它做去重判断则每个 env_id 都会命中而全部跳过，导致一个都注册不进去。
  registry = gym.envs.registration.registry
  registered = []
  for env_id, spec in list(gymnasium.registry.items()):
    entry_point = str(getattr(spec, 'entry_point', '') or '')
    if not any(pkg in entry_point for pkg in packages):
      continue
    if env_id in registry:
      continue

    def _factory(env_id=env_id, entry_point=entry_point, **kwargs):
      return GymnasiumCompatEnv(env_id, entry_point, **kwargs)

    try:
      gym.envs.registration.register(id=env_id, entry_point=_factory)
      registered.append(env_id)
    except Exception:
      # 单个环境注册失败不应影响其它环境
      pass
  return registered
