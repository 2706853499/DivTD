# DivTD: Diversity-Regularized Temporal-Difference Learning

DivTD 是一种**多样性正则化的时序差分学习**算法。核心思想是：用 **N 个 Q 网络（critic）集成** 来估计价值，TD 目标取集成的**均值**；同时为了防止集成在自举（bootstrapping）过程中"坍缩"成同一个函数，引入**多样性（标准差匹配）正则化**，让被更新子集的价值离散度去匹配目标集成的离散度。

为控制集成方法的训练/推理开销，DivTD 采用**稀疏训练（sparse training）**：每次梯度更新只从 N 个网络中随机抽取 M 个（M ≤ N）参与前向/反向传播与参数更新。

DivTD 同时适用于**连续动作空间**（Actor-Critic 形式，基于 TD3/TD7）与**离散动作空间**（DQN 形式）。

## 核心机制

设集成包含 N 个 Q 网络 `Q_1, ..., Q_N`，每次更新随机抽取索引子集 `S ⊂ {1..N}`，`|S| = M`：

1. **TD 目标（集成均值）**
   ```
   Q_target = r + γ · mask · mean_j Q_j(s', a*)      # a* 由集成（或目标 actor）给出的贪婪动作
   ```

2. **目标标准差（多样性信号，仅当 M ≠ 1）**
   ```
   target_std = γ · mask · std_j Q_j(s', a*)
   ```

3. **损失函数**
   ```
   L = Huber(Q_S(s,a), Q_target) + α · MSE( std(Q_S(s,a)), target_std )
   ```
   其中 `Q_S` 为被抽中的 M 个网络的预测值（形状 `(B, M)`）。第二项即**多样性正则化**：稀疏更新倾向于把集成"压扁"，该正则项维持集成成员之间的价值离散度（Div 多样性）。

4. **稀疏训练**：每个梯度步只有被抽中的 M 个网络接收梯度，其余网络保持不变，从而显著降低集成训练开销。

5. **可选的间隔式多样性正则**（默认关闭）：对排序后相邻价值估计之差小于 `margin` 的情况施加惩罚（`diversity_margin`，默认 `4e-2`）。

---

## 目录结构

```
DivTD/
├── DivTD-Continuous/        # 连续动作空间实现（DeepMind Control Suite）
│   ├── algos/
│   │   ├── DivTD.py         # DivTD 主算法（Actor + N 个 Critic + Encoder）
│   │   ├── TD7.py           # 基线：TD7
│   │   ├── TQC.py           # 基线：TQC
│   │   └── dist_module.py
│   ├── main_dmc.py          # DMC 训练入口（DivTD / TD7）
│   ├── utils.py             # 经验回放缓冲区（支持优先级采样）
│   └── results_dmc_clip.png # 实验结果图（hopper-hop / quadruped-run）
│
└── DivTD-Discrete/          # 离散动作空间实现（基于 Explorer 框架，MinAtar 等）
    ├── agents/
    │   ├── DivTDDQN.py      # DivTD 主算法（DivTD(DQN)）
    │   └── ...              # 基线：DQN/DDQN/MaxminDQN/AveragedDQN/EnsembleDQN/...
    ├── configs/
    │   └── DIvTD_minatar.json   # DivTD 在 MinAtar 上的实验配置
    ├── envs/                # 环境封装（MinAtar / gym / Atari / PLE / DMC / PyBullet）
    ├── components/          # 网络 / 回放 / 探索 / 归一化组件
    ├── utils/               # sweeper（网格搜索）/ logger / plotter / submitter
    ├── main.py              # 训练入口（配置文件 + 配置索引）
    ├── experiment.py        # 训练-评估循环封装
    ├── plot.py              # 结果绘图
    ├── analysis.py          # 结果分析
    ├── requirements.txt
    └── results_MinAtar.png  # 实验结果图（4 个 MinAtar 游戏）
```

---

## 1. 连续动作空间（DivTD-Continuous）

### 实现要点（`algos/DivTD.py`）

- **Critic**：`Critic` 内含 N 个 `QuantileCriticNet`（`n_quantiles=1`，即每网每动作输出单值 Q），共享 state-action encoder 的 `zs`（状态嵌入）与 `zsa`（状态-动作嵌入）。
- **Encoder**：`zs → zsa` 的状态转移嵌入，用下一状态嵌入的 MSE 损失训练（latent world-model 式辅助任务）。
- **Actor**：TD3/TD7 风格，输入 `(state, zs)`；TD 目标用 target actor + TD3 目标策略平滑噪声（`target_policy_noise` / `noise_clip`）。
- **稀疏训练**：`q_indices = np.random.choice(N, size=M, replace=False)`，只有被选中的子集参与 critic 损失。
- **Actor 更新**：每 `policy_freq` 步执行一次，最大化**累积使用过的**全部网络索引（`list_index`）上的平均 Q。
- **目标网络**：每 `target_update_rate` 步硬同步 actor / critic / encoder 的 target 副本。
- **其他继承自 TD7 的组件**：LAP 风格优先回放（`prioritized=True`）、AvgL1Norm、checkpoint 机制（`maybe_train_and_checkpoint`，以最差回合回报作为策略保留判据）。

### 依赖

- Python 3.8+，PyTorch，NumPy
- `gym_dmc`（DeepMind Control Suite 的 gym 封装）

### 运行

```bash
cd DivTD-Continuous

# DivTD（默认 N=3, M=2, alpha=0.1）
python main_dmc.py --policy DivTD \
    --domain_name hopper --task_name hop \
    --N 3 --M 2 --alpha 0.1 \
    --action_repeat 2 --seed 0 \
    --start_timesteps 10000 --eval_freq 5000 --max_timesteps 500000 \
    --batch_size 512 --discount 0.99 \
    --prioritized True --device cuda:0

# 基线 TD7
python main_dmc.py --policy TD7 --domain_name hopper --task_name hop --seed 0
```

主要超参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--N` | 3 | 集成中 critic 网络个数 |
| `--M` | 2 | 每次梯度更新随机抽取的网络个数（稀疏训练） |
| `--alpha` | 0.1 | 多样性（std 匹配）正则系数 |
| `--policy_noise` | 0.2 | TD3 目标策略噪声（× max_action） |
| `--noise_clip` | 0.5 | 目标噪声截断 |
| `--policy_freq` | 2 | actor 延迟更新频率 |
| `--action_repeat` | 2 | 帧跳过（frame skip） |
| `--batch_size` | 512（DivTD）/ 256（TD7） | 批大小 |
| `--target_update_rate` | 1 | 目标网络同步周期（步） |

### 输出

结果保存在 `results{N}-{M}-{alpha}-{action_repeat}/` 下：

- `DivTD_{domain}_{task}_{seed}.npy`：每次评估的平均回报曲线
- `true_pred_DivTD_...npy`：评估期记录的 `Q(s,a)` 预测值、真实折扣回报 `G`、集成多样性（std）等统计
- `reqDivTD_...npy`

`--save_model` 时模型存至 `./models/`。

### 已含实验结果

`results_dmc_clip.png` 展示了 `hopper-hop` 与 `quadruped-run` 上的三条曲线：**DivTD (N=2)**、**TD7 (avg)**、**TD7 (min)**，分别对应 Average Return、集成 Diversity（价值 std）、以及价值误差 `Q(s,a) − Q^π(s,a)`。结果显示 DivTD 在保持更高集成多样性的同时降低了价值高估偏差，取得更高回报。

---

## 2. 离散动作空间（DivTD-Discrete）

### 实现要点（`agents/DivTDDQN.py`）

基础算法为 DQN（target network + replay buffer + ε-greedy），在其上：

- 构建 **N 个 Q 网络及对应的 N 个 target 网络**（当前代码硬编码 `N=5`、`M=4`、`alpha=0.5`，见 `DivTDDQN.__init__`，如需修改请直接编辑该文件）；
- TD 目标：先用子集集成的均值 Q 选贪婪动作 `a*`，再取各 target 网络在 `a*` 上的 Q 值求均值与标准差；
- 损失：`SmoothL1Loss(Q_S, Q_target) + α · MSE(std(Q_S), target_std)`（std 匹配多样性正则）；
- 稀疏训练：每次更新只随机抽 M 个网络，只对它们做梯度裁剪与优化器 step；
- 动作选择：使用被抽中的 M 个网络的 Q 值**均值**；
- target 网络每 `target_network_update_steps`（默认 1000 步）硬同步一次；
- 可选配置项（写在 JSON 配置的 `agent` 字段）：`diversity_coeff`（间隔式多样性正则，默认 0 = 关闭）、`diversity_margin`（默认 4e-2）、`sort_targets`（默认 false）。

### 运行

实验通过 JSON 配置文件 + 配置索引驱动（网格搜索式）：

```bash
cd DivTD-Discrete

# DivTD on Seaquest-MinAtar
python main.py --config_file ./configs/DIvTD_minatar.json --config_idx 1

# 验证（evaluation）选项：命令行优先级 > 配置文件 > 默认值
python main.py --config_file ./configs/DIvTD_minatar.json --config_idx 1 \
    --eval_freq 5000 --eval_episodes 10 --eval_epsilon 0.0 --save_model
```

注意：`main.py` 中 `--config_file` 的默认路径指向外部 `Explorer/configs/...`，请显式传入 `./configs/DIvTD_minatar.json`。

`DIvTD_minatar.json` 关键设置：环境 `Seaquest-MinAtar-v0`（pixel 输入），`train_steps=5e6`，`memory_size=1e5`（FiniteReplay），`batch_size=32`，RMSprop（lr 为网格 `{3e-3, 1e-3, 3e-4, 1e-4, 3e-5}`，`alpha=0.95, centered, eps=0.01`），`discount=0.99`，`LinearEpsilonGreedy`（1.0 → 0.1），`target_network_update_steps=1000`。

**网格搜索**：先用 `python utils/sweeper.py` 计算配置文件的总组合数，然后遍历配置索引（相同余数的索引对应相同配置，多 seed 只需加总组合数），可用 GNU parallel 并行：

```bash
parallel --eta --ungroup python main.py --config_file ./configs/DIvTD_minatar.json --config_idx {1} ::: $(seq 1 5)
```

### 输出

日志保存在 `./logs/<Agent>/<Env>/<seed>/` 下：

- `result_Train.feather` / `result_Test.feather`：训练/测试逐回合记录
- `evaluations.npy` / `evaluations.csv`：验证期平均回报及详细统计
- `config.json`：本次运行完整配置

绘图：`python plot.py`（输出到 `./figures/`）；结果分析：`python analysis.py`。

### 基线算法（同一框架内可直接对比）

DQN、DDQN、MaxminDQN、AveragedDQN、EnsembleDQN、BootstrappedDQN、NoisyNetDQN、LSVI-PHE、MeDQN（Uniform/Real）、REINFORCE、Actor-Critic、PPO、SAC、DDPG、TD3、RPG。

### 支持的环境（`envs/env.py`）

MinAtar（Asterix / Breakout / Freeway / Seaquest / SpaceInvaders）、gym classic control / box2d、Atari（含 RAM）、gym-pygame、gym-exploration、PLE（仓库内 `PyGame-Learning-Environment/`）、DMC（dmc2gym）、PyBullet。

### 依赖

见 `requirements.txt`：`torch==2.7.1, numpy==1.22.0, matplotlib==3.5.2, pandas==1.4.2, opencv_python==4.8.1.78, psutil==5.9.0, seaborn==0.11.2`。

另需安装：

```bash
pip install "gym[classic_control,box2d]"   # gym 0.25
pip install gym-minatar gym-exploration gym-pygame
pip install git+https://github.com/denisyarats/dmc2gym.git   # 仅 DMC 需要
# 仓库内已附带：
#   gym-games/                    （pip install -e gym-games）
#   PyGame-Learning-Environment/  （pip install -e PyGame-Learning-Environment）
```

### 已含实验结果

`results_MinAtar.png` 展示了 4 个 MinAtar 游戏（Asterix / Breakout / Seaquest / SpaceInvaders）上 **DivTD (DQN)** 与基线（MaxminDQN、AveragedDQN、DDQN、DQN）的平均回报对比，DivTD 在多数游戏中取得最优表现。

---

## 方法总结

| 组件 | 连续（DivTD-Continuous） | 离散（DivTD-Discrete） |
|---|---|---|
| 基础框架 | TD3/TD7（Actor-Critic + Encoder + LAP 优先回放） | DQN（target network + replay + ε-greedy） |
| 集成 | N 个 critic 网络（单分位数） | N 个 Q 网络及对应 target 网络 |
| TD 目标 | 集成均值 `mean_j Q_j(s', a*)`（a* 来自 target actor + 平滑噪声） | 集成均值 `mean_j Q_j(s', a*)`（a* 为集成贪婪动作） |
| 多样性正则 | `α · MSE(std(Q_S), target_std)` | `α · MSE(std(Q_S), target_std)` |
| 稀疏训练 | 每步随机抽 M/N 个网络 | 每步随机抽 M/N 个网络 |
| 动作选择 | actor（确定性策略） | 被抽中子集的 Q 均值 |
| 默认超参 | N=3, M=2, α=0.1（可由命令行指定） | N=5, M=4, α=0.5（代码内硬编码） |


## 致谢

- 离散部分基于 [Explorer](https://github.com/qlan3/Explorer)（Qingfeng Lan 的 PyTorch RL 框架）构建
- 连续部分参考 [TD3](https://github.com/sfujim/TD3) 与 TD7 的实现
- 环境封装参考 dmc2gym、gym-minatar、gym-games、PyGame-Learning-Environment
