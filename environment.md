# Project Structure & Modules

Eureka项目由三个核心模块组成，它们协同工作以实现基于大语言模型的奖励函数设计和强化学习训练：

## 1. eureka
这是项目的核心模块，负责利用大语言模型（如GPT-4）自动生成和优化奖励函数。主要功能包括：
- 利用LLM的代码生成能力创建初始奖励函数
- 通过迭代优化过程改进奖励函数
- 提取和评估生成的奖励代码
- 与底层强化学习框架集成
- 提供完整的实验管理和结果记录

模块位置：`/home/ubuntu/project/Eureka/eureka`

## 2. isaacgymenvs
提供了基于NVIDIA Isaac Gym物理引擎的强化学习环境集合。主要功能包括：
- 实现各种机器人控制任务（如人形机器人、灵巧手等）
- 提供高性能的物理模拟
- 定义观察空间、动作空间和环境动力学
- 支持大规模并行环境模拟
- 包含训练好的策略检查点（如论文中展示的转笔策略）

模块位置：`/home/ubuntu/project/Eureka/isaacgymenvs`

## 3. rl_games
这是一个高性能的强化学习框架，为Eureka提供强化学习训练后端。主要功能包括：
- 实现多种先进的强化学习算法
- 提供GPU加速的训练流程
- 支持不同类型的策略网络架构
- 提供模型保存、加载和评估功能
- 兼容多种环境接口（包括Isaac Gym）

模块位置：`/home/ubuntu/project/Eureka/rl_games`

这三个模块协同工作，形成了完整的Eureka工作流程：eureka模块生成和优化奖励函数，isaacgymenvs提供环境接口和物理模拟，rl_games执行实际的强化学习训练过程。



## 指标含义

1. Max Success (最大成功次数)
实现方法：
从 TensorBoard 日志中提取 consecutive_successes 指标（metric_cur_max），并将其添加到 successes 列表中。
在每次迭代结束时，从所有生成的奖励函数样本对应的 successes 列表中，选择最大的 consecutive_successes 值作为当前迭代的 max_success。
具体代码：max_success = successes[best_sample_idx]，其中 best_sample_idx 是 successes 列表中最大值的索引。
含义： 表示在当前迭代中，所有成功训练的奖励函数样本中，RL 策略在评估期间能够达到的最高连续成功次数。这个指标直接反映了当前最佳奖励函数引导策略解决任务的能力。
2. Execute Rate (执行率)
实现方法：
计算方法为成功执行的奖励函数样本数除以总样本数。
具体代码：execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample。
这里使用 >= 0. 作为判断成功的条件，因为失败的样本会被标记为 DUMMY_FAILURE (-10000)。
含义： 表示在当前迭代中，LLM 生成的奖励函数代码能够成功被系统执行并用于训练的比例。这个指标衡量了 LLM 生成代码的语法正确性、运行时无错误以及与环境接口的兼容性。高执行率意味着 LLM 能够生成可靠的代码。
3. Max Success Reward Correlation (最大成功奖励相关性)
实现方法：
当训练成功时，通过计算真实奖励 (gt_reward) 和生成的奖励函数 (gpt_reward) 之间的皮尔逊相关系数来获取奖励相关性。
具体代码：reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]。
然后，选择与 max_success 对应的 reward_correlation 作为当前迭代的 max_success_reward_correlation。
具体代码：max_success_reward_correlation = reward_correlations[best_sample_idx]。
含义： 表示在获得最高连续成功次数的那个奖励函数中，其生成的奖励值与环境提供的“真实”奖励值（通常是某种预定义的、可能稀疏的或专家设计的任务完成度指标）之间的相关性。这个指标非常关键，它衡量了 LLM 设计的奖励函数在多大程度上能够捕捉或近似环境的真实任务目标。高相关性意味着 LLM 成功地将稀疏的真实奖励转化为密集且可学习的奖励信号。



### Isaac Gym任务移植完全指南
本指南详细介绍如何在Eureka框架中创建新的Isaac Gym任务以及将现有环境移植到最新版本的步骤。

一、创建新任务
1. 基本架构
在isaacgymenvs/tasks目录下创建新的脚本文件，并继承VecTask基类：

```python
from isaacgym import gymtorch
from isaacgym import gymapi
from .base.vec_task import VecTask

class MyNewTask(VecTask):
    def __init__(self, config_dict, sim_device, headless):
        super().__init__(cfg=config_dict)
        # 初始化状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
```
2. 必须实现的核心方法
2.1 创建物理模拟环境
```python
def create_sim(self):
    # 设置up-axis
    # 调用super().create_sim并提供设备参数
    # 创建地面
    # 设置环境
```
2.2 物理步骤前处理
```python
def pre_physics_step(self, actions):
    # 实现物理模拟前的代码
    # 例如：应用动作
```
2.3 物理步骤后处理
```python
def post_physics_step(self):
    # 实现物理模拟后的代码
    # 例如：计算奖励、计算观测值
```
3. 集成到训练系统
在tasks/__init__.py文件中添加新任务：

```python
from isaacgymenvs.tasks.my_new_task import MyNewTask

isaac_gym_task_map = {
    'Anymal': Anymal,
    # ...
    'MyNewTask': MyNewTask,
}
```
4. 创建配置文件
任务配置：在cfg对应文件夹中创建与任务名称相同的YAML文件（如MyNewTask.yaml）
训练配置：在cfg对应文件夹中创建带有PPO后缀的训练配置文件（如MyNewTaskPPO.yaml）
5. 运行新任务
```bash
# 运行命令
python train.py task=MyNewTask
```
二、更新现有环境
1. 导入更新
原rlgpu.utils.torch_jit_utils现改为utils.torch_jit_utils
原BaseTask类现改为VecTask，导入路径从from rlgpu.tasks.base.base_task import BaseTask改为from .base.vec_task import VecTask
2. 类定义更新
任务类应继承自VecTask而非BaseTask
__init__()方法参数简化为cfg、sim_device和headless
无需在__init__()方法中设置self.sim_params和self.physics_engine
调用父类初始化方法时需要额外三个参数：rl_device、sim_device和headless
```python
super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, headless=headless)
```
3. 方法更新
VecTask现在定义了reset_idx()方法，用于重置指定索引的环境
VecTask中的reset()方法不再接受环境索引作为参数，建议重命名任务中的同名方法
4. 资源路径更新
资源文件已移至assets/目录下，请确保路径正确
5. 配置文件更新
5.1 任务配置更新项
- physics_engine
- numEnvs
- use_gpu_pipeline
- num_threads
- solver_type
- use_gpu
- num_subscenes

5.2 训练配置更新项
- seed
- load_checkpoint
- load_pathname
- full_experiment_name
- num_actors
- max_epochs

5.3 RL Games参数名称变化
- lr_threshold → kl_threshold
- steps_num → horizon_length

三、集成到Eureka框架
1. 使用工具链转换
利用Eureka提供的工具进行任务转换：
- prune_env_isaac.py：将Isaac Gym任务转换为Eureka兼容版本
- create_task.py：通过复制并修改YAML配置文件创建新任务

2. 实现VecTask接口
确保实现所有必需的方法以与Isaac Gym框架兼容：
- create_sim：物理模拟初始化
- pre_physics_step：动作处理
- post_physics_step：奖励和观测计算
- reset_idx：环境重置
- allocate_buffers：数据缓冲区分配
- compute_observations：观测空间计算

3. 处理观测空间
- 支持多种观测类型配置
- 正确处理物理状态张量（如dof_state、root_state、rigid_body_state）
- 遵循Eureka的模块化设计，可将观测计算分离到专门的文件中（如shadow_hand_obs.py）

通过遵循本指南，您可以顺利地在Eureka框架中创建新的Isaac Gym任务或更新现有环境，确保与最新版本的兼容性并充分利用GPU加速的物理模拟能力。