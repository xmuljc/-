# Gorge Chase RL Project

基于 KaiWuDRL 的《峡谷追击》强化学习项目，当前仓库包含两套 agent：

- `agent_ppo`：主要训练版本，使用 PPO
- `agent_diy`：简单基线/对照版本

项目版本信息见 [kaiwu.json](kaiwu.json)。

## 目录结构

- [agent_ppo](agent_ppo)：PPO agent、特征工程、模型、训练 workflow
- [agent_diy](agent_diy)：DIY baseline agent
- [conf](conf)：应用与算法配置
- [train_test.py](train_test.py)：本地训练入口

## 训练入口

当前默认训练算法在 [train_test.py](train_test.py) 中设置为：

```python
algorithm_name = "ppo"
```

本地训练入口：

```bash
python train_test.py
```

应用和算法映射配置位于：

- [conf/configure_app.toml](conf/configure_app.toml)
- [conf/algo_conf_gorge_chase.toml](conf/algo_conf_gorge_chase.toml)
- [conf/app_conf_gorge_chase.toml](conf/app_conf_gorge_chase.toml)

## 当前 PPO 方案

当前 PPO 训练主线集中在“先学基础生存，再逐步抬高上限”。

### Reward

当前只保留最基础的四项 reward，定义见 [agent_ppo/conf/conf.py](agent_ppo/conf/conf.py) 与 [agent_ppo/feature/preprocessor.py](agent_ppo/feature/preprocessor.py)：

- `official_reward`
- `danger_shaping`
- `wall_collision_penalty`
- `final_reward`

这样做的目的是减少局部 shaping 干扰，让 Agent 先学会：

- 别被抓
- 怪物逼近时主动拉开距离
- 避免撞墙和低级卡住行为

### Curriculum

当前课程学习总局数设置为 `100000`，相关参数在 [agent_ppo/conf/conf.py](agent_ppo/conf/conf.py)：

- `CURRICULUM_TOTAL_EPISODES = 100000`
- `CURRICULUM_STAGE1_RATIO = 0.30`
- `CURRICULUM_STAGE2_RATIO = 0.70`

目标是避免过早进入高难度，让 Agent 在前期有足够时间学会基础生存。

## 现阶段训练观察

当前路线已经验证可以学到较稳定的基础策略，并开始出现更高分局。现阶段的判断是：

- reward 不再频繁改动
- 优先观察 `30000 / 70000 / 100000` 局检查点
- 如果后续要继续提升上限，优先考虑 `feature` 和地图表示，而不是继续堆 reward

## 下一步优化方向

如果基础生存已经稳定，下一步更值得尝试：

- 优化 feature 表示，尤其是局部地图的语义通道设计
- 提升怪物、障碍、宝箱、buff 的空间关系表达
- 在确认 feature 成为瓶颈后，再考虑扩大网络结构

## 说明

- 仓库默认不提交训练结果图片、`ckpt`、缓存文件和本地调参记录
- 当前 README 主要用于说明代码结构与训练主线，后续可以继续补充实验记录和效果对比
