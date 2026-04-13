#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    # 每局游戏开始时清空上一局的状态：
    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

        # 特征向量包括
    # hero_feat(5) — 鲁班的位置、闪现CD、buff剩余时间、移动速度
    # monster_feats(10) — 两只怪兽各自的可见性、相对位置、速度、距离
    # treasure_feat(4) — 最近宝箱的方向和距离
    # buff_feat(4) — 最近加速buff的方向和距离
    # obstacle_feat(12) — 八个方向的可通行距离和周围障碍物密度
    # motion_feat(6) — 移动位移、卡住步数、是否碰墙
    # local_map_feat(441) — 21×21的局部地图网格
    # legal_action(16) — 8个移动方向+8个闪现方向的合法性
    # progress_feat(7) — 当前步数、得分、闪现是否可用等进度信息
    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.
        这个方法把环境返回的原始观测数据加工成两部分：
        obs_data — 包含特征向量（feature）和下一步允许的合法动作掩码（legal_action），传给神经网络做推理用。
        remain_info — 包含这一步的奖励值，后面构造SampleData时会用到。
        核心工作其实在 self.preprocessor.feature_process() 里完成，它负责从原始env_obs中提取特征、计算reward。
        这个方法只是把结果包装成统一格式。如果要改reward逻辑，需要去看preprocessor里的 feature_process 实现。

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        # logits — 策略头的原始输出，长度16的数组，对应16个动作（8个移动方向+8个闪现方向）。
        # 数值越大表示网络越倾向于选这个动作。
        
        # value — 价值头的输出，一个标量。表示网络认为当前状态未来能拿到多少累计奖励。
        # 比如输出200，意思是"从现在开始到这局结束，预计还能拿200分"。
        # 这个值用来计算GAE优势估计，指导策略更新。
        
        # prob — 把logits经过legal_action掩码+softmax得到。
        # 先把不合法动作对应的logits设为极小值（-1e20），然后做softmax归一化，
        # 让所有合法动作的概率加起来等于1。比如logits是[3, 1, -1e20, 2, ...]，
        # softmax后可能变成[0.45, 0.06, 0.00, 0.22, ...]，0.00就是被掩码掉的不合法动作。
        
        logits, value, prob = self._run_model(feature, legal_action)
        


        # action是按概率随机采样一个动作（训练用，保持探索）
        action = self._legal_sample(prob, use_max=False)
        # d_action是直接选概率最大的动作（评估用，贪心策略）。
        d_action = self._legal_sample(prob, use_max=True)
        # 两个都算出来存着，训练时用action，评估时用d_action。

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        loaded_state_dict = torch.load(model_file_path, map_location=self.device)
        current_state_dict = self.model.state_dict()

        compatible_state_dict = {}
        for key, value in loaded_state_dict.items():
            if key in current_state_dict and current_state_dict[key].shape == value.shape:
                compatible_state_dict[key] = value

        if len(compatible_state_dict) == len(current_state_dict):
            self.model.load_state_dict(compatible_state_dict)
            self.logger.info(f"load model {model_file_path} successfully")
            return

        current_state_dict.update(compatible_state_dict)
        self.model.load_state_dict(current_state_dict)
        self.logger.warning(
            f"load model {model_file_path} partially: "
            f"matched {len(compatible_state_dict)}/{len(current_state_dict)} tensors"
        )

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference, return logits, value, prob.

        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
