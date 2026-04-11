#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
"""

from pathlib import Path

import numpy as np
import torch
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model_dir = Path(__file__).resolve().parents[1] / "ckpt"
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
        self.preprocessor.reset()
        self.last_action = -1

    def set_curriculum(self, curriculum_info):
        self.preprocessor.set_curriculum(curriculum_info)



    def observation_process(self, env_obs):
        feature, legal_action, remain_info = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),#特征向量
            legal_action=legal_action,#下一步允许的动作
        )
        return obs_data, remain_info

    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        _, value, prob = self._run_model(feature, legal_action)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        if isinstance(env_obs, list):
            obs_data = env_obs[0]
        elif hasattr(env_obs, "feature") and hasattr(env_obs, "legal_action"):
            obs_data = env_obs
        else:
            obs_data, _ = self.observation_process(env_obs)

        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="latest"):
        model_dir = self._resolve_model_dir(path, create=True)
        model_file_path = model_dir / f"model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)

        latest_file_path = model_dir / "model.ckpt-latest.pkl"
        if model_file_path != latest_file_path:
            torch.save(state_dict_cpu, latest_file_path)

        self._log("info", f"save model {model_file_path} successfully")
        return str(model_file_path)

    def load_model(self, path=None, id="latest"):
        model_dir = self._resolve_model_dir(path, create=False)
        model_file_path = self._resolve_load_path(model_dir, id)

        if model_file_path is None or not model_file_path.exists():
            self._log("warning", f"model checkpoint not found for id={id}, skip loading")
            return False

        try:
            state_dict = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as exc:
            self._log("warning", f"load model {model_file_path} failed: {exc}")
            return False

        self._log("info", f"load model {model_file_path} successfully")
        return True

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        weight = 1e20
        eps = 1e-5
        tmp = input_hidden - weight * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -weight, 1)
        tmp = (np.exp(tmp) + eps) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _resolve_model_dir(self, path, create):
        model_dir = Path(path) if path else self.model_dir
        if create:
            model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _resolve_load_path(self, model_dir, id):
        if str(id) != "latest":
            return model_dir / f"model.ckpt-{str(id)}.pkl"

        latest_file_path = model_dir / "model.ckpt-latest.pkl"
        if latest_file_path.exists():
            return latest_file_path

        if not model_dir.exists():
            return None

        candidates = sorted(model_dir.glob("model.ckpt-*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if candidate.name != "model.ckpt-latest.pkl":
                return candidate
        return None

    def _log(self, level, message):
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if log_fn is None:
            self.logger.info(message)
            return
        log_fn(message)
