#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
"""

import os
import time

import torch

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.entropy_beta_start = Config.BETA_START
        self.entropy_beta_end = Config.BETA_END
        self.entropy_decay_steps = max(1, Config.BETA_DECAY_STEPS)
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0
        self.train_step = 0
        self.value_running_mean = 0.0
        self.value_running_var = 1.0
        self.value_running_count = 1e-4

    def learn(self, list_sample_data):
        if not list_sample_data:
            return None

        obs = self._stack_to_device([f.obs for f in list_sample_data], dtype=torch.float32)
        legal_action = self._stack_to_device([f.legal_action for f in list_sample_data], dtype=torch.float32)
        act = self._stack_to_device([f.act for f in list_sample_data], dtype=torch.float32).view(-1, 1)
        old_prob = self._stack_to_device([f.prob for f in list_sample_data], dtype=torch.float32)
        reward = self._stack_to_device([f.reward for f in list_sample_data], dtype=torch.float32)
        advantage = self._stack_to_device([f.advantage for f in list_sample_data], dtype=torch.float32)
        old_value = self._stack_to_device([f.value for f in list_sample_data], dtype=torch.float32)
        reward_sum = self._stack_to_device([f.reward_sum for f in list_sample_data], dtype=torch.float32)

        self._update_value_stats(reward_sum)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)
        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        results = {
            "total_loss": round(total_loss.item(), 4),
            "value_loss": round(info_list[0].item(), 4),
            "policy_loss": round(info_list[1].item(), 4),
            "entropy_loss": round(info_list[2].item(), 4),
            "entropy_beta": round(info_list[3], 6),
            "sample_reward_mean": round(reward.mean().item(), 4),
        }

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            self._log(
                "info",
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} "
                f"entropy_beta:{results['entropy_beta']}",
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

        return results

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        prob_dist = self._masked_softmax(logits, legal_action)

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        vp = self._normalize_value(value_pred)
        ov = self._normalize_value(old_value)
        tdret = self._normalize_value(reward_sum)
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()
        entropy_beta = self._current_entropy_beta()
        total_loss = self.vf_coef * value_loss + policy_loss - entropy_beta * entropy_loss

        return total_loss, [value_loss, policy_loss, entropy_loss, entropy_beta]

    def _masked_softmax(self, logits, legal_action):
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)

    def _stack_to_device(self, values, dtype):
        tensors = []
        for value in values:
            if torch.is_tensor(value):
                tensor = value
            else:
                tensor = torch.as_tensor(value)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            tensors.append(tensor)
        return torch.stack(tensors).to(self.device)

    def _current_entropy_beta(self):
        decay_progress = min(1.0, self.train_step / float(self.entropy_decay_steps))
        return self.entropy_beta_start + (self.entropy_beta_end - self.entropy_beta_start) * decay_progress

    def _update_value_stats(self, values):
        flat_values = values.detach().view(-1).float()
        if flat_values.numel() <= 0:
            return

        batch_mean = flat_values.mean().item()
        batch_var = flat_values.var(unbiased=False).item()
        batch_count = float(flat_values.numel())

        delta = batch_mean - self.value_running_mean
        total_count = self.value_running_count + batch_count
        new_mean = self.value_running_mean + delta * batch_count / total_count

        m_a = self.value_running_var * self.value_running_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.value_running_count * batch_count / total_count

        self.value_running_mean = new_mean
        self.value_running_var = max(m2 / total_count, Config.VALUE_NORM_EPS)
        self.value_running_count = total_count

    def _normalize_value(self, value_tensor):
        mean = value_tensor.new_tensor(self.value_running_mean)
        std = value_tensor.new_tensor(max(self.value_running_var, Config.VALUE_NORM_EPS)).sqrt()
        return (value_tensor - mean) / std

    def _log(self, level, message):
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if log_fn is None:
            self.logger.info(message)
            return
        log_fn(message)
