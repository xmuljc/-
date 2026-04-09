#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(conv.weight.data)
    nn.init.zeros_(conv.bias.data)
    return conv


class Model(nn.Module):
    """Single MLP backbone + Actor/Critic dual heads.

    单 MLP 骨干 + Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        scalar_dim = Config.DIM_OF_OBSERVATION - Config.LOCAL_MAP_DIM
        local_map_size = Config.LOCAL_MAP_SIZE
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.scalar_encoder = nn.Sequential(
            make_fc_layer(scalar_dim, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
        )

        self.map_encoder = nn.Sequential(
            make_conv_layer(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            make_conv_layer(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_layer(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        pooled_map_size = local_map_size // 2
        map_hidden_dim = 32 * pooled_map_size * pooled_map_size
        self.map_projector = nn.Sequential(
            make_fc_layer(map_hidden_dim, 128),
            nn.ReLU(),
        )

        self.backbone = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(128, action_num)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(128, value_num)

    def forward(self, obs, inference=False):
        feature_groups = torch.split(obs, Config.FEATURE_SPLIT_SHAPE, dim=1)
        local_map = feature_groups[Config.LOCAL_MAP_GROUP_INDEX].reshape(
            -1,
            1,
            Config.LOCAL_MAP_SIZE,
            Config.LOCAL_MAP_SIZE,
        )
        scalar_groups = feature_groups[: Config.LOCAL_MAP_GROUP_INDEX] + feature_groups[Config.LOCAL_MAP_GROUP_INDEX + 1 :]
        scalar_obs = torch.cat(scalar_groups, dim=1)

        scalar_hidden = self.scalar_encoder(scalar_obs)
        map_hidden = self.map_projector(self.map_encoder(local_map))
        hidden = self.backbone(torch.cat([scalar_hidden, map_hidden], dim=1))
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
