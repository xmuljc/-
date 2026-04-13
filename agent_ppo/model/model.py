#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Split-branch actor-critic model for Gorge Chase PPO.
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight.data)
    nn.init.zeros_(layer.bias.data)
    return layer


def make_conv_layer(in_channels, out_channels, kernel_size=3, padding=1):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    nn.init.orthogonal_(layer.weight.data)
    nn.init.zeros_(layer.bias.data)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_situation_ppo"
        self.device = device

        self.spatial_channels = Config.SPATIAL_CHANNELS
        self.spatial_size = Config.SPATIAL_MAP_SIZE
        self.spatial_dim = Config.SPATIAL_DIM
        self.summary_dim = Config.SUMMARY_DIM

        self.spatial_encoder = nn.Sequential(
            make_conv_layer(self.spatial_channels, 16),
            nn.ReLU(),
            make_conv_layer(16, 32),
            nn.ReLU(),
            nn.Flatten(),
            make_fc_layer(32 * self.spatial_size * self.spatial_size, 128),
            nn.ReLU(),
        )

        self.summary_encoder = nn.Sequential(
            make_fc_layer(self.summary_dim, 64),
            nn.ReLU(),
            make_fc_layer(64, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            make_fc_layer(128 + 64, 128),
            nn.ReLU(),
            make_fc_layer(128, 64),
            nn.ReLU(),
        )

        self.actor_head = make_fc_layer(64, Config.ACTION_NUM)
        self.critic_head = make_fc_layer(64, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        spatial_obs = obs[:, : self.spatial_dim].reshape(
            -1,
            self.spatial_channels,
            self.spatial_size,
            self.spatial_size,
        )
        summary_obs = obs[:, self.spatial_dim : self.spatial_dim + self.summary_dim]

        spatial_hidden = self.spatial_encoder(spatial_obs)
        summary_hidden = self.summary_encoder(summary_obs)
        fused_hidden = self.fusion(torch.cat([spatial_hidden, summary_hidden], dim=1))

        logits = self.actor_head(fused_hidden)
        value = self.critic_head(fused_hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
