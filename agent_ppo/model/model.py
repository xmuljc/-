#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
"""

import math

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
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
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_structured_attention"
        self.device = device

        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        local_map_size = Config.LOCAL_MAP_SIZE

        self.hero_encoder = nn.Sequential(
            make_fc_layer(Config.HERO_FEATURE_DIM, 64),
            nn.ReLU(),
            make_fc_layer(64, 64),
            nn.ReLU(),
        )
        self.monster_encoder = nn.Sequential(
            make_fc_layer(Config.MONSTER_FEATURE_DIM, 64),
            nn.ReLU(),
            make_fc_layer(64, 64),
            nn.ReLU(),
        )
        self.target_encoder = nn.Sequential(
            make_fc_layer(Config.TARGET_FEATURE_DIM, 64),
            nn.ReLU(),
            make_fc_layer(64, 64),
            nn.ReLU(),
        )
        self.skill_encoder = nn.Sequential(
            make_fc_layer(Config.SKILL_FEATURE_DIM, 48),
            nn.ReLU(),
            make_fc_layer(48, 48),
            nn.ReLU(),
        )
        self.obstacle_encoder = nn.Sequential(
            make_fc_layer(Config.OBSTACLE_FEATURE_DIM, 64),
            nn.ReLU(),
            make_fc_layer(64, 64),
            nn.ReLU(),
        )
        self.motion_encoder = nn.Sequential(
            make_fc_layer(Config.MOTION_FEATURE_DIM, 32),
            nn.ReLU(),
            make_fc_layer(32, 32),
            nn.ReLU(),
        )
        self.progress_encoder = nn.Sequential(
            make_fc_layer(Config.PROGRESS_FEATURE_DIM, 48),
            nn.ReLU(),
            make_fc_layer(48, 48),
            nn.ReLU(),
        )
        self.legal_encoder = nn.Sequential(
            make_fc_layer(Config.LEGAL_ACTION_DIM, 32),
            nn.ReLU(),
            make_fc_layer(32, 32),
            nn.ReLU(),
        )

        self.map_encoder = nn.Sequential(
            make_conv_layer(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            make_conv_layer(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_conv_layer(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        pooled_map_size = local_map_size // 4
        map_hidden_dim = 64 * pooled_map_size * pooled_map_size
        self.map_projector = nn.Sequential(
            make_fc_layer(map_hidden_dim, 128),
            nn.ReLU(),
        )

        self.attention_query = nn.Sequential(
            make_fc_layer(64 + 48 + 64 + 32 + 48, 64),
            nn.ReLU(),
            make_fc_layer(64, 64),
        )

        fusion_input_dim = 64 + 48 + 64 + 32 + 48 + 32 + 128 + 64 + 64 + 64
        self.backbone = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            make_fc_layer(fusion_input_dim, 384),
            nn.ReLU(),
            make_fc_layer(384, 192),
            nn.ReLU(),
            make_fc_layer(192, 128),
            nn.ReLU(),
        )

        self.actor_head = make_fc_layer(128, action_num)
        self.critic_head = make_fc_layer(128, value_num)

    def forward(self, obs, inference=False):
        del inference
        feature_groups = torch.split(obs, Config.FEATURE_SPLIT_SHAPE, dim=1)

        hero_obs = feature_groups[Config.HERO_GROUP_INDEX]
        monster_obs_1 = feature_groups[Config.MONSTER_GROUP_INDICES[0]]
        monster_obs_2 = feature_groups[Config.MONSTER_GROUP_INDICES[1]]
        treasure_obs = feature_groups[Config.TARGET_GROUP_INDICES[0]]
        buff_obs = feature_groups[Config.TARGET_GROUP_INDICES[1]]
        skill_obs = feature_groups[Config.SKILL_GROUP_INDEX]
        obstacle_obs = feature_groups[Config.OBSTACLE_GROUP_INDEX]
        motion_obs = feature_groups[Config.MOTION_GROUP_INDEX]
        local_map_obs = feature_groups[Config.LOCAL_MAP_GROUP_INDEX].reshape(
            -1,
            1,
            Config.LOCAL_MAP_SIZE,
            Config.LOCAL_MAP_SIZE,
        )
        legal_obs = feature_groups[Config.LEGAL_ACTION_GROUP_INDEX]
        progress_obs = feature_groups[Config.PROGRESS_GROUP_INDEX]

        hero_hidden = self.hero_encoder(hero_obs)
        monster_hidden_1 = self.monster_encoder(monster_obs_1)
        monster_hidden_2 = self.monster_encoder(monster_obs_2)
        treasure_hidden = self.target_encoder(treasure_obs)
        buff_hidden = self.target_encoder(buff_obs)
        skill_hidden = self.skill_encoder(skill_obs)
        obstacle_hidden = self.obstacle_encoder(obstacle_obs)
        motion_hidden = self.motion_encoder(motion_obs)
        legal_hidden = self.legal_encoder(legal_obs)
        progress_hidden = self.progress_encoder(progress_obs)
        map_hidden = self.map_projector(self.map_encoder(local_map_obs))

        entity_stack = torch.stack(
            [monster_hidden_1, monster_hidden_2, treasure_hidden, buff_hidden],
            dim=1,
        )
        query_hidden = self.attention_query(
            torch.cat([hero_hidden, skill_hidden, obstacle_hidden, motion_hidden, progress_hidden], dim=1)
        )
        attention_logits = torch.matmul(entity_stack, query_hidden.unsqueeze(-1)).squeeze(-1)
        attention_logits = attention_logits / math.sqrt(entity_stack.size(-1))
        attention_weight = torch.softmax(attention_logits, dim=1)
        entity_context = torch.sum(entity_stack * attention_weight.unsqueeze(-1), dim=1)

        monster_summary = torch.max(
            torch.stack([monster_hidden_1, monster_hidden_2], dim=1),
            dim=1,
        ).values
        resource_summary = torch.max(
            torch.stack([treasure_hidden, buff_hidden], dim=1),
            dim=1,
        ).values

        fusion_input = torch.cat(
            [
                hero_hidden,
                skill_hidden,
                obstacle_hidden,
                motion_hidden,
                progress_hidden,
                legal_hidden,
                map_hidden,
                entity_context,
                monster_summary,
                resource_summary,
            ],
            dim=1,
        )
        hidden = self.backbone(fusion_input)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
