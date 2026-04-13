#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO.
"""


class Config:
    # Spatial branch: 6 local situation maps on a 5x5 patch.
    # passable / freedom / danger / trap / control / opportunity
    SPATIAL_CHANNELS = 6
    SPATIAL_MAP_SIZE = 5
    SPATIAL_DIM = SPATIAL_CHANNELS * SPATIAL_MAP_SIZE * SPATIAL_MAP_SIZE

    # Summary branch: feature groups aligned with 指导思路2.
    SUMMARY_FEATURES = [
        11,  # hero main features
        8,   # treasure features
        11,  # monster features
        6,   # skill / buff features
        12,  # local map summary + safe directions
    ]
    SUMMARY_DIM = sum(SUMMARY_FEATURES)

    FEATURES = [SPATIAL_DIM, SUMMARY_DIM]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 movement directions.
    ACTION_NUM = 8

    # Value head size.
    VALUE_NUM = 1

    # PPO hyperparameters.
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    # Static entropy coefficient. Keep it moderate because this project does
    # not currently decay beta over time and the action space is only 8D.
    BETA_START = 0.008
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
