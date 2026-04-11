#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase PPO.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()
    return (
        monitor.title("Gorge Chase PPO")
        .add_group(
            group_name="Training Metrics",
            group_name_en="training",
        )
        .add_panel(
            name="Train Reward",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="Official Score",
            name_en="official_score",
            type="line",
        )
        .add_metric(
            metrics_name="official_score",
            expr="avg(official_score{})",
        )
        .end_panel()
        .add_panel(
            name="Step Score",
            name_en="official_step_score",
            type="line",
        )
        .add_metric(
            metrics_name="official_step_score",
            expr="avg(official_step_score{})",
        )
        .end_panel()
        .add_panel(
            name="Treasure Score",
            name_en="official_treasure_score",
            type="line",
        )
        .add_metric(
            metrics_name="official_treasure_score",
            expr="avg(official_treasure_score{})",
        )
        .end_panel()
        .add_panel(
            name="Val Score",
            name_en="val_official_score",
            type="line",
        )
        .add_metric(
            metrics_name="val_official_score",
            expr="avg(val_official_score{})",
        )
        .end_panel()
        .add_panel(
            name="Val Step Score",
            name_en="val_step_score",
            type="line",
        )
        .add_metric(
            metrics_name="val_step_score",
            expr="avg(val_step_score{})",
        )
        .end_panel()
        .add_panel(
            name="Val Steps",
            name_en="val_episode_steps",
            type="line",
        )
        .add_metric(
            metrics_name="val_episode_steps",
            expr="avg(val_episode_steps{})",
        )
        .end_panel()
        .add_panel(
            name="Episode Count",
            name_en="episode_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="episode_cnt",
            expr="sum(episode_cnt{})",
        )
        .end_panel()
        .add_panel(
            name="Curriculum Stage",
            name_en="curriculum_stage",
            type="line",
        )
        .add_metric(
            metrics_name="curriculum_stage",
            expr="avg(curriculum_stage{})",
        )
        .end_panel()
        .add_panel(
            name="Curriculum Progress",
            name_en="curriculum_progress",
            type="line",
        )
        .add_metric(
            metrics_name="curriculum_progress",
            expr="avg(curriculum_progress{})",
        )
        .end_panel()
        .add_panel(
            name="Final Reward",
            name_en="final_reward",
            type="line",
        )
        .add_metric(
            metrics_name="final_reward",
            expr="avg(final_reward{})",
        )
        .end_panel()
        .add_panel(
            name="Total Loss",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="Value Loss",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="Policy Loss",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="Entropy Loss",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="Sample Reward Mean",
            name_en="sample_reward_mean",
            type="line",
        )
        .add_metric(
            metrics_name="sample_reward_mean",
            expr="avg(sample_reward_mean{})",
        )
        .end_panel()
        .add_panel(
            name="Blocked Moves",
            name_en="blocked_moves",
            type="line",
        )
        .add_metric(
            metrics_name="blocked_moves",
            expr="avg(blocked_moves{})",
        )
        .end_panel()
        .add_panel(
            name="Max Stationary",
            name_en="max_stationary",
            type="line",
        )
        .add_metric(
            metrics_name="max_stationary",
            expr="avg(max_stationary{})",
        )
        .end_panel()
        .add_panel(
            name="Flash Good Uses",
            name_en="flash_good_uses",
            type="line",
        )
        .add_metric(
            metrics_name="flash_good_uses",
            expr="avg(flash_good_uses{})",
        )
        .end_panel()
        .add_panel(
            name="Flash Wasted Uses",
            name_en="flash_wasted_uses",
            type="line",
        )
        .add_metric(
            metrics_name="flash_wasted_uses",
            expr="avg(flash_wasted_uses{})",
        )
        .end_panel()
        .add_panel(
            name="Buff Pickups",
            name_en="buff_pickups",
            type="line",
        )
        .add_metric(
            metrics_name="buff_pickups",
            expr="avg(buff_pickups{})",
        )
        .end_panel()
        .add_panel(
            name="Treasure Pickups",
            name_en="treasure_pickups",
            type="line",
        )
        .add_metric(
            metrics_name="treasure_pickups",
            expr="avg(treasure_pickups{})",
        )
        .end_panel()
        .add_panel(
            name="Speedup Reached",
            name_en="speedup_reached",
            type="line",
        )
        .add_metric(
            metrics_name="speedup_reached",
            expr="avg(speedup_reached{})",
        )
        .end_panel()
        .add_panel(
            name="Post Speedup Steps",
            name_en="post_speedup_steps",
            type="line",
        )
        .add_metric(
            metrics_name="post_speedup_steps",
            expr="avg(post_speedup_steps{})",
        )
        .end_panel()
        .add_panel(
            name="Pre Total Gain",
            name_en="pre_total_gain",
            type="line",
        )
        .add_metric(
            metrics_name="pre_total_gain",
            expr="avg(pre_total_gain{})",
        )
        .end_panel()
        .add_panel(
            name="Post Total Gain",
            name_en="post_total_gain",
            type="line",
        )
        .add_metric(
            metrics_name="post_total_gain",
            expr="avg(post_total_gain{})",
        )
        .end_panel()
        .add_panel(
            name="Val Reward",
            name_en="val_reward",
            type="line",
        )
        .add_metric(
            metrics_name="val_reward",
            expr="avg(val_reward{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
