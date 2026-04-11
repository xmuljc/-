#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
"""

import copy
import os
import time

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf

CHECKPOINT_SAVE_INTERVAL_SECONDS = 600


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        if logger:
            logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for game_data in episode_runner.run_episodes():
            if not game_data:
                continue

            agent.send_sample_data(game_data)
            game_data.clear()

            now = time.time()
            if now - last_save_model_time >= CHECKPOINT_SAVE_INTERVAL_SECONDS:
                checkpoint_id = f"episode-{episode_runner.episode_cnt:08d}"
                agent.save_model(id=checkpoint_id)
                if logger:
                    logger.info(f"periodic checkpoint saved: {checkpoint_id}")
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.last_curriculum_phase = None
        self.latest_global_episode_cnt = None
        self.local_train_episode_cnt = 0
        self.curriculum_stage_id = 1
        self.curriculum_stage_episode_cnt = 0
        self.curriculum_score_ema = 0.0
        self.curriculum_step_score_ema = 0.0
        self.curriculum_post_speedup_steps_ema = 0.0
        self.curriculum_treasure_pickup_ema = 0.0
        self.curriculum_speedup_reached_ema = 0.0
        env_conf = self._get_env_conf(self.usr_conf)
        self.default_monster_interval = int(self._get_conf_value(env_conf, "monster_interval", 300))
        self.default_monster_speedup = int(self._get_conf_value(env_conf, "monster_speedup", 500))
        self.default_monster_speed = float(self._get_conf_value(env_conf, "monster_speed", 1.0))
        self.has_monster_speed = self._has_conf_key(env_conf, "monster_speed")

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.latest_global_episode_cnt = self._extract_global_episode_cnt(training_metrics)
                    self._log("info", f"training_metrics is {training_metrics}")

            next_episode = self.episode_cnt + 1
            episode_usr_conf, curriculum_info = self._build_curriculum_usr_conf(next_episode)
            is_validation = bool(curriculum_info["is_validation"])
            env_obs = self.env.reset(episode_usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            if hasattr(self.agent, "set_curriculum"):
                self.agent.set_curriculum(curriculum_info)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt = next_episode
            done = False
            step = 0
            total_reward = 0.0
            blocked_move_steps = 0
            max_stationary_steps = 0
            total_stuck_penalty = 0.0
            total_revisit_penalty = 0.0
            total_wall_collision_penalty = 0.0
            total_wall_repulsion_penalty = 0.0
            total_flash_bonus = 0.0
            total_survival_reward = 0.0
            total_predictive_danger_penalty = 0.0
            total_direction_consistency_reward = 0.0
            total_final_reward = 0.0
            total_space_control_reward = 0.0
            total_open_space_bonus = 0.0
            total_dead_end_penalty = 0.0
            total_second_monster_penalty = 0.0
            total_encirclement_penalty = 0.0
            total_speedup_buffer_reward = 0.0
            total_post_speedup_survival_reward = 0.0
            min_free_path_sum = 0.0
            max_free_path_sum = 0.0
            flash_good_uses = 0
            flash_wasted_uses = 0
            buff_pickups = 0
            treasure_pickups = 0
            wall_hits = 0
            post_speedup_steps = 0
            speedup_reached = False
            pre_total_gain = 0.0
            post_total_gain = 0.0
            pre_step_gain = 0.0
            post_step_gain = 0.0
            pre_treasure_gain = 0.0
            post_treasure_gain = 0.0
            pre_total_reward = 0.0
            post_total_reward = 0.0
            pre_shaped_reward = 0.0
            post_shaped_reward = 0.0
            prev_step_score = 0.0
            prev_treasure_score = 0.0

            self._log(
                "info",
                f"Episode {self.episode_cnt} start phase:{curriculum_info['phase']} "
                f"split:{curriculum_info['split']} "
                f"curriculum_episode_cnt:{curriculum_info['curriculum_episode_cnt']} "
                f"source:{curriculum_info['episode_source']} "
                f"progress:{curriculum_info['progress']:.3f} "
                f"treasure_count:{curriculum_info['treasure_count']} "
                f"buff_count:{curriculum_info['buff_count']} "
                f"monster_interval:{curriculum_info['monster_interval']} "
                f"monster_speedup:{curriculum_info['monster_speedup']} "
                f"max_step:{curriculum_info['max_step']} "
                f"maps:{curriculum_info['maps']}"
                + (
                    f" monster_speed:{curriculum_info['monster_speed']:.3f}"
                    if curriculum_info["monster_speed"] is not None
                    else ""
                ),
            )

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                _, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
                reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])
                current_total_score = float(next_remain_info.get("total_score", 0.0))
                current_step_score = float(
                    next_remain_info.get(
                        "step_score",
                        max(0.0, current_total_score - 100.0 * treasure_pickups),
                    )
                )
                current_treasure_score = float(
                    next_remain_info.get("treasure_score", 100.0 * treasure_pickups)
                )
                score_delta = float(next_remain_info.get("score_delta", 0.0))
                step_gain = max(0.0, current_step_score - prev_step_score)
                treasure_gain = max(0.0, current_treasure_score - prev_treasure_score)
                official_reward_component = score_delta / 100.0
                shaped_reward_component = float(reward[0]) - official_reward_component
                blocked_move_steps += int(next_remain_info.get("blocked_move", 0.0) > 0.5)
                max_stationary_steps = max(max_stationary_steps, int(next_remain_info.get("stationary_steps", 0)))
                total_stuck_penalty += float(next_remain_info.get("stuck_penalty", 0.0))
                total_revisit_penalty += float(next_remain_info.get("revisit_penalty", 0.0))
                total_wall_collision_penalty += float(next_remain_info.get("wall_collision_penalty", 0.0))
                total_wall_repulsion_penalty += float(next_remain_info.get("wall_repulsion_penalty", 0.0))
                total_flash_bonus += float(next_remain_info.get("flash_bonus", 0.0))
                total_survival_reward += float(next_remain_info.get("survival_reward", 0.0))
                total_predictive_danger_penalty += float(next_remain_info.get("predictive_danger_penalty", 0.0))
                total_direction_consistency_reward += float(
                    next_remain_info.get("direction_consistency_reward", 0.0)
                )
                total_final_reward += float(next_remain_info.get("final_reward", 0.0))
                total_space_control_reward += float(next_remain_info.get("space_control_reward", 0.0))
                total_open_space_bonus += float(next_remain_info.get("open_space_bonus", 0.0))
                total_dead_end_penalty += float(next_remain_info.get("dead_end_penalty", 0.0))
                total_second_monster_penalty += float(next_remain_info.get("second_monster_penalty", 0.0))
                total_encirclement_penalty += float(next_remain_info.get("encirclement_penalty", 0.0))
                total_speedup_buffer_reward += float(next_remain_info.get("speedup_buffer_reward", 0.0))
                total_post_speedup_survival_reward += float(
                    next_remain_info.get("post_speedup_survival_reward", 0.0)
                )
                min_free_path_sum += float(next_remain_info.get("min_free_path", 0.0))
                max_free_path_sum += float(next_remain_info.get("max_free_path", 0.0))
                flash_good_uses += int(next_remain_info.get("flash_escape_bonus", 0.0) > 0.0)
                flash_wasted_uses += int(next_remain_info.get("flash_waste_penalty", 0.0) < 0.0)
                buff_pickups += int(next_remain_info.get("buff_collected", 0.0) > 0.5)
                treasure_pickups += int(next_remain_info.get("treasure_collected", 0.0) > 0.5)
                wall_hits += int(next_remain_info.get("wall_collision_penalty", 0.0) < 0.0)
                is_post_speedup = bool(next_remain_info.get("post_speedup_active", 0.0) > 0.5)
                post_speedup_steps += int(is_post_speedup)
                speedup_reached = speedup_reached or is_post_speedup
                if is_post_speedup:
                    post_total_gain += score_delta
                    post_step_gain += step_gain
                    post_treasure_gain += treasure_gain
                    post_total_reward += float(reward[0])
                    post_shaped_reward += shaped_reward_component
                else:
                    pre_total_gain += score_delta
                    pre_step_gain += step_gain
                    pre_treasure_gain += treasure_gain
                    pre_total_reward += float(reward[0])
                    pre_shaped_reward += shaped_reward_component
                prev_step_score = current_step_score
                prev_treasure_score = current_treasure_score

                if done:
                    if terminated:
                        result_str = "CAUGHT"
                    else:
                        result_str = "SURVIVE"

                    self._log(
                        "info",
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"split:{curriculum_info['split']} "
                        f"result:{result_str} sim_score:{current_total_score:.1f} "
                        f"step_score:{current_step_score:.1f} "
                        f"treasure_score:{current_treasure_score:.1f} "
                        f"train_reward:{total_reward:.3f} "
                        f"phase:{curriculum_info['phase']} "
                        f"speedup_reached:{int(speedup_reached)} "
                        f"pre_total_gain:{pre_total_gain:.1f} "
                        f"post_total_gain:{post_total_gain:.1f} "
                        f"pre_step_gain:{pre_step_gain:.1f} "
                        f"post_step_gain:{post_step_gain:.1f} "
                        f"pre_trea_gain:{pre_treasure_gain:.1f} "
                        f"post_trea_gain:{post_treasure_gain:.1f} "
                        f"pre_reward:{pre_total_reward:.3f} "
                        f"post_reward:{post_total_reward:.3f} "
                        f"pre_shaped:{pre_shaped_reward:.3f} "
                        f"post_shaped:{post_shaped_reward:.3f} "
                        f"blocked_moves:{blocked_move_steps} "
                        f"wall_hits:{wall_hits} "
                        f"max_stationary:{max_stationary_steps} "
                        f"wall_penalty:{total_wall_collision_penalty:.3f} "
                        f"wall_repel:{total_wall_repulsion_penalty:.3f} "
                        f"revisit_penalty:{total_revisit_penalty:.3f} "
                        f"predictive_penalty:{total_predictive_danger_penalty:.3f} "
                        f"space_reward:{total_space_control_reward:.3f} "
                        f"dead_end_penalty:{total_dead_end_penalty:.3f} "
                        f"second_monster_penalty:{total_second_monster_penalty:.3f} "
                        f"encirclement_penalty:{total_encirclement_penalty:.3f} "
                        f"speedup_buffer_reward:{total_speedup_buffer_reward:.3f} "
                        f"post_speedup_reward:{total_post_speedup_survival_reward:.3f} "
                        f"direction_reward:{total_direction_consistency_reward:.3f} "
                        f"survival_reward:{total_survival_reward:.3f} "
                        f"final_reward:{total_final_reward:.3f} "
                        f"post_speedup_steps:{post_speedup_steps} "
                        f"flash_good:{flash_good_uses} "
                        f"flash_waste:{flash_wasted_uses} "
                        f"buff_pickups:{buff_pickups} "
                        f"treasure_pickups:{treasure_pickups}",
                    )

                    self._update_curriculum_state(
                        stage_info=curriculum_info,
                        total_score=current_total_score,
                        step_score=current_step_score,
                        post_speedup_steps=post_speedup_steps,
                        treasure_pickups=treasure_pickups,
                        speedup_reached=speedup_reached,
                    )
                    if not is_validation:
                        self.local_train_episode_cnt = int(
                            max(self.local_train_episode_cnt, curriculum_info["curriculum_episode_cnt"])
                        )

                if not is_validation:
                    frame = SampleData(
                        obs=np.array(obs_data.feature, dtype=np.float32),
                        legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                        act=np.array([act_data.action[0]], dtype=np.float32),
                        reward=reward,
                        done=np.array([float(done)], dtype=np.float32),
                        reward_sum=np.zeros(1, dtype=np.float32),
                        value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                        next_value=np.zeros(1, dtype=np.float32),
                        advantage=np.zeros(1, dtype=np.float32),
                        prob=np.array(act_data.prob, dtype=np.float32),
                    )
                    collector.append(frame)

                if done:
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        if is_validation:
                            monitor_data = {
                                "val_reward": round(total_reward, 4),
                                "val_official_score": round(current_total_score, 4),
                                "val_step_score": round(current_step_score, 4),
                                "val_treasure_score": round(current_treasure_score, 4),
                                "val_episode_steps": step,
                                "val_speedup_reached": int(speedup_reached),
                                "val_post_speedup_steps": post_speedup_steps,
                                "val_pre_total_gain": round(pre_total_gain, 4),
                                "val_post_total_gain": round(post_total_gain, 4),
                                "curriculum_stage": curriculum_info["stage_id"],
                                "curriculum_progress": round(curriculum_info["progress"], 4),
                            }
                        else:
                            monitor_data = {
                                "reward": round(total_reward, 4),
                                "official_score": round(current_total_score, 4),
                                "official_step_score": round(current_step_score, 4),
                                "official_treasure_score": round(current_treasure_score, 4),
                                "episode_steps": step,
                                "episode_cnt": self.episode_cnt,
                                "curriculum_stage": curriculum_info["stage_id"],
                                "curriculum_progress": round(curriculum_info["progress"], 4),
                                "curriculum_score_ema": round(self.curriculum_score_ema or 0.0, 4),
                                "curriculum_step_score_ema": round(self.curriculum_step_score_ema or 0.0, 4),
                                "curriculum_post_speedup_ema": round(
                                    self.curriculum_post_speedup_steps_ema or 0.0, 4
                                ),
                                "blocked_moves": blocked_move_steps,
                                "wall_hits": wall_hits,
                                "max_stationary": max_stationary_steps,
                                "wall_collision_penalty": round(total_wall_collision_penalty, 4),
                                "wall_repulsion_penalty": round(total_wall_repulsion_penalty, 4),
                                "stuck_penalty": round(total_stuck_penalty, 4),
                                "revisit_penalty": round(total_revisit_penalty, 4),
                                "flash_bonus": round(total_flash_bonus, 4),
                                "survival_reward": round(total_survival_reward, 4),
                                "predictive_danger_penalty": round(total_predictive_danger_penalty, 4),
                                "space_control_reward": round(total_space_control_reward, 4),
                                "open_space_bonus": round(total_open_space_bonus, 4),
                                "dead_end_penalty": round(total_dead_end_penalty, 4),
                                "second_monster_penalty": round(total_second_monster_penalty, 4),
                                "encirclement_penalty": round(total_encirclement_penalty, 4),
                                "speedup_buffer_reward": round(total_speedup_buffer_reward, 4),
                                "post_speedup_survival_reward": round(total_post_speedup_survival_reward, 4),
                                "direction_consistency_reward": round(total_direction_consistency_reward, 4),
                                "final_reward": round(total_final_reward, 4),
                                "speedup_reached": int(speedup_reached),
                                "pre_total_gain": round(pre_total_gain, 4),
                                "post_total_gain": round(post_total_gain, 4),
                                "pre_step_gain": round(pre_step_gain, 4),
                                "post_step_gain": round(post_step_gain, 4),
                                "pre_treasure_gain": round(pre_treasure_gain, 4),
                                "post_treasure_gain": round(post_treasure_gain, 4),
                                "pre_total_reward": round(pre_total_reward, 4),
                                "post_total_reward": round(post_total_reward, 4),
                                "pre_shaped_reward": round(pre_shaped_reward, 4),
                                "post_shaped_reward": round(post_shaped_reward, 4),
                                "post_speedup_steps": post_speedup_steps,
                                "flash_good_uses": flash_good_uses,
                                "flash_wasted_uses": flash_wasted_uses,
                                "buff_pickups": buff_pickups,
                                "treasure_pickups": treasure_pickups,
                                "avg_free_path": round(min_free_path_sum / max(step, 1), 4),
                                "avg_max_free_path": round(max_free_path_sum / max(step, 1), 4),
                            }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if is_validation:
                        yield None
                    else:
                        yield sample_process(collector)
                    break

                obs_data = next_obs_data

    def _build_curriculum_usr_conf(self, episode_index):
        try:
            usr_conf = copy.deepcopy(self.usr_conf)
        except Exception:
            usr_conf = self.usr_conf

        env_conf = self._get_env_conf(usr_conf)

        is_validation = episode_index % int(max(1, Config.VAL_INTERVAL)) == 0
        curriculum_episode_cnt, episode_source = self._get_curriculum_episode_cnt(episode_index, is_validation)
        max_stage_index = self._resolve_curriculum_stage_index(curriculum_episode_cnt)
        self._maybe_advance_curriculum_stage(max_stage_index)
        stage_index = min(max_stage_index, max(0, self.curriculum_stage_id - 1))
        stage_id = stage_index + 1
        phase = Config.CURRICULUM_STAGE_PHASES[stage_index]
        progress = self._compute_curriculum_progress(stage_index)
        split = "val" if is_validation else "train"
        maps = list(Config.VAL_MAPS if is_validation else Config.TRAIN_MAPS)
        treasure_count = self._sample_range(Config.CURRICULUM_STAGE_TREASURE_RANGE[stage_index])
        buff_count = self._sample_range(Config.CURRICULUM_STAGE_BUFF_RANGE[stage_index])
        monster_interval = self._sample_range(Config.CURRICULUM_STAGE_MONSTER_INTERVAL_RANGE[stage_index])
        monster_speedup = self._sample_range(Config.CURRICULUM_STAGE_MONSTER_SPEEDUP_RANGE[stage_index])
        max_step = int(Config.CURRICULUM_STAGE_MAX_STEP[stage_index])
        monster_speed = self.default_monster_speed if self.has_monster_speed else None

        self._set_conf_value(env_conf, "map", maps)
        self._set_conf_value(env_conf, "map_random", True)
        self._set_conf_value(env_conf, "treasure_count", treasure_count)
        self._set_conf_value(env_conf, "buff_count", buff_count)
        self._set_conf_value(env_conf, "monster_interval", monster_interval)
        self._set_conf_value(env_conf, "monster_speedup", monster_speedup)
        self._set_conf_value(env_conf, "max_step", max_step)
        if monster_speed is not None:
            self._set_conf_value(env_conf, "monster_speed", monster_speed)

        if phase != self.last_curriculum_phase:
            self._log(
                "info",
                f"[curriculum] switch_to:{phase} episode:{episode_index} "
                f"curriculum_episode_cnt:{curriculum_episode_cnt} "
                f"source:{episode_source} "
                f"progress:{progress:.3f} monster_interval:{monster_interval} "
                f"monster_speedup:{monster_speedup} "
                f"score_ema:{(self.curriculum_score_ema or 0.0):.1f} "
                f"step_score_ema:{(self.curriculum_step_score_ema or 0.0):.1f} "
                f"post_speedup_ema:{(self.curriculum_post_speedup_steps_ema or 0.0):.1f} "
                f"treasure_pickup_ema:{(self.curriculum_treasure_pickup_ema or 0.0):.3f} "
                f"speedup_reached_ema:{(self.curriculum_speedup_reached_ema or 0.0):.3f}"
                + (f" monster_speed:{monster_speed:.3f}" if monster_speed is not None else ""),
            )
            self.last_curriculum_phase = phase

        return usr_conf, {
            "phase": phase,
            "split": split,
            "is_validation": is_validation,
            "stage_id": stage_id,
            "progress": progress,
            "curriculum_episode_cnt": curriculum_episode_cnt,
            "episode_source": episode_source,
            "maps": maps,
            "treasure_count": treasure_count,
            "buff_count": buff_count,
            "monster_interval": monster_interval,
            "monster_speedup": monster_speedup,
            "max_step": max_step,
            "monster_speed": monster_speed,
            "curriculum_score_ema": self.curriculum_score_ema,
            "curriculum_step_score_ema": self.curriculum_step_score_ema,
            "curriculum_post_speedup_steps_ema": self.curriculum_post_speedup_steps_ema,
            "curriculum_treasure_pickup_ema": self.curriculum_treasure_pickup_ema,
            "curriculum_speedup_reached_ema": self.curriculum_speedup_reached_ema,
        }

    def _get_env_conf(self, usr_conf):
        if isinstance(usr_conf, dict):
            env_conf = usr_conf.get("env_conf")
            if env_conf is not None:
                return env_conf
        env_conf = getattr(usr_conf, "env_conf", None)
        if env_conf is not None:
            return env_conf
        return usr_conf

    def _get_conf_value(self, conf, key, default):
        if isinstance(conf, dict):
            return conf.get(key, default)
        return getattr(conf, key, default)

    def _has_conf_key(self, conf, key):
        if isinstance(conf, dict):
            return key in conf
        return hasattr(conf, key)

    def _set_conf_value(self, conf, key, value):
        if isinstance(conf, dict):
            conf[key] = value
        else:
            setattr(conf, key, value)

    def _extract_global_episode_cnt(self, training_metrics):
        basic_metrics = self._get_conf_value(training_metrics, "basic", None)
        if basic_metrics is None:
            return None
        episode_cnt = self._get_conf_value(basic_metrics, "episode_cnt", None)
        if episode_cnt is None:
            return None
        try:
            return int(float(episode_cnt))
        except (TypeError, ValueError):
            return None

    def _get_curriculum_episode_cnt(self, episode_index, is_validation):
        if (
            getattr(Config, "CURRICULUM_USE_GLOBAL_EPISODE", False)
            and self.latest_global_episode_cnt is not None
            and self.latest_global_episode_cnt > 0
        ):
            return self.latest_global_episode_cnt, "global_training_metrics"

        local_train_episode_cnt = self.local_train_episode_cnt + (0 if is_validation else 1)
        if local_train_episode_cnt <= 0:
            local_train_episode_cnt = int(max(1, episode_index))
        return int(local_train_episode_cnt), "local_train_episode"

    def _resolve_curriculum_stage_index(self, curriculum_episode_cnt):
        stage_end_episodes = tuple(int(value) for value in Config.CURRICULUM_STAGE_END_EPISODES)
        for idx, stage_end in enumerate(stage_end_episodes):
            if curriculum_episode_cnt <= stage_end:
                return idx
        return len(Config.CURRICULUM_STAGE_PHASES) - 1

    def _compute_curriculum_progress(self, stage_index):
        stage_progress = tuple(float(value) for value in Config.CURRICULUM_STAGE_PROGRESS)
        lower_bound = 0.0 if stage_index <= 0 else stage_progress[stage_index - 1]
        upper_bound = stage_progress[stage_index]

        stage_end_episodes = tuple(int(value) for value in Config.CURRICULUM_STAGE_END_EPISODES)
        prev_stage_end = 0 if stage_index <= 0 else stage_end_episodes[min(stage_index - 1, len(stage_end_episodes) - 1)]
        if stage_index < len(stage_end_episodes):
            stage_target_episodes = max(1, stage_end_episodes[stage_index] - prev_stage_end)
        else:
            total_episodes = max(int(getattr(Config, "CURRICULUM_TOTAL_EPISODES", stage_end_episodes[-1])), stage_end_episodes[-1])
            stage_target_episodes = max(1, total_episodes - prev_stage_end)

        within_stage_progress = min(1.0, self.curriculum_stage_episode_cnt / float(stage_target_episodes))
        return float(lower_bound + (upper_bound - lower_bound) * within_stage_progress)

    def _sample_range(self, value_range):
        low = int(value_range[0])
        high = int(value_range[-1])
        if low >= high:
            return low
        return int(np.random.randint(low, high + 1))

    def _update_curriculum_state(
        self,
        stage_info,
        total_score,
        step_score,
        post_speedup_steps,
        treasure_pickups,
        speedup_reached,
    ):
        if stage_info["stage_id"] != self.curriculum_stage_id:
            self.curriculum_stage_id = int(stage_info["stage_id"])
            self._reset_curriculum_stage_stats()

        if stage_info.get("is_validation", False):
            return

        alpha = float(getattr(Config, "CURRICULUM_SCORE_EMA_ALPHA", 0.06))
        self.curriculum_stage_episode_cnt += 1
        self.curriculum_score_ema = self._ema_update(self.curriculum_score_ema, float(total_score), alpha)
        self.curriculum_step_score_ema = self._ema_update(self.curriculum_step_score_ema, float(step_score), alpha)
        self.curriculum_post_speedup_steps_ema = self._ema_update(
            self.curriculum_post_speedup_steps_ema,
            float(post_speedup_steps),
            alpha,
        )
        self.curriculum_treasure_pickup_ema = self._ema_update(
            self.curriculum_treasure_pickup_ema,
            float(treasure_pickups > 0),
            alpha,
        )
        self.curriculum_speedup_reached_ema = self._ema_update(
            self.curriculum_speedup_reached_ema,
            float(bool(speedup_reached)),
            alpha,
        )

    def _ema_update(self, old_value, new_value, alpha):
        return float((1.0 - alpha) * old_value + alpha * new_value)

    def _reset_curriculum_stage_stats(self):
        self.curriculum_stage_episode_cnt = 0
        self.curriculum_score_ema = 0.0
        self.curriculum_step_score_ema = 0.0
        self.curriculum_post_speedup_steps_ema = 0.0
        self.curriculum_treasure_pickup_ema = 0.0
        self.curriculum_speedup_reached_ema = 0.0

    def _maybe_advance_curriculum_stage(self, max_allowed_stage_index):
        current_stage_index = max(0, int(self.curriculum_stage_id) - 1)
        while current_stage_index < max_allowed_stage_index:
            if not self._curriculum_stage_ready(current_stage_index):
                break

            previous_phase = Config.CURRICULUM_STAGE_PHASES[current_stage_index]
            score_ema = self.curriculum_score_ema
            step_score_ema = self.curriculum_step_score_ema
            treasure_pickup_ema = self.curriculum_treasure_pickup_ema
            speedup_reached_ema = self.curriculum_speedup_reached_ema
            post_speedup_ema = self.curriculum_post_speedup_steps_ema
            current_stage_index += 1
            self.curriculum_stage_id = current_stage_index + 1
            self._reset_curriculum_stage_stats()
            next_phase = Config.CURRICULUM_STAGE_PHASES[current_stage_index]
            self._log(
                "info",
                f"[curriculum_gate] advance_from:{previous_phase} to:{next_phase} "
                f"score_ema:{score_ema:.1f} "
                f"step_score_ema:{step_score_ema:.1f} "
                f"treasure_pickup_ema:{treasure_pickup_ema:.3f} "
                f"speedup_reached_ema:{speedup_reached_ema:.3f} "
                f"post_speedup_ema:{post_speedup_ema:.1f}",
            )

    def _curriculum_stage_ready(self, stage_index):
        if stage_index >= len(Config.CURRICULUM_STAGE_PHASES) - 1:
            return False

        min_episodes = int(Config.CURRICULUM_STAGE_MIN_EPISODES[stage_index])
        score_threshold = float(Config.CURRICULUM_STAGE_SCORE_THRESHOLDS[stage_index])
        step_score_threshold = float(Config.CURRICULUM_STAGE_STEP_SCORE_THRESHOLDS[stage_index])
        treasure_pickup_threshold = float(Config.CURRICULUM_STAGE_TREASURE_PICKUP_THRESHOLDS[stage_index])
        speedup_reached_threshold = float(Config.CURRICULUM_STAGE_SPEEDUP_REACHED_THRESHOLDS[stage_index])
        post_speedup_steps_threshold = float(Config.CURRICULUM_STAGE_POST_SPEEDUP_STEPS_THRESHOLDS[stage_index])

        if self.curriculum_stage_episode_cnt < min_episodes:
            return False
        if self.curriculum_score_ema < score_threshold:
            return False
        if self.curriculum_step_score_ema < step_score_threshold:
            return False
        if self.curriculum_treasure_pickup_ema < treasure_pickup_threshold:
            return False
        if self.curriculum_speedup_reached_ema < speedup_reached_threshold:
            return False
        if self.curriculum_post_speedup_steps_ema < post_speedup_steps_threshold:
            return False
        return True

    def _interp_curriculum_value(self, schedule, progress):
        progress_knots = getattr(Config, "CURRICULUM_PROGRESS_KNOTS", (0.0, 1.0))
        if not schedule:
            return 0.0
        if len(schedule) != len(progress_knots):
            return float(schedule[-1])

        clamped_progress = float(min(1.0, max(0.0, progress)))
        for idx in range(len(progress_knots) - 1):
            left_progress = float(progress_knots[idx])
            right_progress = float(progress_knots[idx + 1])
            if clamped_progress <= right_progress or idx == len(progress_knots) - 2:
                left_value = float(schedule[idx])
                right_value = float(schedule[idx + 1])
                span = max(1e-6, right_progress - left_progress)
                ratio = min(1.0, max(0.0, (clamped_progress - left_progress) / span))
                return left_value + (right_value - left_value) * ratio
        return float(schedule[-1])

    def _log(self, level, message):
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if log_fn is None:
            self.logger.info(message)
            return
        log_fn(message)
