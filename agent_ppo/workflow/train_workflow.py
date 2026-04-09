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
                    self._log("info", f"training_metrics is {training_metrics}")

            next_episode = self.episode_cnt + 1
            episode_usr_conf, curriculum_info = self._build_curriculum_usr_conf(next_episode)
            env_obs = self.env.reset(episode_usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            if hasattr(self.agent, "set_curriculum"):
                self.agent.set_curriculum(
                    curriculum_info["stage_id"],
                )
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
            min_free_path_sum = 0.0
            max_free_path_sum = 0.0
            flash_good_uses = 0
            flash_wasted_uses = 0
            buff_pickups = 0
            treasure_pickups = 0
            wall_hits = 0

            self._log(
                "info",
                f"Episode {self.episode_cnt} start phase:{curriculum_info['phase']} "
                f"progress:{curriculum_info['progress']:.3f} "
                f"monster_interval:{curriculum_info['monster_interval']} "
                f"monster_speedup:{curriculum_info['monster_speedup']}"
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
                min_free_path_sum += float(next_remain_info.get("min_free_path", 0.0))
                max_free_path_sum += float(next_remain_info.get("max_free_path", 0.0))
                flash_good_uses += int(next_remain_info.get("flash_escape_bonus", 0.0) > 0.0)
                flash_wasted_uses += int(next_remain_info.get("flash_waste_penalty", 0.0) < 0.0)
                buff_pickups += int(next_remain_info.get("buff_collected", 0.0) > 0.5)
                treasure_pickups += int(next_remain_info.get("treasure_collected", 0.0) > 0.5)
                wall_hits += int(next_remain_info.get("wall_collision_penalty", 0.0) < 0.0)

                if done:
                    if terminated:
                        result_str = "CAUGHT"
                    else:
                        result_str = "SURVIVE"

                    self._log(
                        "info",
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{current_total_score:.1f} "
                        f"train_reward:{total_reward:.3f} "
                        f"phase:{curriculum_info['phase']} "
                        f"blocked_moves:{blocked_move_steps} "
                        f"wall_hits:{wall_hits} "
                        f"max_stationary:{max_stationary_steps} "
                        f"wall_penalty:{total_wall_collision_penalty:.3f} "
                        f"wall_repel:{total_wall_repulsion_penalty:.3f} "
                        f"revisit_penalty:{total_revisit_penalty:.3f} "
                        f"predictive_penalty:{total_predictive_danger_penalty:.3f} "
                        f"direction_reward:{total_direction_consistency_reward:.3f} "
                        f"survival_reward:{total_survival_reward:.3f} "
                        f"final_reward:{total_final_reward:.3f} "
                        f"flash_good:{flash_good_uses} "
                        f"flash_waste:{flash_wasted_uses} "
                        f"buff_pickups:{buff_pickups} "
                        f"treasure_pickups:{treasure_pickups}",
                    )

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
                        monitor_data = {
                            "reward": round(total_reward, 4),
                            "official_score": round(current_total_score, 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "curriculum_stage": curriculum_info["stage_id"],
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
                            "direction_consistency_reward": round(total_direction_consistency_reward, 4),
                            "final_reward": round(total_final_reward, 4),
                            "flash_good_uses": flash_good_uses,
                            "flash_wasted_uses": flash_wasted_uses,
                            "buff_pickups": buff_pickups,
                            "treasure_pickups": treasure_pickups,
                            "avg_free_path": round(min_free_path_sum / max(step, 1), 4),
                            "avg_max_free_path": round(max_free_path_sum / max(step, 1), 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    yield sample_process(collector)
                    break

                obs_data = next_obs_data

    def _build_curriculum_usr_conf(self, episode_index):
        try:
            usr_conf = copy.deepcopy(self.usr_conf)
        except Exception:
            usr_conf = self.usr_conf

        env_conf = self._get_env_conf(usr_conf)

        progress = min(1.0, max(0.0, episode_index / float(max(1, Config.CURRICULUM_TOTAL_EPISODES))))
        if progress < Config.CURRICULUM_STAGE1_RATIO:
            phase = "early"
            stage_id = 1
            monster_speed_scale = Config.CURRICULUM_STAGE1_MONSTER_SPEED_SCALE
            monster_interval_scale = Config.CURRICULUM_STAGE1_MONSTER_INTERVAL_SCALE
        elif progress < Config.CURRICULUM_STAGE2_RATIO:
            phase = "mid"
            stage_id = 2
            monster_speed_scale = Config.CURRICULUM_STAGE2_MONSTER_SPEED_SCALE
            monster_interval_scale = Config.CURRICULUM_STAGE2_MONSTER_INTERVAL_SCALE
        else:
            phase = "late"
            stage_id = 3
            monster_speed_scale = Config.CURRICULUM_STAGE3_MONSTER_SPEED_SCALE
            monster_interval_scale = Config.CURRICULUM_STAGE3_MONSTER_INTERVAL_SCALE

        monster_interval = min(2000, max(11, int(round(self.default_monster_interval * monster_interval_scale))))
        monster_speedup = min(
            2000,
            max(1, int(round(self.default_monster_speedup / max(monster_speed_scale, 1e-6)))),
        )
        monster_speed = self.default_monster_speed * monster_speed_scale if self.has_monster_speed else None

        self._set_conf_value(env_conf, "monster_interval", monster_interval)
        self._set_conf_value(env_conf, "monster_speedup", monster_speedup)
        if monster_speed is not None:
            self._set_conf_value(env_conf, "monster_speed", monster_speed)

        if phase != self.last_curriculum_phase:
            self._log(
                "info",
                f"[curriculum] switch_to:{phase} episode:{episode_index} "
                f"progress:{progress:.3f} monster_interval:{monster_interval} "
                f"monster_speedup:{monster_speedup}"
                + (f" monster_speed:{monster_speed:.3f}" if monster_speed is not None else ""),
            )
            self.last_curriculum_phase = phase

        return usr_conf, {
            "phase": phase,
            "stage_id": stage_id,
            "progress": progress,
            "monster_interval": monster_interval,
            "monster_speedup": monster_speedup,
            "monster_speed": monster_speed,
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

    def _log(self, level, message):
        if self.logger is None:
            return
        log_fn = getattr(self.logger, level, None)
        if log_fn is None:
            self.logger.info(message)
            return
        log_fn(message)
