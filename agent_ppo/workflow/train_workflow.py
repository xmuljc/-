#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
"""

import copy
import os
import time

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def _conf_get(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _conf_set(container, key, value):
    if isinstance(container, dict):
        container[key] = value
        return
    setattr(container, key, value)


def _env_conf_view(usr_conf):
    env_conf = _conf_get(usr_conf, "env_conf", None)
    return env_conf if env_conf is not None else usr_conf


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_as_int(v) for v in value]
    return [_as_int(value)]


def _extract_scores(env_obs):
    observation = env_obs.get("observation", {})
    env_info = observation.get("env_info", {})
    return {
        "step_score": float(env_info.get("step_score", 0.0)),
        "treasure_score": float(env_info.get("treasure_score", 0.0)),
        "total_score": float(env_info.get("total_score", 0.0)),
    }


def _build_episode_split(usr_conf):
    train_conf = copy.deepcopy(usr_conf)
    train_env_conf = _env_conf_view(train_conf)

    train_maps = _as_int_list(_conf_get(train_env_conf, "map", []))
    val_maps = _as_int_list(_conf_get(train_env_conf, "val_map", []))
    val_interval = max(0, _as_int(_conf_get(train_env_conf, "val_interval", 0), 0))
    val_episodes = max(1, _as_int(_conf_get(train_env_conf, "val_episodes", 1), 1))

    if train_maps:
        _conf_set(train_env_conf, "map", train_maps)

    val_conf = None
    if val_maps and val_interval > 0:
        val_conf = copy.deepcopy(usr_conf)
        val_env_conf = _env_conf_view(val_conf)
        _conf_set(val_env_conf, "map", val_maps)
    else:
        val_maps = []
        val_interval = 0
        val_episodes = 0

    return train_conf, val_conf, train_maps, val_maps, val_interval, val_episodes


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
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
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.monitor = monitor

        (
            self.train_usr_conf,
            self.val_usr_conf,
            self.train_maps,
            self.val_maps,
            self.val_interval,
            self.val_episodes,
        ) = _build_episode_split(usr_conf)

        self.train_episode_cnt = 0
        self.val_round_cnt = 0
        self.last_val_trigger_episode = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

        if self.val_maps:
            self.logger.info(
                f"train/val split enabled: train_maps={self.train_maps} "
                f"val_maps={self.val_maps} val_interval={self.val_interval} "
                f"val_episodes={self.val_episodes}"
            )
        else:
            self.logger.info(f"validation disabled, train_maps={self.train_maps}")

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            if self._should_run_validation():
                self._run_validation_cycle()

            collector = self._run_train_episode()
            if collector:
                yield collector

    def _should_run_validation(self):
        return (
            self.val_usr_conf is not None
            and self.val_interval > 0
            and self.train_episode_cnt > 0
            and self.train_episode_cnt % self.val_interval == 0
            and self.last_val_trigger_episode != self.train_episode_cnt
        )

    def _reset_episode(self, episode_conf):
        env_obs = self.env.reset(episode_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return None

        self.agent.reset(env_obs)
        self.agent.load_model(id="latest")
        return env_obs

    def _run_train_episode(self):
        env_obs = self._reset_episode(self.train_usr_conf)
        if env_obs is None:
            return None

        obs_data, _ = self.agent.observation_process(env_obs)

        collector = []
        self.train_episode_cnt += 1
        done = False
        step = 0
        total_reward = 0.0

        self.logger.info(
            f"[TRAIN] episode:{self.train_episode_cnt} start maps:{self.train_maps}"
        )

        while not done:
            act_data = self.agent.predict(list_obs_data=[obs_data])[0]
            act = self.agent.action_process(act_data)

            _, env_obs = self.env.step(act)
            if handle_disaster_recovery(env_obs, self.logger):
                return None

            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            step += 1
            done = terminated or truncated

            next_obs_data, next_remain_info = self.agent.observation_process(env_obs)

            reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)
            total_reward += float(reward[0])

            final_reward = np.zeros(1, dtype=np.float32)
            final_scores = None
            result_str = "RUNNING"
            if done:
                final_scores = _extract_scores(env_obs)
                max_step = _conf_get(_env_conf_view(self.train_usr_conf), "max_step", 0)
                if terminated:
                    final_reward[0] = -1.2
                    result_str = "FAIL"
                elif truncated and step >= _as_int(max_step, 0):
                    final_reward[0] = 0.8
                    result_str = "WIN"
                else:
                    final_reward[0] = 0.0
                    result_str = "TRUNC"

                self.logger.info(
                    f"[GAMEOVER][TRAIN] episode:{self.train_episode_cnt} steps:{step} "
                    f"result:{result_str} step_score:{final_scores['step_score']:.1f} "
                    f"treasure_score:{final_scores['treasure_score']:.1f} "
                    f"sim_score:{final_scores['total_score']:.1f} "
                    f"total_reward:{total_reward:.3f}"
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
                if collector:
                    collector[-1].reward = collector[-1].reward + final_reward

                if final_scores is not None:
                    self._report_train_monitor(
                        step=step,
                        total_reward=total_reward + float(final_reward[0]),
                        scores=final_scores,
                    )

                if collector:
                    return sample_process(collector)
                return None

            obs_data = next_obs_data

        return None

    def _run_validation_cycle(self):
        self.last_val_trigger_episode = self.train_episode_cnt
        self.val_round_cnt += 1
        stats_list = []

        for val_idx in range(1, self.val_episodes + 1):
            stats = self._run_val_episode(val_idx)
            if stats is not None:
                stats_list.append(stats)

        if not stats_list:
            self.logger.warning(
                f"[VAL] round:{self.val_round_cnt} after_train_episode:{self.train_episode_cnt} "
                "finished with no valid episodes"
            )
            return

        val_score = float(np.mean([item["total_score"] for item in stats_list]))
        val_step_score = float(np.mean([item["step_score"] for item in stats_list]))
        val_treasure_score = float(np.mean([item["treasure_score"] for item in stats_list]))
        val_win_rate = float(np.mean([item["win"] for item in stats_list]))

        self.logger.info(
            f"[VAL][SUMMARY] round:{self.val_round_cnt} "
            f"after_train_episode:{self.train_episode_cnt} "
            f"episodes:{len(stats_list)} val_score:{val_score:.1f} "
            f"val_step_score:{val_step_score:.1f} "
            f"val_treasure_score:{val_treasure_score:.1f} "
            f"val_win_rate:{val_win_rate:.2f}"
        )

    def _run_val_episode(self, val_idx):
        env_obs = self._reset_episode(self.val_usr_conf)
        if env_obs is None:
            return None

        done = False
        step = 0

        self.logger.info(
            f"[VAL] round:{self.val_round_cnt} episode:{val_idx}/{self.val_episodes} "
            f"after_train_episode:{self.train_episode_cnt} maps:{self.val_maps}"
        )

        while not done:
            act = self.agent.exploit(env_obs)
            _, env_obs = self.env.step(act)
            if handle_disaster_recovery(env_obs, self.logger):
                return None

            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            step += 1
            done = terminated or truncated

        scores = _extract_scores(env_obs)
        result_str = "FAIL" if env_obs["terminated"] else "WIN"
        self.logger.info(
            f"[GAMEOVER][VAL] round:{self.val_round_cnt} "
            f"episode:{val_idx}/{self.val_episodes} after_train_episode:{self.train_episode_cnt} "
            f"steps:{step} result:{result_str} "
            f"step_score:{scores['step_score']:.1f} "
            f"treasure_score:{scores['treasure_score']:.1f} "
            f"sim_score:{scores['total_score']:.1f}"
        )

        return {
            "step_score": scores["step_score"],
            "treasure_score": scores["treasure_score"],
            "total_score": scores["total_score"],
            "win": 0.0 if env_obs["terminated"] else 1.0,
        }

    def _report_train_monitor(self, step, total_reward, scores):
        if not self.monitor:
            return

        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return

        monitor_data = {
            "reward": round(total_reward, 4),
            "episode_steps": step,
            "episode_cnt": self.train_episode_cnt,
        }
        self.monitor.put_data({os.getpid(): monitor_data})
        self.last_report_monitor_time = now
