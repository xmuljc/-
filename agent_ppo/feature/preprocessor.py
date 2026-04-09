#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
"""

import numpy as np

from agent_ppo.conf.conf import Config

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
MAX_VISIBLE_TREASURE = 10.0
MAX_VISIBLE_BUFF = 2.0
MAX_OFFICIAL_SCORE = 2500.0
LOCAL_MAP_SIZE = 21
LOCAL_MAP_HALF = LOCAL_MAP_SIZE // 2
MAX_MOVE_DELTA = 10.0
MAX_STATIONARY_STEPS = 8.0
STATIONARY_EPS = 0.1
FLASH_DANGER_DIST_NORM = 0.14
BUFF_DANGER_DIST_NORM = 0.18
TREASURE_SAFE_DIST_NORM = 0.30
TREASURE_SCORE_DELTA_THRESHOLD = 50.0
POST_FLASH_BONUS_STEPS = 5
DIRECTION_HISTORY_WINDOW = 4
DIRECTION_HISTORY_MIN_SAMPLES = 3
DIRECTION_CONSISTENCY_REWARD_MAX = 0.0
DIRECTION_CONSISTENCY_COS_THRESHOLD = 0.5

DIRECTION_OFFSETS = (
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _signed_norm(v, v_abs_max):
    if v_abs_max <= 1e-6:
        return 0.0
    return float(np.clip(v / v_abs_max, -1.0, 1.0))


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _extract_pos(entity):
    if not isinstance(entity, dict):
        return None

    pos = entity.get("pos") or entity.get("position")
    if isinstance(pos, dict) and "x" in pos and "z" in pos:
        return {"x": float(pos["x"]), "z": float(pos["z"])}

    if "x" in entity and "z" in entity:
        return {"x": float(entity["x"]), "z": float(entity["z"])}

    return None


def _entity_is_active(entity):
    if not isinstance(entity, dict):
        return True
    for key in ("is_in_view", "visible", "active", "exist", "exists"):
        if key in entity:
            return bool(entity[key])
    return True


def _flatten_entities(value):
    if isinstance(value, dict):
        if _extract_pos(value) is not None:
            return [value]
        entities = []
        for child in value.values():
            entities.extend(_flatten_entities(child))
        return entities

    if isinstance(value, (list, tuple)):
        entities = []
        for child in value:
            entities.extend(_flatten_entities(child))
        return entities

    return []


def _find_entities(frame_state, key_candidates, substring_candidates):
    for key in key_candidates:
        if key in frame_state:
            return _flatten_entities(frame_state[key])

    entities = []
    for key, value in frame_state.items():
        key_lower = str(key).lower()
        if any(token in key_lower for token in substring_candidates):
            entities.extend(_flatten_entities(value))
    return entities


def _distance(hero_pos, target_pos):
    return float(np.sqrt((hero_pos["x"] - target_pos["x"]) ** 2 + (hero_pos["z"] - target_pos["z"]) ** 2))


def _nearest_target_feature(entities, hero_pos):
    active_entities = []
    for entity in entities:
        if not _entity_is_active(entity):
            continue
        pos = _extract_pos(entity)
        if pos is None:
            continue
        active_entities.append(pos)

    if not active_entities:
        return np.zeros(4, dtype=np.float32), 1.0, 0

    distances = [(target_pos, _distance(hero_pos, target_pos)) for target_pos in active_entities]
    target_pos, dist = min(distances, key=lambda item: item[1])
    feature = np.array(
        [
            1.0,
            _signed_norm(target_pos["x"] - hero_pos["x"], MAP_SIZE),
            _signed_norm(target_pos["z"] - hero_pos["z"], MAP_SIZE),
            _norm(dist, MAP_SIZE * 1.42),
        ],
        dtype=np.float32,
    )
    return feature, feature[3], len(active_entities)


def _extract_centered_map(map_info):
    local_map = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE), dtype=np.float32)
    if map_info is None:
        return local_map

    try:
        map_array = np.asarray(map_info, dtype=np.float32)
    except Exception:
        return local_map

    if map_array.ndim != 2:
        return local_map

    src_h, src_w = map_array.shape
    src_center_h = src_h // 2
    src_center_w = src_w // 2
    dst_half = LOCAL_MAP_HALF

    for dst_r in range(LOCAL_MAP_SIZE):
        src_r = src_center_h - dst_half + dst_r
        if src_r < 0 or src_r >= src_h:
            continue
        for dst_c in range(LOCAL_MAP_SIZE):
            src_c = src_center_w - dst_half + dst_c
            if src_c < 0 or src_c >= src_w:
                continue
            local_map[dst_r, dst_c] = map_array[src_r, src_c]

    return local_map


def _extract_local_map_feature(local_map):
    normalized_map = np.clip(local_map, -5.0, 5.0) / 5.0
    return normalized_map.astype(np.float32).flatten()


def _build_obstacle_mask(local_map, move_legal_action):
    center = LOCAL_MAP_HALF
    passable_values = [local_map[center, center]]

    for idx, (dr, dc) in enumerate(DIRECTION_OFFSETS):
        if idx >= len(move_legal_action) or move_legal_action[idx] <= 0:
            continue

        row = center + dr
        col = center + dc
        if 0 <= row < LOCAL_MAP_SIZE and 0 <= col < LOCAL_MAP_SIZE:
            passable_values.append(local_map[row, col])

        if dr != 0 and 0 <= center + dr < LOCAL_MAP_SIZE:
            passable_values.append(local_map[center + dr, center])
        if dc != 0 and 0 <= center + dc < LOCAL_MAP_SIZE:
            passable_values.append(local_map[center, center + dc])

    candidate_values = np.unique(np.round(np.asarray(passable_values, dtype=np.float32), 4))
    blocked_mask = np.ones_like(local_map, dtype=bool)
    for value in candidate_values:
        blocked_mask &= np.abs(local_map - value) > 1e-4

    blocked_mask[center, center] = False
    return blocked_mask


def _local_blocked_ratio(blocked_mask, radius):
    center = LOCAL_MAP_HALF
    row_start = max(0, center - radius)
    row_end = min(LOCAL_MAP_SIZE, center + radius + 1)
    col_start = max(0, center - radius)
    col_end = min(LOCAL_MAP_SIZE, center + radius + 1)

    window = blocked_mask[row_start:row_end, col_start:col_end].astype(np.float32).flatten()
    if window.size == 0:
        return 0.0

    center_idx = (center - row_start) * (col_end - col_start) + (center - col_start)
    if 0 <= center_idx < window.size:
        window = np.delete(window, center_idx)
    return float(window.mean()) if window.size > 0 else 0.0


def _extract_obstacle_feature(local_map, legal_action):
    move_legal_action = np.asarray(legal_action[:8], dtype=np.float32)
    blocked_mask = _build_obstacle_mask(local_map, move_legal_action)

    directional_free_space = []
    for dr, dc in DIRECTION_OFFSETS:
        free_steps = 0
        for step in range(1, LOCAL_MAP_HALF + 1):
            row = LOCAL_MAP_HALF + dr * step
            col = LOCAL_MAP_HALF + dc * step
            if row < 0 or row >= LOCAL_MAP_SIZE or col < 0 or col >= LOCAL_MAP_SIZE:
                break
            if blocked_mask[row, col]:
                break
            free_steps = step
        directional_free_space.append(free_steps / float(LOCAL_MAP_HALF))

    obstacle_summary = np.array(
        [
            _local_blocked_ratio(blocked_mask, radius=1),
            _local_blocked_ratio(blocked_mask, radius=2),
            _local_blocked_ratio(blocked_mask, radius=4),
            float(move_legal_action.mean()) if move_legal_action.size > 0 else 0.0,
        ],
        dtype=np.float32,
    )
    obstacle_feature = np.concatenate(
        [
            np.asarray(directional_free_space, dtype=np.float32),
            obstacle_summary,
        ]
    )
    return obstacle_feature, blocked_mask


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.curriculum_stage_id = 1
        self.last_min_monster_dist_norm = 0.5
        self.last_treasure_dist_norm = 1.0
        self.last_buff_dist_norm = 1.0
        self.last_total_score = 0.0
        self.last_hero_pos = None
        self.stationary_steps = 0
        self.repeat_blocked_steps = 0
        self.prev_action = -1
        self.last_visible_monster_count = 0
        self.last_buff_active = False
        self.pending_flash_survival_steps = 0
        self.pending_flash_survival_reward = 0.0
        self.last_monster_positions = [None, None]
        self.treasure_pickup_count = 0
        self.buff_pickup_count = 0
        self.recent_move_directions = []
        self.recent_positions = []

    def set_curriculum(self, stage_id):
        self.curriculum_stage_id = int(max(1, stage_id))

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        hero = frame_state["heroes"]
        hero_pos = _extract_pos(hero) or {"x": 0.0, "z": 0.0}
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd = float(hero.get("flash_cooldown", MAX_FLASH_CD))
        flash_cd_norm = _norm(flash_cd, MAX_FLASH_CD)
        buff_remain = float(hero.get("buff_remaining_time", 0.0))
        buff_remain_norm = _norm(buff_remain, MAX_BUFF_DURATION)
        hero_speed = float(hero.get("speed", hero.get("move_speed", 1.0)))
        hero_speed_norm = _norm(hero_speed, 2.0, 1.0)
        hero_feat = np.array(
            [
                hero_x_norm,
                hero_z_norm,
                flash_cd_norm,
                buff_remain_norm,
                hero_speed_norm,
            ],
            dtype=np.float32,
        )

        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_monster_dist_norm = 1.0
        visible_monster_count = 0
        current_monster_positions = [None, None]
        for i in range(2):
            if i < len(monsters):
                monster = monsters[i]
                is_in_view = float(monster.get("is_in_view", 0))
                monster_pos = _extract_pos(monster) or {"x": 0.0, "z": 0.0}
                if is_in_view:
                    visible_monster_count += 1
                    current_monster_positions[i] = {"x": monster_pos["x"], "z": monster_pos["z"]}
                    raw_dist = _distance(hero_pos, monster_pos)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.42)
                    cur_min_monster_dist_norm = min(cur_min_monster_dist_norm, dist_norm)
                    monster_feat = np.array(
                        [
                            is_in_view,
                            _signed_norm(monster_pos["x"] - hero_pos["x"], MAP_SIZE),
                            _signed_norm(monster_pos["z"] - hero_pos["z"], MAP_SIZE),
                            _norm(monster.get("speed", 1.0), MAX_MONSTER_SPEED),
                            dist_norm,
                        ],
                        dtype=np.float32,
                    )
                else:
                    monster_feat = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            else:
                monster_feat = np.zeros(5, dtype=np.float32)
                monster_feat[-1] = 1.0
            monster_feats.append(monster_feat)

        treasure_entities = _find_entities(
            frame_state,
            key_candidates=("treasures", "treasure"),
            substring_candidates=("treasure", "chest", "box"),
        )
        buff_entities = _find_entities(
            frame_state,
            key_candidates=("buffs", "speed_buffs", "speed_buff"),
            substring_candidates=("buff",),
        )
        treasure_feat, treasure_dist_norm, visible_treasure_count = _nearest_target_feature(treasure_entities, hero_pos)
        buff_feat, buff_dist_norm, visible_buff_count = _nearest_target_feature(buff_entities, hero_pos)

        legal_action = self._build_legal_action_mask(legal_act_raw, flash_cd)
        local_map = _extract_centered_map(map_info)
        obstacle_feat, _ = _extract_obstacle_feature(local_map, legal_action)
        local_map_feat = _extract_local_map_feature(local_map)
        min_free_path = float(np.min(obstacle_feat[:8])) if obstacle_feat.size >= 8 else 0.0
        max_free_path = float(np.max(obstacle_feat[:8])) if obstacle_feat.size >= 8 else 0.0
        local_blocked_ratio = float(obstacle_feat[9]) if obstacle_feat.size >= 10 else 0.0
        move_legal_count = int(np.count_nonzero(np.asarray(legal_action[:8], dtype=np.float32) > 0.5))

        move_dx = 0.0
        move_dz = 0.0
        move_dist = 0.0
        blocked_move = 0.0
        current_move_direction = None
        if self.last_hero_pos is not None:
            move_dx = hero_pos["x"] - self.last_hero_pos["x"]
            move_dz = hero_pos["z"] - self.last_hero_pos["z"]
            move_dist = float(np.sqrt(move_dx**2 + move_dz**2))
            if move_dist <= STATIONARY_EPS:
                self.stationary_steps += 1
                blocked_move = float(last_action >= 0)
                if last_action >= 0 and last_action == self.prev_action:
                    self.repeat_blocked_steps += 1
                else:
                    self.repeat_blocked_steps = 1 if last_action >= 0 else 0
            else:
                self.stationary_steps = 0
                self.repeat_blocked_steps = 0
                current_move_direction = np.array([move_dx, move_dz], dtype=np.float32) / max(move_dist, 1e-6)
        else:
            self.stationary_steps = 0
            self.repeat_blocked_steps = 0

        motion_feat = np.array(
            [
                _signed_norm(move_dx, MAX_MOVE_DELTA),
                _signed_norm(move_dz, MAX_MOVE_DELTA),
                _norm(move_dist, MAX_MOVE_DELTA),
                _norm(self.stationary_steps, MAX_STATIONARY_STEPS),
                _norm(self.repeat_blocked_steps, MAX_STATIONARY_STEPS),
                blocked_move,
            ],
            dtype=np.float32,
        )

        current_total_score = float(env_info.get("total_score", 0.0))
        score_delta = max(0.0, current_total_score - self.last_total_score)
        terminated = bool(env_obs.get("terminated", False))
        truncated = bool(env_obs.get("truncated", False))
        official_reward = score_delta / 100.0
        survival_reward = self._compute_survival_reward()
        danger_shaping = Config.DANGER_SHAPING_COEF * (
            cur_min_monster_dist_norm - self.last_min_monster_dist_norm
        )
        safe_treasure_state = visible_monster_count == 0
        treasure_shaping_coef = (
            Config.SAFE_TREASURE_SHAPING_COEF if safe_treasure_state else Config.TREASURE_SHAPING_COEF
        )
        treasure_shaping = treasure_shaping_coef * (self.last_treasure_dist_norm - treasure_dist_norm)
        treasure_collected = float(score_delta >= TREASURE_SCORE_DELTA_THRESHOLD)
        treasure_pickup_bonus = 0.0
        if treasure_collected:
            self.treasure_pickup_count += 1
            treasure_pickup_bonus = Config.TREASURE_PICKUP_BONUS + (
                Config.COLLECTION_STREAK_BONUS
                * min(self.treasure_pickup_count - 1, Config.MAX_COLLECTION_STREAK_BONUS)
            )

        buff_danger_state = visible_monster_count > 0 and cur_min_monster_dist_norm < BUFF_DANGER_DIST_NORM
        buff_shaping_coef = Config.BUFF_DANGER_SHAPING_COEF if buff_danger_state else Config.BUFF_SHAPING_COEF
        buff_shaping = buff_shaping_coef * (self.last_buff_dist_norm - buff_dist_norm) if buff_remain <= 0 else 0.0
        buff_active = buff_remain > 0.0
        buff_collected = float(buff_active and not self.last_buff_active)
        buff_pickup_bonus = 0.0
        if buff_collected:
            self.buff_pickup_count += 1
            buff_pickup_bonus = Config.BUFF_PICKUP_BONUS + (
                Config.COLLECTION_STREAK_BONUS
                * min(self.buff_pickup_count - 1, Config.MAX_COLLECTION_STREAK_BONUS)
            )

        wall_collision_penalty = 0.0
        stuck_penalty = 0.0
        revisit_penalty = 0.0
        revisit_hits = 0
        wall_repulsion_penalty = 0.0
        wall_repulsion_scale = 1.0
        if min_free_path < Config.WALL_REPULSION_THRESHOLD:
            if move_legal_count < (len(DIRECTION_OFFSETS) // 2):
                wall_repulsion_scale = move_legal_count / float(len(DIRECTION_OFFSETS) // 2)
            wall_repulsion_penalty = -Config.WALL_REPULSION_COEF * (
                (Config.WALL_REPULSION_THRESHOLD - min_free_path) / Config.WALL_REPULSION_THRESHOLD
            ) * wall_repulsion_scale
        if self.last_hero_pos is not None and move_dist <= STATIONARY_EPS and last_action >= 0:
            wall_collision_penalty = Config.WALL_COLLISION_PENALTY
            if self.stationary_steps >= 3:
                stuck_penalty -= Config.STUCK_STATIONARY_PENALTY_COEF * min(self.stationary_steps - 2, 6)
            if self.repeat_blocked_steps >= 2:
                stuck_penalty -= Config.STUCK_REPEAT_PENALTY_COEF * min(self.repeat_blocked_steps - 1, 6)
            if buff_danger_state or (visible_monster_count > 0 and cur_min_monster_dist_norm < FLASH_DANGER_DIST_NORM):
                stuck_penalty *= Config.STUCK_DANGER_MULTIPLIER
            stuck_penalty = max(stuck_penalty, -Config.STUCK_PENALTY_CAP)

        confined_state = (
            move_legal_count <= 4
            or min_free_path < Config.WALL_REPULSION_THRESHOLD
            or local_blocked_ratio > 0.35
        )
        if (
            move_dist > STATIONARY_EPS
            and score_delta <= 0.0
            and visible_monster_count == 0
            and self.recent_positions
        ):
            for prev_pos in self.recent_positions:
                if _distance(hero_pos, prev_pos) <= Config.LOOP_REVISIT_DISTANCE:
                    revisit_hits += 1

            if revisit_hits > 0:
                revisit_ratio = revisit_hits / float(max(1, len(self.recent_positions)))
                revisit_penalty = -Config.LOOP_REVISIT_PENALTY_COEF * revisit_ratio
                if confined_state:
                    revisit_penalty *= Config.LOOP_CONFINED_MULTIPLIER
                revisit_penalty = max(revisit_penalty, -Config.LOOP_REVISIT_PENALTY_CAP)

        flash_survival_bonus = 0.0
        if self.pending_flash_survival_steps > 0:
            flash_survival_bonus = self.pending_flash_survival_reward * (
                self.pending_flash_survival_steps / float(POST_FLASH_BONUS_STEPS)
            )
            self.pending_flash_survival_steps -= 1

        flash_escape_bonus = 0.0
        flash_waste_penalty = 0.0
        if self.last_hero_pos is not None and last_action >= 8:
            dist_gain = cur_min_monster_dist_norm - self.last_min_monster_dist_norm
            flash_in_danger = (
                self.last_visible_monster_count > 0 and self.last_min_monster_dist_norm < FLASH_DANGER_DIST_NORM
            )
            flash_without_pressure = self.last_visible_monster_count == 0 and visible_monster_count == 0

            if move_dist <= STATIONARY_EPS:
                flash_waste_penalty += Config.FLASH_WASTE_BLOCKED_PENALTY
            elif flash_in_danger:
                flash_escape_bonus += Config.FLASH_ESCAPE_BONUS_BASE + (
                    Config.FLASH_ESCAPE_BONUS_GAIN_COEF * max(0.0, dist_gain)
                )
                if dist_gain > 0.02:
                    self.pending_flash_survival_steps = POST_FLASH_BONUS_STEPS
                    self.pending_flash_survival_reward = min(
                        Config.FLASH_SURVIVAL_BONUS_MAX,
                        Config.FLASH_SURVIVAL_BONUS_BASE
                        + Config.FLASH_SURVIVAL_BONUS_GAIN_COEF * dist_gain,
                    )
            elif flash_without_pressure:
                flash_waste_penalty += Config.FLASH_WASTE_SAFE_PENALTY
            elif dist_gain <= 0.0:
                flash_waste_penalty += Config.FLASH_WASTE_BAD_GAIN_PENALTY

        flash_reward = flash_escape_bonus + flash_survival_bonus + flash_waste_penalty
        predictive_danger_penalty = self._compute_predictive_monster_penalty(hero_pos, current_monster_positions)
        direction_consistency_reward = 0.0
        direction_consistency_cosine = 0.0
        if current_move_direction is not None and len(self.recent_move_directions) >= DIRECTION_HISTORY_MIN_SAMPLES:
            avg_direction = np.mean(np.asarray(self.recent_move_directions, dtype=np.float32), axis=0)
            avg_direction_norm = float(np.linalg.norm(avg_direction))
            if avg_direction_norm > 1e-6:
                direction_consistency_cosine = float(
                    np.clip(np.dot(current_move_direction, avg_direction / avg_direction_norm), -1.0, 1.0)
                )
                if direction_consistency_cosine > DIRECTION_CONSISTENCY_COS_THRESHOLD:
                    direction_consistency_reward = DIRECTION_CONSISTENCY_REWARD_MAX * (
                        (direction_consistency_cosine - DIRECTION_CONSISTENCY_COS_THRESHOLD)
                        / max(1e-6, 1.0 - DIRECTION_CONSISTENCY_COS_THRESHOLD)
                    )
        final_reward = 0.0
        if terminated:
            final_reward = Config.TERMINAL_CAUGHT_PENALTY
        elif truncated:
            final_reward = Config.TERMINAL_SURVIVE_BONUS

        reward = [
            official_reward
            + danger_shaping
            + wall_collision_penalty
            + final_reward
        ]

        score_norm = _norm(current_total_score, MAX_OFFICIAL_SCORE)
        progress_feat = np.array(
            [
                _norm(self.step_no, self.max_step),
                score_norm,
                float(flash_cd <= 1e-6),
                float(buff_remain > 0.0),
                float(last_action >= 8),
                _norm(visible_treasure_count, MAX_VISIBLE_TREASURE),
                _norm(visible_buff_count, MAX_VISIBLE_BUFF),
            ],
            dtype=np.float32,
        )

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                buff_feat,
                obstacle_feat,
                motion_feat,
                local_map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        self.last_total_score = current_total_score
        self.last_min_monster_dist_norm = cur_min_monster_dist_norm
        self.last_treasure_dist_norm = treasure_dist_norm
        self.last_buff_dist_norm = buff_dist_norm
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
        self.prev_action = last_action
        self.last_visible_monster_count = visible_monster_count
        self.last_buff_active = buff_active
        self.last_monster_positions = current_monster_positions
        if current_move_direction is not None:
            self.recent_move_directions.append(current_move_direction)
            if len(self.recent_move_directions) > DIRECTION_HISTORY_WINDOW:
                self.recent_move_directions.pop(0)
        self.recent_positions.append({"x": hero_pos["x"], "z": hero_pos["z"]})
        if len(self.recent_positions) > Config.LOOP_REVISIT_WINDOW:
            self.recent_positions.pop(0)

        remain_info = {
            "reward": reward,
            "total_score": current_total_score,
            "score_delta": score_delta,
            "move_dist": move_dist,
            "stationary_steps": self.stationary_steps,
            "repeat_blocked_steps": self.repeat_blocked_steps,
            "blocked_move": blocked_move,
            "wall_collision_penalty": wall_collision_penalty,
            "wall_repulsion_penalty": wall_repulsion_penalty,
            "stuck_penalty": stuck_penalty,
            "revisit_penalty": revisit_penalty,
            "revisit_hits": revisit_hits,
            "flash_bonus": flash_reward,
            "flash_escape_bonus": flash_escape_bonus,
            "flash_survival_bonus": flash_survival_bonus,
            "flash_waste_penalty": flash_waste_penalty,
            "danger_shaping": danger_shaping,
            "survival_reward": survival_reward,
            "predictive_danger_penalty": predictive_danger_penalty,
            "final_reward": final_reward,
            "buff_collected": buff_collected,
            "buff_pickup_bonus": buff_pickup_bonus,
            "treasure_collected": treasure_collected,
            "treasure_pickup_bonus": treasure_pickup_bonus,
            "min_free_path": min_free_path,
            "max_free_path": max_free_path,
            "legal_move_count": move_legal_count,
            "wall_repulsion_scale": wall_repulsion_scale,
            "direction_consistency_reward": direction_consistency_reward,
            "direction_consistency_cosine": direction_consistency_cosine,
        }
        return feature, legal_action, remain_info

    def _compute_predictive_monster_penalty(self, hero_pos, current_monster_positions):
        danger_scale = self._get_predictive_danger_scale()
        penalty_coef = Config.PREDICTIVE_DANGER_PENALTY_COEF * danger_scale
        penalty_cap = Config.PREDICTIVE_DANGER_PENALTY_CAP * danger_scale
        total_penalty = 0.0
        for idx, monster_pos in enumerate(current_monster_positions):
            prev_pos = self.last_monster_positions[idx] if idx < len(self.last_monster_positions) else None
            if monster_pos is None or prev_pos is None:
                continue

            velocity_x = monster_pos["x"] - prev_pos["x"]
            velocity_z = monster_pos["z"] - prev_pos["z"]
            velocity_norm = float(np.sqrt(velocity_x**2 + velocity_z**2))
            if velocity_norm <= 1e-6:
                continue

            highest_risk = 0.0
            for horizon in Config.PREDICTIVE_MONSTER_HORIZONS:
                predicted_pos = {
                    "x": monster_pos["x"] + velocity_x * horizon,
                    "z": monster_pos["z"] + velocity_z * horizon,
                }
                predicted_dist_norm = _norm(_distance(hero_pos, predicted_pos), MAP_SIZE * 1.42)
                if predicted_dist_norm < Config.PREDICTIVE_DANGER_DIST_NORM:
                    risk = (
                        Config.PREDICTIVE_DANGER_DIST_NORM - predicted_dist_norm
                    ) / Config.PREDICTIVE_DANGER_DIST_NORM
                    highest_risk = max(highest_risk, risk)

            total_penalty -= penalty_coef * highest_risk

        return max(total_penalty, -penalty_cap)

    def _get_predictive_danger_scale(self):
        if self.curriculum_stage_id <= 1:
            return Config.PREDICTIVE_DANGER_STAGE1_SCALE
        if self.curriculum_stage_id == 2:
            return Config.PREDICTIVE_DANGER_STAGE2_SCALE
        return Config.PREDICTIVE_DANGER_STAGE3_SCALE

    def _compute_survival_reward(self):
        max_step = max(1, self.max_step)
        step_progress = min(1.0, max(0.0, self.step_no / float(max_step)))
        decay_start = float(np.clip(Config.SURVIVAL_REWARD_DECAY_START, 0.0, 1.0))
        min_scale = float(np.clip(Config.SURVIVAL_REWARD_MIN_SCALE, 0.0, 1.0))

        if step_progress <= decay_start:
            reward_scale = 1.0
        else:
            decay_progress = (step_progress - decay_start) / max(1e-6, 1.0 - decay_start)
            reward_scale = 1.0 - (1.0 - min_scale) * decay_progress

        return Config.SURVIVAL_STEP_REWARD * max(min_scale, reward_scale)

    def _build_legal_action_mask(self, legal_act_raw, flash_cd):
        legal_action = [1] * 16

        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for idx in range(min(len(legal_act_raw), 16)):
                    legal_action[idx] = int(legal_act_raw[idx])
            else:
                valid_set = {int(action) for action in legal_act_raw if int(action) < 16}
                legal_action = [1 if idx in valid_set else 0 for idx in range(16)]

        move_action = legal_action[:8]
        flash_legal_from_env = legal_action[8:]
        if len(flash_legal_from_env) < 8:
            flash_legal_from_env = [1] * 8

        if flash_cd > 0:
            flash_action = [0] * 8
        elif not any(flash_legal_from_env) and len(legal_action) == 16:
            flash_action = move_action.copy()
        else:
            flash_action = flash_legal_from_env[:8]

        legal_action = move_action + flash_action
        if sum(legal_action) == 0:
            legal_action = [1] * 8 + ([0] * 8 if flash_cd > 0 else [1] * 8)
        return legal_action
