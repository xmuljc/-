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
        self.curriculum_progress = 0.0
        self.curriculum_phase = "warmup_stable"
        self.curriculum_monster_interval = 300
        self.curriculum_monster_speedup = 500
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

    def set_curriculum(self, curriculum):
        if isinstance(curriculum, dict):
            self.curriculum_stage_id = int(max(1, curriculum.get("stage_id", 1)))
            self.curriculum_progress = float(np.clip(curriculum.get("progress", 0.0), 0.0, 1.0))
            self.curriculum_phase = str(curriculum.get("phase", self.curriculum_phase))
            self.curriculum_monster_interval = int(max(11, curriculum.get("monster_interval", 300)))
            self.curriculum_monster_speedup = int(max(1, curriculum.get("monster_speedup", 500)))
            return

        self.curriculum_stage_id = int(max(1, curriculum))
        self.curriculum_progress = 0.0

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
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        cur_min_monster_dist_norm = 1.0
        visible_monster_count = 0
        current_monster_positions = [None, None]
        monster_dist_norms = [1.0, 1.0]
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
                    monster_dist_norms[i] = dist_norm
                    cur_min_monster_dist_norm = min(cur_min_monster_dist_norm, dist_norm)
                    prev_monster_pos = self.last_monster_positions[i] if i < len(self.last_monster_positions) else None
                    approach_signal = 0.0
                    if prev_monster_pos is not None:
                        prev_dist = _distance(hero_pos, prev_monster_pos)
                        approach_signal = _signed_norm(prev_dist - raw_dist, MAX_MOVE_DELTA)
                    danger_flag = float(dist_norm < Config.POST_SPEEDUP_DANGER_DIST_NORM)
                    monster_feat = np.array(
                        [
                            is_in_view,
                            _signed_norm(monster_pos["x"] - hero_pos["x"], MAP_SIZE),
                            _signed_norm(monster_pos["z"] - hero_pos["z"], MAP_SIZE),
                            _norm(monster.get("speed", 1.0), MAX_MONSTER_SPEED),
                            dist_norm,
                            approach_signal,
                            danger_flag,
                        ],
                        dtype=np.float32,
                    )
                else:
                    monster_feat = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            else:
                monster_feat = np.zeros(Config.MONSTER_FEATURE_DIM, dtype=np.float32)
                monster_feat[4] = 1.0
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
        speedup_remaining_steps = max(0, int(self.curriculum_monster_speedup) - int(self.step_no))
        speedup_countdown_norm = (
            speedup_remaining_steps / float(max(1, int(self.curriculum_monster_speedup)))
        )
        high_pressure_flag = float(self.step_no >= int(self.curriculum_monster_speedup))
        speedup_buffer_flag = float(0 < speedup_remaining_steps <= Config.SPEEDUP_BUFFER_WINDOW)

        treasure_safety = 1.0
        if visible_monster_count > 0:
            treasure_safety = float(np.clip(cur_min_monster_dist_norm / max(TREASURE_SAFE_DIST_NORM, 1e-6), 0.0, 1.0))
        if high_pressure_flag > 0.5:
            treasure_safety *= 0.7
        treasure_opportunity = (1.0 - treasure_dist_norm) * treasure_safety
        treasure_feat = np.concatenate(
            [
                treasure_feat,
                np.array(
                    [
                        _norm(visible_treasure_count, MAX_VISIBLE_TREASURE),
                        float(np.clip(treasure_opportunity, 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        buff_safety = 1.0 if visible_monster_count == 0 else float(np.clip(cur_min_monster_dist_norm / 0.25, 0.0, 1.0))
        buff_opportunity = (1.0 - buff_dist_norm) * buff_safety * (0.2 if buff_remain > 0 else 1.0)
        buff_feat = np.concatenate(
            [
                buff_feat,
                np.array(
                    [
                        _norm(visible_buff_count, MAX_VISIBLE_BUFF),
                        float(np.clip(buff_opportunity, 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        legal_action = self._build_legal_action_mask(legal_act_raw, flash_cd)
        local_map = _extract_centered_map(map_info)
        obstacle_feat, _ = _extract_obstacle_feature(local_map, legal_action)
        local_map_feat = _extract_local_map_feature(local_map)
        min_free_path = float(np.min(obstacle_feat[:8])) if obstacle_feat.size >= 8 else 0.0
        max_free_path = float(np.max(obstacle_feat[:8])) if obstacle_feat.size >= 8 else 0.0
        local_blocked_ratio = float(obstacle_feat[9]) if obstacle_feat.size >= 10 else 0.0
        move_legal_count = int(np.count_nonzero(np.asarray(legal_action[:8], dtype=np.float32) > 0.5))
        reward_cfg = self._build_reward_config()
        is_speedup_buffer = speedup_buffer_flag > 0.5
        is_post_speedup = high_pressure_flag > 0.5
        if is_post_speedup:
            danger_phase_scale = Config.POST_SPEEDUP_DANGER_BOOST
            treasure_phase_scale = Config.POST_SPEEDUP_TREASURE_DAMPING
            buff_phase_scale = Config.POST_SPEEDUP_PICKUP_DAMPING
            space_phase_scale = Config.POST_SPEEDUP_SPACE_REWARD_BOOST
        elif is_speedup_buffer:
            danger_phase_scale = Config.SPEEDUP_BUFFER_DANGER_BOOST
            treasure_phase_scale = Config.SPEEDUP_BUFFER_TREASURE_DAMPING
            buff_phase_scale = Config.SPEEDUP_BUFFER_TREASURE_DAMPING
            space_phase_scale = Config.SPEEDUP_BUFFER_SPACE_REWARD_BOOST
        else:
            danger_phase_scale = 1.0
            treasure_phase_scale = 1.0
            buff_phase_scale = 1.0
            space_phase_scale = Config.PRE_SPEEDUP_SPACE_REWARD_DAMPING
            if visible_monster_count > 0 and cur_min_monster_dist_norm < Config.SPEEDUP_BUFFER_DANGER_DIST_NORM:
                space_phase_scale = 0.85

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
        survival_reward = self._compute_survival_reward(reward_cfg["survival_step_reward"])
        danger_shaping = reward_cfg["danger_shaping_coef"] * (
            cur_min_monster_dist_norm - self.last_min_monster_dist_norm
        ) * danger_phase_scale
        safe_treasure_state = treasure_safety >= Config.TREASURE_SAFE_OPPORTUNITY_THRESHOLD
        treasure_shaping_coef = (
            reward_cfg["safe_treasure_shaping_coef"] if safe_treasure_state else reward_cfg["treasure_shaping_coef"]
        )
        treasure_shaping = (
            treasure_shaping_coef
            * (self.last_treasure_dist_norm - treasure_dist_norm)
            * max(Config.TREASURE_OPPORTUNITY_MIN_SCALE, treasure_opportunity)
            * treasure_phase_scale
        )
        treasure_collected = float(score_delta >= TREASURE_SCORE_DELTA_THRESHOLD)
        treasure_pickup_bonus = 0.0
        if treasure_collected:
            self.treasure_pickup_count += 1
            treasure_pickup_bonus = reward_cfg["treasure_pickup_bonus"] + (
                Config.COLLECTION_STREAK_BONUS
                * min(self.treasure_pickup_count - 1, Config.MAX_COLLECTION_STREAK_BONUS)
            )
            treasure_pickup_bonus *= treasure_phase_scale

        buff_danger_state = visible_monster_count > 0 and cur_min_monster_dist_norm < BUFF_DANGER_DIST_NORM
        buff_shaping_coef = (
            reward_cfg["buff_danger_shaping_coef"] if buff_danger_state else reward_cfg["buff_shaping_coef"]
        )
        buff_shaping = (
            buff_shaping_coef
            * (self.last_buff_dist_norm - buff_dist_norm)
            * max(0.2, buff_opportunity)
            * buff_phase_scale
            if buff_remain <= 0
            else 0.0
        )
        buff_active = buff_remain > 0.0
        buff_collected = float(buff_active and not self.last_buff_active)
        buff_pickup_bonus = 0.0
        if buff_collected:
            self.buff_pickup_count += 1
            buff_pickup_bonus = reward_cfg["buff_pickup_bonus"] + (
                Config.COLLECTION_STREAK_BONUS
                * min(self.buff_pickup_count - 1, Config.MAX_COLLECTION_STREAK_BONUS)
            )
            buff_pickup_bonus *= buff_phase_scale

        wall_collision_penalty = 0.0
        stuck_penalty = 0.0
        revisit_penalty = 0.0
        revisit_hits = 0
        wall_repulsion_penalty = 0.0
        wall_repulsion_scale = 1.0
        if min_free_path < Config.WALL_REPULSION_THRESHOLD:
            if move_legal_count < (len(DIRECTION_OFFSETS) // 2):
                wall_repulsion_scale = move_legal_count / float(len(DIRECTION_OFFSETS) // 2)
            wall_repulsion_penalty = -reward_cfg["wall_repulsion_coef"] * (
                (Config.WALL_REPULSION_THRESHOLD - min_free_path) / Config.WALL_REPULSION_THRESHOLD
            ) * wall_repulsion_scale
        if self.last_hero_pos is not None and move_dist <= STATIONARY_EPS and last_action >= 0:
            wall_collision_penalty = Config.WALL_COLLISION_PENALTY
            if self.stationary_steps >= 3:
                stuck_penalty -= reward_cfg["stuck_stationary_penalty_coef"] * min(self.stationary_steps - 2, 6)
            if self.repeat_blocked_steps >= 2:
                stuck_penalty -= reward_cfg["stuck_repeat_penalty_coef"] * min(self.repeat_blocked_steps - 1, 6)
            if buff_danger_state or (visible_monster_count > 0 and cur_min_monster_dist_norm < FLASH_DANGER_DIST_NORM):
                stuck_penalty *= Config.STUCK_DANGER_MULTIPLIER
            stuck_penalty = max(stuck_penalty, -Config.STUCK_PENALTY_CAP)
        if wall_collision_penalty < 0.0:
            wall_collision_penalty *= max(1.0, danger_phase_scale)
        if stuck_penalty < 0.0:
            stuck_penalty *= max(1.0, danger_phase_scale)
        if wall_repulsion_penalty < 0.0:
            wall_repulsion_penalty *= max(1.0, 0.5 + 0.5 * danger_phase_scale)

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
                revisit_penalty = -reward_cfg["loop_revisit_penalty_coef"] * revisit_ratio
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
                flash_waste_penalty -= reward_cfg["flash_waste_blocked_penalty"]
            elif flash_in_danger:
                flash_escape_bonus += reward_cfg["flash_escape_bonus_base"] + (
                    reward_cfg["flash_escape_bonus_gain_coef"] * max(0.0, dist_gain)
                )
                if dist_gain > 0.02:
                    self.pending_flash_survival_steps = POST_FLASH_BONUS_STEPS
                    self.pending_flash_survival_reward = min(
                        reward_cfg["flash_survival_bonus_max"],
                        reward_cfg["flash_survival_bonus_base"]
                        + reward_cfg["flash_survival_bonus_gain_coef"] * dist_gain,
                    )
            elif flash_without_pressure:
                flash_waste_penalty -= reward_cfg["flash_waste_safe_penalty"]
            elif dist_gain <= 0.0:
                flash_waste_penalty -= reward_cfg["flash_waste_bad_gain_penalty"]

        flash_reward = flash_escape_bonus + flash_survival_bonus + flash_waste_penalty
        second_monster_dist_norm = monster_dist_norms[1] if len(monster_dist_norms) > 1 else 1.0
        speedup_buffer_reward, post_speedup_survival_reward, post_speedup_active = self._compute_speedup_rewards(
            cur_min_monster_dist_norm,
            second_monster_dist_norm,
            max_free_path,
            local_blocked_ratio,
            reward_cfg,
        )

        predictive_danger_penalty = self._compute_predictive_monster_penalty(
            hero_pos,
            current_monster_positions,
            reward_cfg,
        )
        predictive_danger_penalty *= danger_phase_scale
        space_control_reward, open_space_bonus, dead_end_penalty = self._compute_space_control_reward(
            min_free_path,
            max_free_path,
            local_blocked_ratio,
            reward_cfg,
        )
        space_control_reward *= space_phase_scale
        open_space_bonus *= space_phase_scale
        dead_end_penalty *= space_phase_scale
        second_monster_penalty = self._compute_second_monster_penalty(
            second_monster_dist_norm,
            visible_monster_count,
            reward_cfg,
        )
        second_monster_penalty *= danger_phase_scale
        encirclement_penalty = self._compute_encirclement_penalty(
            hero_pos,
            current_monster_positions,
            visible_monster_count,
            min_free_path,
            local_blocked_ratio,
            reward_cfg,
        )
        encirclement_penalty *= danger_phase_scale
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
            + survival_reward
            + danger_shaping
            + treasure_shaping
            + treasure_pickup_bonus
            + buff_shaping
            + buff_pickup_bonus
            + wall_collision_penalty
            + wall_repulsion_penalty
            + stuck_penalty
            + revisit_penalty
            + flash_reward
            + predictive_danger_penalty
            + space_control_reward
            + second_monster_penalty
            + encirclement_penalty
            + speedup_buffer_reward
            + post_speedup_survival_reward
            + direction_consistency_reward
            + final_reward
        ]

        score_norm = _norm(current_total_score, MAX_OFFICIAL_SCORE)
        hero_feat = np.array(
            [
                hero_x_norm,
                hero_z_norm,
                hero_speed_norm,
                score_norm,
                _norm(self.stationary_steps, MAX_STATIONARY_STEPS),
                _norm(self.repeat_blocked_steps, MAX_STATIONARY_STEPS),
                blocked_move,
                float(move_legal_count) / float(len(DIRECTION_OFFSETS)),
            ],
            dtype=np.float32,
        )
        skill_feat = np.array(
            [
                flash_cd_norm,
                float(flash_cd <= 1e-6),
                buff_remain_norm,
                float(buff_remain > 0.0),
                speedup_countdown_norm,
                high_pressure_flag,
                speedup_buffer_flag,
            ],
            dtype=np.float32,
        )
        obstacle_context_feat = np.concatenate(
            [
                obstacle_feat,
                np.array(
                    [
                        min_free_path,
                        max_free_path,
                        float(min_free_path < Config.DEAD_END_THRESHOLD),
                        float(
                            (0.5 * (min_free_path + max_free_path) > Config.SPACE_OPEN_THRESHOLD)
                            and local_blocked_ratio < Config.SPACE_OPEN_MAX_BLOCKED_RATIO
                        ),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        visible_monster_count_norm = _norm(visible_monster_count, 2.0)
        progress_feat = np.array(
            [
                _norm(self.step_no, self.max_step),
                score_norm,
                float(buff_remain > 0.0),
                float(last_action >= 8),
                _norm(visible_treasure_count, MAX_VISIBLE_TREASURE),
                _norm(visible_buff_count, MAX_VISIBLE_BUFF),
                visible_monster_count_norm,
                speedup_countdown_norm,
                high_pressure_flag,
                self.curriculum_progress,
                _norm(self.curriculum_stage_id, 4.0, 1.0),
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
                skill_feat,
                obstacle_context_feat,
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
            "step_score": float(env_info.get("step_score", current_total_score)),
            "treasure_score": float(env_info.get("treasure_score", 0.0)),
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
            "treasure_shaping": treasure_shaping,
            "treasure_pickup_bonus": treasure_pickup_bonus,
            "buff_shaping": buff_shaping,
            "survival_reward": survival_reward,
            "predictive_danger_penalty": predictive_danger_penalty,
            "space_control_reward": space_control_reward,
            "open_space_bonus": open_space_bonus,
            "dead_end_penalty": dead_end_penalty,
            "second_monster_penalty": second_monster_penalty,
            "encirclement_penalty": encirclement_penalty,
            "speedup_buffer_reward": speedup_buffer_reward,
            "post_speedup_survival_reward": post_speedup_survival_reward,
            "post_speedup_active": float(post_speedup_active),
            "final_reward": final_reward,
            "buff_collected": buff_collected,
            "buff_pickup_bonus": buff_pickup_bonus,
            "treasure_collected": treasure_collected,
            "min_free_path": min_free_path,
            "max_free_path": max_free_path,
            "legal_move_count": move_legal_count,
            "wall_repulsion_scale": wall_repulsion_scale,
            "danger_phase_scale": danger_phase_scale,
            "treasure_phase_scale": treasure_phase_scale,
            "space_phase_scale": space_phase_scale,
            "direction_consistency_reward": direction_consistency_reward,
            "direction_consistency_cosine": direction_consistency_cosine,
        }
        return feature, legal_action, remain_info

    def _compute_predictive_monster_penalty(self, hero_pos, current_monster_positions, reward_cfg):
        penalty_coef = reward_cfg["predictive_danger_penalty_coef"]
        penalty_cap = reward_cfg["predictive_danger_penalty_cap"]
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

    def _compute_survival_reward(self, survival_step_reward):
        max_step = max(1, self.max_step)
        step_progress = min(1.0, max(0.0, self.step_no / float(max_step)))
        decay_start = float(np.clip(Config.SURVIVAL_REWARD_DECAY_START, 0.0, 1.0))
        min_scale = float(np.clip(Config.SURVIVAL_REWARD_MIN_SCALE, 0.0, 1.0))

        if step_progress <= decay_start:
            reward_scale = 1.0
        else:
            decay_progress = (step_progress - decay_start) / max(1e-6, 1.0 - decay_start)
            reward_scale = 1.0 - (1.0 - min_scale) * decay_progress

        return survival_step_reward * max(min_scale, reward_scale)

    def _build_reward_config(self):
        return {
            "survival_step_reward": self._curriculum_interp(Config.SURVIVAL_STEP_REWARD_SCHEDULE),
            "danger_shaping_coef": self._curriculum_interp(Config.DANGER_SHAPING_SCHEDULE),
            "treasure_shaping_coef": self._curriculum_interp(Config.TREASURE_SHAPING_SCHEDULE),
            "safe_treasure_shaping_coef": self._curriculum_interp(Config.SAFE_TREASURE_SHAPING_SCHEDULE),
            "buff_shaping_coef": self._curriculum_interp(Config.BUFF_SHAPING_SCHEDULE),
            "buff_danger_shaping_coef": self._curriculum_interp(Config.BUFF_DANGER_SHAPING_SCHEDULE),
            "treasure_pickup_bonus": self._curriculum_interp(Config.TREASURE_PICKUP_BONUS_SCHEDULE),
            "buff_pickup_bonus": self._curriculum_interp(Config.BUFF_PICKUP_BONUS_SCHEDULE),
            "stuck_stationary_penalty_coef": self._curriculum_interp(Config.STUCK_STATIONARY_PENALTY_SCHEDULE),
            "stuck_repeat_penalty_coef": self._curriculum_interp(Config.STUCK_REPEAT_PENALTY_SCHEDULE),
            "loop_revisit_penalty_coef": self._curriculum_interp(Config.LOOP_REVISIT_PENALTY_SCHEDULE),
            "wall_repulsion_coef": self._curriculum_interp(Config.WALL_REPULSION_SCHEDULE),
            "flash_escape_bonus_base": self._curriculum_interp(Config.FLASH_ESCAPE_BASE_SCHEDULE),
            "flash_escape_bonus_gain_coef": self._curriculum_interp(Config.FLASH_ESCAPE_GAIN_SCHEDULE),
            "flash_survival_bonus_base": self._curriculum_interp(Config.FLASH_SURVIVAL_BASE_SCHEDULE),
            "flash_survival_bonus_gain_coef": self._curriculum_interp(Config.FLASH_SURVIVAL_GAIN_SCHEDULE),
            "flash_survival_bonus_max": self._curriculum_interp(Config.FLASH_SURVIVAL_MAX_SCHEDULE),
            "flash_waste_blocked_penalty": self._curriculum_interp(
                Config.FLASH_WASTE_BLOCKED_PENALTY_SCHEDULE
            ),
            "flash_waste_safe_penalty": self._curriculum_interp(Config.FLASH_WASTE_SAFE_PENALTY_SCHEDULE),
            "flash_waste_bad_gain_penalty": self._curriculum_interp(
                Config.FLASH_WASTE_BAD_GAIN_PENALTY_SCHEDULE
            ),
            "predictive_danger_penalty_coef": self._curriculum_interp(
                Config.PREDICTIVE_DANGER_PENALTY_SCHEDULE
            ),
            "predictive_danger_penalty_cap": self._curriculum_interp(Config.PREDICTIVE_DANGER_CAP_SCHEDULE),
            "space_open_bonus": self._curriculum_interp(Config.SPACE_OPEN_BONUS_SCHEDULE),
            "dead_end_penalty": self._curriculum_interp(Config.DEAD_END_PENALTY_SCHEDULE),
            "second_monster_penalty": self._curriculum_interp(Config.SECOND_MONSTER_PENALTY_SCHEDULE),
            "encirclement_penalty": self._curriculum_interp(Config.ENCIRCLEMENT_PENALTY_SCHEDULE),
            "speedup_buffer_reward": self._curriculum_interp(Config.SPEEDUP_BUFFER_REWARD_SCHEDULE),
            "post_speedup_survival_reward": self._curriculum_interp(
                Config.POST_SPEEDUP_SURVIVAL_REWARD_SCHEDULE
            ),
        }

    def _compute_space_control_reward(self, min_free_path, max_free_path, local_blocked_ratio, reward_cfg):
        open_space_bonus = 0.0
        dead_end_penalty = 0.0
        avg_free_path = 0.5 * (min_free_path + max_free_path)

        if (
            avg_free_path > Config.SPACE_OPEN_THRESHOLD
            and local_blocked_ratio < Config.SPACE_OPEN_MAX_BLOCKED_RATIO
        ):
            free_path_gain = (avg_free_path - Config.SPACE_OPEN_THRESHOLD) / max(
                1e-6, 1.0 - Config.SPACE_OPEN_THRESHOLD
            )
            clearance = (Config.SPACE_OPEN_MAX_BLOCKED_RATIO - local_blocked_ratio) / max(
                1e-6, Config.SPACE_OPEN_MAX_BLOCKED_RATIO
            )
            open_space_bonus = reward_cfg["space_open_bonus"] * max(0.0, free_path_gain) * max(0.0, clearance)

        if min_free_path < Config.DEAD_END_THRESHOLD:
            dead_end_ratio = (Config.DEAD_END_THRESHOLD - min_free_path) / max(1e-6, Config.DEAD_END_THRESHOLD)
            dead_end_penalty = -reward_cfg["dead_end_penalty"] * dead_end_ratio
            if local_blocked_ratio > Config.SPACE_OPEN_MAX_BLOCKED_RATIO:
                confinement_ratio = min(
                    1.0,
                    (local_blocked_ratio - Config.SPACE_OPEN_MAX_BLOCKED_RATIO)
                    / max(1e-6, 1.0 - Config.SPACE_OPEN_MAX_BLOCKED_RATIO),
                )
                dead_end_penalty *= 1.0 + confinement_ratio

        return open_space_bonus + dead_end_penalty, open_space_bonus, dead_end_penalty

    def _compute_second_monster_penalty(self, second_monster_dist_norm, visible_monster_count, reward_cfg):
        if visible_monster_count < 2:
            return 0.0
        if second_monster_dist_norm >= Config.SECOND_MONSTER_DANGER_DIST_NORM:
            return 0.0

        pressure_ratio = (
            Config.SECOND_MONSTER_DANGER_DIST_NORM - second_monster_dist_norm
        ) / max(1e-6, Config.SECOND_MONSTER_DANGER_DIST_NORM)
        return -reward_cfg["second_monster_penalty"] * pressure_ratio

    def _compute_encirclement_penalty(
        self,
        hero_pos,
        current_monster_positions,
        visible_monster_count,
        min_free_path,
        local_blocked_ratio,
        reward_cfg,
    ):
        if visible_monster_count < 2:
            return 0.0

        visible_positions = [pos for pos in current_monster_positions if pos is not None]
        if len(visible_positions) < 2:
            return 0.0

        vectors = []
        close_pressure = 0.0
        for monster_pos in visible_positions[:2]:
            delta = np.array(
                [
                    monster_pos["x"] - hero_pos["x"],
                    monster_pos["z"] - hero_pos["z"],
                ],
                dtype=np.float32,
            )
            distance_norm = _norm(float(np.linalg.norm(delta)), MAP_SIZE * 1.42)
            if distance_norm < Config.ENCIRCLEMENT_MONSTER_DIST_NORM:
                close_pressure += (
                    Config.ENCIRCLEMENT_MONSTER_DIST_NORM - distance_norm
                ) / max(1e-6, Config.ENCIRCLEMENT_MONSTER_DIST_NORM)
            vectors.append(delta)

        if close_pressure <= 0.0:
            return 0.0

        vec_norm_0 = float(np.linalg.norm(vectors[0]))
        vec_norm_1 = float(np.linalg.norm(vectors[1]))
        if vec_norm_0 <= 1e-6 or vec_norm_1 <= 1e-6:
            return 0.0

        monster_cosine = float(
            np.clip(
                np.dot(vectors[0], vectors[1]) / max(1e-6, vec_norm_0 * vec_norm_1),
                -1.0,
                1.0,
            )
        )
        if monster_cosine > Config.ENCIRCLEMENT_OPPOSITE_COS_THRESHOLD:
            return 0.0

        opposite_ratio = (
            Config.ENCIRCLEMENT_OPPOSITE_COS_THRESHOLD - monster_cosine
        ) / max(1e-6, Config.ENCIRCLEMENT_OPPOSITE_COS_THRESHOLD + 1.0)
        free_path_pressure = max(
            0.0,
            (Config.ENCIRCLEMENT_FREE_PATH_THRESHOLD - min_free_path)
            / max(1e-6, Config.ENCIRCLEMENT_FREE_PATH_THRESHOLD),
        )
        blocked_pressure = max(
            0.0,
            (local_blocked_ratio - Config.SPACE_OPEN_MAX_BLOCKED_RATIO)
            / max(1e-6, 1.0 - Config.SPACE_OPEN_MAX_BLOCKED_RATIO),
        )
        confinement = max(free_path_pressure, blocked_pressure)
        return -reward_cfg["encirclement_penalty"] * min(1.0, 0.5 * close_pressure) * opposite_ratio * (
            0.4 + 0.6 * confinement
        )

    def _compute_speedup_rewards(
        self,
        cur_min_monster_dist_norm,
        second_monster_dist_norm,
        max_free_path,
        local_blocked_ratio,
        reward_cfg,
    ):
        monster_speedup_step = max(1, int(self.curriculum_monster_speedup))
        remaining_steps = monster_speedup_step - self.step_no
        speedup_buffer_reward = 0.0
        post_speedup_survival_reward = 0.0
        post_speedup_active = False

        if 0 < remaining_steps <= Config.SPEEDUP_BUFFER_WINDOW:
            proximity = 1.0 - remaining_steps / float(max(1, Config.SPEEDUP_BUFFER_WINDOW))
            monster_safety = max(0.0, cur_min_monster_dist_norm - Config.SPEEDUP_BUFFER_DANGER_DIST_NORM)
            second_monster_safety = max(
                0.0, second_monster_dist_norm - Config.SPEEDUP_BUFFER_DANGER_DIST_NORM
            )
            space_margin = max(0.0, max_free_path - Config.SPEEDUP_BUFFER_FREE_PATH_THRESHOLD)
            clearance = max(
                0.0,
                (Config.SPACE_OPEN_MAX_BLOCKED_RATIO - local_blocked_ratio)
                / max(1e-6, Config.SPACE_OPEN_MAX_BLOCKED_RATIO),
            )
            speedup_buffer_reward = reward_cfg["speedup_buffer_reward"] * proximity * (
                0.8 * monster_safety + 0.4 * second_monster_safety + 0.8 * space_margin * max(0.25, clearance)
            )

        if self.step_no >= monster_speedup_step:
            post_speedup_active = True
            monster_safety = max(0.0, cur_min_monster_dist_norm - Config.POST_SPEEDUP_DANGER_DIST_NORM)
            space_margin = max(0.0, max_free_path - Config.POST_SPEEDUP_FREE_PATH_THRESHOLD)
            post_speedup_survival_reward = reward_cfg["post_speedup_survival_reward"] * (
                0.8 + monster_safety + 0.6 * space_margin
            )

        return speedup_buffer_reward, post_speedup_survival_reward, post_speedup_active

    def _curriculum_interp(self, schedule):
        progress_knots = getattr(Config, "CURRICULUM_PROGRESS_KNOTS", (0.0, 1.0))
        if not schedule:
            return 0.0
        if len(schedule) != len(progress_knots):
            return float(schedule[-1])

        progress = float(np.clip(self.curriculum_progress, 0.0, 1.0))
        for idx in range(len(progress_knots) - 1):
            left_progress = float(progress_knots[idx])
            right_progress = float(progress_knots[idx + 1])
            if progress <= right_progress or idx == len(progress_knots) - 2:
                left_value = float(schedule[idx])
                right_value = float(schedule[idx + 1])
                span = max(1e-6, right_progress - left_progress)
                ratio = min(1.0, max(0.0, (progress - left_progress) / span))
                return left_value + (right_value - left_value) * ratio
        return float(schedule[-1])

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
