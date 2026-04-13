#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Situation-oriented feature preprocessor and reward design for Gorge Chase PPO.
"""

import numpy as np

from agent_ppo.conf.conf import Config


MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
PATCH_RADIUS = Config.SPATIAL_MAP_SIZE // 2
PATCH_SIZE = Config.SPATIAL_MAP_SIZE
MAX_WORLD_DIST = MAP_SIZE * 1.41
DEFAULT_SECOND_MONSTER_STEP = 500.0
DEFAULT_MONSTER_SPEEDUP_STEP = 700.0
MAX_TOTAL_SCORE = 2000.0
HISTORY_LIMIT = 12
STUCK_DISTANCE = 1.5
LOOP_DISTANCE = 2.5

# Clockwise from east. These are used as directional summaries, not hard-coded
# action semantics.
DIRECTION_VECTORS = np.array(
    [
        [1.0, 0.0],
        [0.7071, -0.7071],
        [0.0, -1.0],
        [-0.7071, -0.7071],
        [-1.0, 0.0],
        [-0.7071, 0.7071],
        [0.0, 1.0],
        [0.7071, 0.7071],
    ],
    dtype=np.float32,
)

DIRECTION_RAYS = [
    [(PATCH_RADIUS, PATCH_RADIUS + 1), (PATCH_RADIUS, PATCH_RADIUS + 2)],
    [(PATCH_RADIUS - 1, PATCH_RADIUS + 1), (PATCH_RADIUS - 2, PATCH_RADIUS + 2)],
    [(PATCH_RADIUS - 1, PATCH_RADIUS), (PATCH_RADIUS - 2, PATCH_RADIUS)],
    [(PATCH_RADIUS - 1, PATCH_RADIUS - 1), (PATCH_RADIUS - 2, PATCH_RADIUS - 2)],
    [(PATCH_RADIUS, PATCH_RADIUS - 1), (PATCH_RADIUS, PATCH_RADIUS - 2)],
    [(PATCH_RADIUS + 1, PATCH_RADIUS - 1), (PATCH_RADIUS + 2, PATCH_RADIUS - 2)],
    [(PATCH_RADIUS + 1, PATCH_RADIUS), (PATCH_RADIUS + 2, PATCH_RADIUS)],
    [(PATCH_RADIUS + 1, PATCH_RADIUS + 1), (PATCH_RADIUS + 2, PATCH_RADIUS + 2)],
]


def _norm(value, value_max, value_min=0.0):
    value = float(np.clip(value, value_min, value_max))
    scale = value_max - value_min
    if scale <= 1e-6:
        return 0.0
    return (value - value_min) / scale


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _as_float(value, default=0.0):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_pos(entity):
    return entity.get("pos", {}) if isinstance(entity, dict) else {}


def _direction_and_distance(src_pos, dst_pos):
    dx = float(dst_pos.get("x", 0.0) - src_pos.get("x", 0.0))
    dz = float(dst_pos.get("z", 0.0) - src_pos.get("z", 0.0))
    dist = float(np.sqrt(dx * dx + dz * dz))
    if dist <= 1e-6:
        return np.zeros(2, dtype=np.float32), 0.0, 0.0
    return np.array([dx / dist, dz / dist], dtype=np.float32), dist, _norm(dist, MAX_WORLD_DIST)


def _extract_obstacle_patch(map_info):
    obstacle_patch = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    if map_info is None or len(map_info) == 0 or len(map_info[0]) == 0:
        obstacle_patch[PATCH_RADIUS, PATCH_RADIUS] = 0.0
        return obstacle_patch

    center_row = len(map_info) // 2
    center_col = len(map_info[0]) // 2
    for patch_row in range(PATCH_SIZE):
        for patch_col in range(PATCH_SIZE):
            row = center_row + patch_row - PATCH_RADIUS
            col = center_col + patch_col - PATCH_RADIUS
            if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                obstacle_patch[patch_row, patch_col] = float(map_info[row][col] != 0)

    obstacle_patch[PATCH_RADIUS, PATCH_RADIUS] = 0.0
    return obstacle_patch


def _compute_local_freedom(passable_patch):
    freedom = np.zeros_like(passable_patch, dtype=np.float32)
    for row in range(PATCH_SIZE):
        for col in range(PATCH_SIZE):
            if passable_patch[row, col] <= 0.0:
                continue
            row_start = max(0, row - 1)
            row_end = min(PATCH_SIZE, row + 2)
            col_start = max(0, col - 1)
            col_end = min(PATCH_SIZE, col + 2)
            neighborhood = passable_patch[row_start:row_end, col_start:col_end]
            freedom[row, col] = float(np.mean(neighborhood))
    return freedom


def _project_to_patch(direction, dist_norm):
    anchor_scale = np.clip(dist_norm * (PATCH_RADIUS * 2.5) + 0.5, 0.5, PATCH_RADIUS * 3.0)
    return direction * anchor_scale


def _make_field_from_anchor(anchor, strength):
    field = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for row in range(PATCH_SIZE):
        for col in range(PATCH_SIZE):
            coord = np.array([col - PATCH_RADIUS, row - PATCH_RADIUS], dtype=np.float32)
            dist = float(np.linalg.norm(coord - anchor))
            field[row, col] = strength / (1.0 + dist)
    return field


def _parse_legal_action(legal_act_raw):
    legal_action = [1] * Config.ACTION_NUM
    if isinstance(legal_act_raw, list) and legal_act_raw:
        if isinstance(legal_act_raw[0], bool):
            for idx in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                legal_action[idx] = int(legal_act_raw[idx])
        else:
            valid = {int(action) for action in legal_act_raw if 0 <= int(action) < Config.ACTION_NUM}
            legal_action = [1 if idx in valid else 0 for idx in range(Config.ACTION_NUM)]

    if sum(legal_action) == 0:
        legal_action = [1] * Config.ACTION_NUM
    return legal_action


def _nearest_direction_index(direction):
    if np.linalg.norm(direction) <= 1e-6:
        return 0
    similarities = DIRECTION_VECTORS @ direction.astype(np.float32)
    return int(np.argmax(similarities))


def _directional_boundary_clearance(direction, boundary_clearance):
    horizontal = boundary_clearance[1] if direction[0] > 0 else boundary_clearance[0]
    vertical = boundary_clearance[3] if direction[1] > 0 else boundary_clearance[2]
    if abs(float(direction[0])) > 1e-3 and abs(float(direction[1])) > 1e-3:
        return 0.5 * (horizontal + vertical)
    return horizontal if abs(float(direction[0])) >= abs(float(direction[1])) else vertical


def _summarize_resource_states(resource_states, direction_scores):
    if not resource_states:
        return np.zeros(8, dtype=np.float32), 0.0, 0.0

    nearest_state = min(resource_states, key=lambda item: item["raw_dist"])
    best_state = max(resource_states, key=lambda item: item["safe_value"])
    best_direction_support = direction_scores[best_state["direction_idx"]]
    best_reachability = _clip01(0.5 * best_direction_support + 0.5 * (1.0 - best_state["approach_risk"]))
    summary = np.array(
        [
            1.0,
            nearest_state["direction"][0],
            nearest_state["direction"][1],
            nearest_state["dist_score"],
            nearest_state["approach_risk"],
            nearest_state["safe_value"],
            best_state["safe_value"],
            best_reachability,
        ],
        dtype=np.float32,
    )
    resource_risk = max(nearest_state["approach_risk"], best_state["approach_risk"])
    return summary, best_state["safe_value"], resource_risk


def _summarize_monster_states(monster_states, enclosure_pressure):
    if not monster_states:
        return np.zeros(11, dtype=np.float32), None, None

    monster_states = sorted(monster_states, key=lambda item: item["raw_dist"])
    nearest_state = monster_states[0]
    second_state = monster_states[1] if len(monster_states) > 1 else None

    summary = np.array(
        [
            1.0,
            nearest_state["direction"][0],
            nearest_state["direction"][1],
            1.0 - nearest_state["dist_norm"],
            nearest_state["pressure"],
            1.0 if second_state is not None else 0.0,
            second_state["direction"][0] if second_state is not None else 0.0,
            second_state["direction"][1] if second_state is not None else 0.0,
            1.0 - second_state["dist_norm"] if second_state is not None else 0.0,
            second_state["pressure"] if second_state is not None else 0.0,
            enclosure_pressure,
        ],
        dtype=np.float32,
    )
    return summary, nearest_state, second_state


def _compute_terrain_context(passable_patch, local_freedom):
    ray_openness = np.zeros(Config.ACTION_NUM, dtype=np.float32)
    for idx, ray in enumerate(DIRECTION_RAYS):
        passable_values = []
        freedom_values = []
        for row, col in ray:
            passable_values.append(passable_patch[row, col])
            freedom_values.append(local_freedom[row, col])
        ray_openness[idx] = _clip01(0.45 * np.mean(passable_values) + 0.55 * np.mean(freedom_values))

    openness_score = _clip01(np.mean(ray_openness))
    corridor_score = max(
        min(ray_openness[0], ray_openness[4]),
        min(ray_openness[1], ray_openness[5]),
        min(ray_openness[2], ray_openness[6]),
        min(ray_openness[3], ray_openness[7]),
    )
    top_two = sorted(ray_openness.tolist(), reverse=True)[:2]
    dead_end_risk = _clip01(1.0 - 0.5 * sum(top_two))
    trap_risk = _clip01(0.6 * dead_end_risk + 0.4 * (1.0 - local_freedom[PATCH_RADIUS, PATCH_RADIUS]))
    return ray_openness, openness_score, corridor_score, dead_end_risk, trap_risk


def _compute_path_pattern_features(current_pos_vec, pos_history, recent_actions):
    if not pos_history:
        return 0.0, 0.0

    recent_positions = pos_history[-4:]
    move_dists = [float(np.linalg.norm(current_pos_vec - pos)) for pos in recent_positions]
    farthest_recent = max(move_dists) if move_dists else STUCK_DISTANCE
    stuck_score = _clip01((STUCK_DISTANCE - farthest_recent) / STUCK_DISTANCE)

    revisit_score = 0.0
    if len(pos_history) > 4:
        older_positions = pos_history[:-4]
        if older_positions:
            min_revisit_dist = min(float(np.linalg.norm(current_pos_vec - pos)) for pos in older_positions[-6:])
            revisit_score = _clip01((LOOP_DISTANCE - min_revisit_dist) / LOOP_DISTANCE)

    if recent_actions:
        unique_ratio = len(set(recent_actions)) / float(len(recent_actions))
        action_repeat = _clip01(1.0 - unique_ratio)
    else:
        action_repeat = 0.0

    loop_score = _clip01(0.65 * revisit_score + 0.35 * action_repeat)
    return stuck_score, loop_score


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.state_initialized = False
        self.pos_history = []
        self.action_history = []
        self.last_safety_score = 0.5
        self.last_best_escape = 0.5
        self.last_control_collapse = 0.5
        self.last_mobility_score = 0.5
        self.last_best_opportunity = 0.0
        self.last_risky_pursuit = 0.0
        self.last_step_score = 0.0
        self.last_treasure_score = 0.0
        self.last_total_score = 0.0
        self.last_nearest_treasure_dist_score = 0.0
        self.last_nearest_monster_dist_norm = 1.0
        self.last_pre_speedup_buffer = 0.0
        self.last_corridor_metric = 0.0
        self.last_danger_penalty = 0.0
        self.last_flash_cd_norm = 0.0
        self.last_flash_ready = 1.0
        self.last_buff_active = 0.0
        self.last_buff_remain_norm = 0.0

    def _collect_monster_states(self, hero_pos, monsters):
        monster_states = []
        for monster in monsters[:2]:
            if not monster.get("is_in_view", 0):
                continue
            direction, raw_dist, dist_norm = _direction_and_distance(hero_pos, _safe_pos(monster))
            speed_norm = _norm(monster.get("speed", 1.0), MAX_MONSTER_SPEED)
            pressure = float(np.clip((1.0 - dist_norm) * (0.7 + 0.3 * speed_norm), 0.0, 1.0))
            monster_states.append(
                {
                    "direction": direction,
                    "raw_dist": raw_dist,
                    "dist_norm": dist_norm,
                    "speed_norm": speed_norm,
                    "pressure": pressure,
                    "anchor": _project_to_patch(direction, dist_norm),
                }
            )
        return monster_states

    def _build_danger_field(self, monster_states):
        danger_field = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for monster_state in monster_states:
            anchor = monster_state["anchor"]
            pressure = monster_state["pressure"]
            direction = monster_state["direction"]
            monster_field = _make_field_from_anchor(anchor, pressure)
            for row in range(PATCH_SIZE):
                for col in range(PATCH_SIZE):
                    coord = np.array([col - PATCH_RADIUS, row - PATCH_RADIUS], dtype=np.float32)
                    norm = float(np.linalg.norm(coord))
                    if norm > 1e-6:
                        alignment = max(0.0, float(np.dot(coord / norm, direction)))
                    else:
                        alignment = 0.0
                    monster_field[row, col] *= 0.7 + 0.3 * alignment
            danger_field += monster_field
        return np.clip(danger_field, 0.0, 1.0)

    def _build_control_field(self, passable_patch, danger_field):
        local_freedom = _compute_local_freedom(passable_patch)
        control_field = passable_patch * np.clip(0.55 * (1.0 - danger_field) + 0.45 * local_freedom, 0.0, 1.0)
        control_field[PATCH_RADIUS, PATCH_RADIUS] = np.clip(
            0.65 * (1.0 - danger_field[PATCH_RADIUS, PATCH_RADIUS]) + 0.35 * local_freedom[PATCH_RADIUS, PATCH_RADIUS],
            0.0,
            1.0,
        )
        return control_field, local_freedom

    def _collect_resource_states(
        self,
        hero_pos,
        resources,
        control_score,
        combined_threat_dir,
        combined_threat_mag,
        enclosure_pressure,
        active_key=None,
    ):
        resource_states = []
        for resource in resources:
            if active_key is not None and not resource.get(active_key, True):
                continue
            direction, raw_dist, dist_norm = _direction_and_distance(hero_pos, _safe_pos(resource))
            if raw_dist <= 1e-6:
                continue
            dist_score = 1.0 - dist_norm
            threat_alignment = 0.0
            if combined_threat_mag > 1e-6:
                threat_alignment = max(0.0, float(np.dot(direction, combined_threat_dir)))
            approach_risk = float(
                np.clip(0.6 * combined_threat_mag * threat_alignment + 0.4 * enclosure_pressure, 0.0, 1.0)
            )
            safe_value = float(dist_score * (0.35 + 0.65 * control_score) * (1.0 - approach_risk))
            resource_states.append(
                {
                    "direction": direction,
                    "raw_dist": raw_dist,
                    "dist_norm": dist_norm,
                    "dist_score": dist_score,
                    "approach_risk": approach_risk,
                    "safe_value": safe_value,
                    "anchor": _project_to_patch(direction, dist_norm),
                    "direction_idx": _nearest_direction_index(direction),
                }
            )
        return resource_states

    def _build_opportunity_field(self, control_field, resource_states):
        opportunity_field = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for resource_state in resource_states:
            opportunity_field += _make_field_from_anchor(resource_state["anchor"], resource_state["safe_value"])
        opportunity_field = np.clip(opportunity_field, 0.0, 1.0)
        return opportunity_field * control_field

    def _compute_direction_scores(self, control_field, danger_field, passable_patch, boundary_clearance, legal_action):
        direction_scores = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        for idx, ray in enumerate(DIRECTION_RAYS):
            control_values = []
            danger_values = []
            passable_values = []
            for row, col in ray:
                control_values.append(control_field[row, col])
                danger_values.append(danger_field[row, col])
                passable_values.append(passable_patch[row, col])
            ray_control = float(np.mean(control_values))
            ray_danger = float(np.mean(danger_values))
            ray_passable = float(np.mean(passable_values))
            boundary_bonus = _directional_boundary_clearance(DIRECTION_VECTORS[idx], boundary_clearance)
            legal_bonus = 1.0 if legal_action[idx] > 0 else 0.0
            direction_scores[idx] = np.clip(
                0.45 * ray_control + 0.25 * ray_passable + 0.15 * boundary_bonus + 0.15 * legal_bonus - 0.25 * ray_danger,
                0.0,
                1.0,
            )
        return direction_scores

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)
        step_score = _as_float(env_info.get("step_score", 0.0))
        treasure_score = _as_float(env_info.get("treasure_score", 0.0))
        total_score = _as_float(env_info.get("total_score", step_score + treasure_score))
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        current_pos_vec = np.array([hero_pos.get("x", 0.0), hero_pos.get("z", 0.0)], dtype=np.float32)
        flash_cd_norm = _norm(hero.get("flash_cooldown", 0.0), MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0.0), MAX_BUFF_DURATION)
        flash_ready = 1.0 if hero.get("flash_cooldown", 0.0) <= 1e-6 else 0.0
        buff_active = 1.0 if hero.get("buff_remaining_time", 0.0) > 1e-6 else 0.0
        step_norm = _norm(self.step_no, self.max_step)
        total_score_norm = _norm(total_score, MAX_TOTAL_SCORE)
        second_monster_remaining = max(0.0, DEFAULT_SECOND_MONSTER_STEP - self.step_no)
        monster_speedup_remaining = max(0.0, DEFAULT_MONSTER_SPEEDUP_STEP - self.step_no)
        time_to_second_monster = _norm(second_monster_remaining, DEFAULT_SECOND_MONSTER_STEP)
        time_to_speedup = _norm(monster_speedup_remaining, DEFAULT_MONSTER_SPEEDUP_STEP)

        boundary_clearance = np.array(
            [
                _norm(hero_pos.get("x", 0.0), MAP_SIZE),
                _norm(MAP_SIZE - hero_pos.get("x", 0.0), MAP_SIZE),
                _norm(hero_pos.get("z", 0.0), MAP_SIZE),
                _norm(MAP_SIZE - hero_pos.get("z", 0.0), MAP_SIZE),
            ],
            dtype=np.float32,
        )

        legal_action = _parse_legal_action(legal_act_raw)
        legal_action_feat = np.array(legal_action, dtype=np.float32)
        mobility_score = float(np.mean(legal_action_feat))

        monsters = frame_state.get("monsters", [])
        monster_states = self._collect_monster_states(hero_pos, monsters)

        obstacle_patch = _extract_obstacle_patch(map_info)
        passable_patch = 1.0 - obstacle_patch

        danger_field = self._build_danger_field(monster_states)
        control_field, local_freedom = self._build_control_field(passable_patch, danger_field)
        _, openness_score, corridor_score, dead_end_risk, trap_risk = _compute_terrain_context(passable_patch, local_freedom)

        monster_pressures = sorted((state["pressure"] for state in monster_states), reverse=True)
        nearest_pressure = monster_pressures[0] if monster_pressures else 0.0
        second_pressure = monster_pressures[1] if len(monster_pressures) > 1 else 0.0

        combined_threat_vec = np.zeros(2, dtype=np.float32)
        for state in monster_states:
            combined_threat_vec += state["direction"] * state["pressure"]
        combined_threat_mag = float(np.clip(np.linalg.norm(combined_threat_vec), 0.0, 1.5))
        combined_threat_dir = (
            combined_threat_vec / (np.linalg.norm(combined_threat_vec) + 1e-6)
            if combined_threat_mag > 1e-6
            else np.zeros(2, dtype=np.float32)
        )

        if len(monster_states) >= 2:
            opposition = max(
                0.0,
                float(-np.dot(monster_states[0]["direction"], monster_states[1]["direction"])),
            )
            enclosure_pressure = float(
                np.clip(opposition * min(monster_pressures[0], monster_pressures[1]) + 0.35 * np.mean(monster_pressures[:2]), 0.0, 1.0)
            )
        else:
            enclosure_pressure = float(np.clip(0.4 * nearest_pressure, 0.0, 1.0))

        control_score = float(np.mean(control_field))
        control_collapse = float(np.clip(1.0 - control_score, 0.0, 1.0))
        monster_feature, nearest_monster_state, second_monster_state = _summarize_monster_states(monster_states, enclosure_pressure)

        treasures = frame_state.get("treasures", [])
        buffs = frame_state.get("speed_buffs", frame_state.get("buffs", []))
        treasure_states = self._collect_resource_states(
            hero_pos=hero_pos,
            resources=treasures,
            control_score=control_score,
            combined_threat_dir=combined_threat_dir,
            combined_threat_mag=combined_threat_mag,
            enclosure_pressure=enclosure_pressure,
        )
        buff_states = self._collect_resource_states(
            hero_pos=hero_pos,
            resources=buffs,
            control_score=control_score,
            combined_threat_dir=combined_threat_dir,
            combined_threat_mag=combined_threat_mag,
            enclosure_pressure=enclosure_pressure,
            active_key="is_active",
        )

        opportunity_field = self._build_opportunity_field(control_field, treasure_states + buff_states)
        direction_scores = self._compute_direction_scores(
            control_field=control_field,
            danger_field=danger_field,
            passable_patch=passable_patch,
            boundary_clearance=boundary_clearance,
            legal_action=legal_action,
        )

        best_escape = float(np.max(direction_scores))
        safety_score = float(
            np.clip(
                0.4 * control_score
                + 0.35 * best_escape
                + 0.15 * (1.0 - nearest_pressure)
                + 0.10 * mobility_score,
                0.0,
                1.0,
            )
        )

        treasure_feature, best_treasure_value, treasure_risk = _summarize_resource_states(treasure_states, direction_scores)
        _, best_buff_value, buff_risk = _summarize_resource_states(buff_states, direction_scores)

        best_opportunity = max(best_treasure_value, best_buff_value)
        risky_pursuit = float(np.clip(max(treasure_risk, buff_risk) * (1.0 - safety_score), 0.0, 1.0))
        action_trace = list(self.action_history[-(HISTORY_LIMIT - 1) :])
        try:
            action_to_record = int(last_action)
        except (TypeError, ValueError):
            action_to_record = -1
        if action_to_record >= 0:
            action_trace.append(action_to_record)
        stuck_score, loop_score = _compute_path_pattern_features(current_pos_vec, self.pos_history, action_trace[-6:])
        stage_progress = max(1.0 - time_to_second_monster, 1.0 - time_to_speedup)
        high_pressure_stage = _clip01(
            0.45 * stage_progress
            + 0.35 * enclosure_pressure
            + 0.20 * max(nearest_pressure, second_pressure)
        )
        greed_window = _clip01(
            (1.0 - high_pressure_stage)
            * (0.55 + 0.45 * flash_ready)
            * (0.35 + 0.65 * (1.0 - trap_risk))
        )
        preserve_mode = _clip01(max(high_pressure_stage, nearest_pressure, trap_risk))
        nearest_treasure_dist_score = float(treasure_feature[3]) if treasure_feature[0] > 0.0 else 0.0
        nearest_monster_dist_norm = nearest_monster_state["dist_norm"] if nearest_monster_state is not None else 1.0
        safe_gate = _clip01(safety_score * (1.0 - max(treasure_risk, trap_risk)))
        enclosure_penalty = enclosure_pressure
        dead_corner_penalty = dead_end_risk
        danger_penalty = _clip01(0.45 * nearest_pressure + 0.25 * control_collapse + 0.20 * trap_risk + 0.10 * risky_pursuit)
        pre_speedup_imminence = 0.0
        if monster_speedup_remaining > 0.0:
            pre_speedup_imminence = _clip01(1.0 - time_to_speedup)
        pre_speedup_buffer = _clip01(pre_speedup_imminence * (0.4 + 0.6 * safety_score) * (0.5 + 0.5 * flash_ready))
        late_survival_gate = _clip01((step_norm - 0.6) / 0.4)
        corridor_metric = _clip01(corridor_score * (0.5 + 0.5 * safety_score))
        prev_pos_vec = self.pos_history[-1] if self.pos_history else None
        move_dist = float(np.linalg.norm(current_pos_vec - prev_pos_vec)) if prev_pos_vec is not None else 0.0
        invalid_move_event = 1.0 if (self.state_initialized and action_to_record >= 0 and move_dist < 0.25) else 0.0
        buff_gain = 1.0 if (
            self.state_initialized
            and (
                (buff_active > self.last_buff_active + 0.5)
                or (buff_remain_norm > self.last_buff_remain_norm + 0.20)
            )
        ) else 0.0
        flash_used_event = 1.0 if (
            self.state_initialized
            and self.last_flash_ready > 0.5
            and flash_cd_norm > self.last_flash_cd_norm + 0.05
        ) else 0.0
        danger_relief = _clip01(
            max(self.last_danger_penalty - danger_penalty, 0.0)
            + max(safety_score - self.last_safety_score, 0.0)
        )
        flash_escape_event = 1.0 if (flash_used_event > 0.5 and danger_relief > 0.15) else 0.0
        flash_waste_event = 1.0 if (flash_used_event > 0.5 and flash_escape_event < 0.5) else 0.0
        second_monster_penalty = second_monster_state["pressure"] if second_monster_state is not None else 0.0
        revisit_penalty = loop_score

        hero_main_feature = np.concatenate(
            [
                boundary_clearance,
                np.array(
                    [
                        step_norm,
                        total_score_norm,
                        time_to_second_monster,
                        time_to_speedup,
                        high_pressure_stage,
                        stuck_score,
                        loop_score,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        skill_buff_feature = np.array(
            [
                flash_ready,
                flash_cd_norm,
                buff_active,
                buff_remain_norm,
                greed_window,
                preserve_mode,
            ],
            dtype=np.float32,
        )
        local_map_feature = np.concatenate(
            [
                np.array(
                    [
                        openness_score,
                        corridor_score,
                        dead_end_risk,
                        trap_risk,
                    ],
                    dtype=np.float32,
                ),
                direction_scores.astype(np.float32),
            ]
        )

        summary_feature = np.concatenate(
            [
                hero_main_feature.astype(np.float32),
                treasure_feature.astype(np.float32),
                monster_feature.astype(np.float32),
                skill_buff_feature,
                local_map_feature.astype(np.float32),
            ]
        ).astype(np.float32)

        trap_field = passable_patch * np.clip(1.0 - local_freedom, 0.0, 1.0)
        spatial_feature = np.stack(
            [
                passable_patch,
                local_freedom,
                danger_field,
                trap_field,
                control_field,
                opportunity_field,
            ],
            axis=0,
        ).astype(np.float32)

        feature = np.concatenate(
            [
                spatial_feature.reshape(-1),
                summary_feature,
            ]
        ).astype(np.float32)

        reward = np.array([0.0], dtype=np.float32)
        if self.state_initialized:
            step_gain = float(np.clip(step_score - self.last_step_score, 0.0, 5.0))
            treasure_gain = float(np.clip(treasure_score - self.last_treasure_score, 0.0, 50.0))
            treasure_approach_gain = float(np.clip(nearest_treasure_dist_score - self.last_nearest_treasure_dist_score, 0.0, 1.0))
            monster_dist_gain = float(np.clip(nearest_monster_dist_norm - self.last_nearest_monster_dist_norm, 0.0, 1.0))
            pre_speedup_buffer_gain = float(np.clip(pre_speedup_buffer - self.last_pre_speedup_buffer, 0.0, 1.0))
            corridor_gain = float(np.clip(corridor_metric - self.last_corridor_metric, 0.0, 1.0))
            milestone_reward = 0.0
            if self.step_no == 200:
                milestone_reward = 1.0
            if self.step_no == 500:
                milestone_reward = 2.0
            if self.step_no == 800:
                milestone_reward = 5.0

            survival_reward = 0.01
            step_score_reward = 0.006 * step_gain
            treasure_score_reward = 1.0 * treasure_gain
            buff_reward = 0.08 * buff_gain
            treasure_approach_reward = 0.15 * treasure_approach_gain
            monster_distance_reward = 0.04 * monster_dist_gain
            pre_speedup_reward = 0.05 * pre_speedup_buffer_gain
            late_survival_reward = 0.006 * late_survival_gate
            corridor_reward = 0.03 * corridor_gain
            flash_escape_reward = 0.3 * flash_escape_event

            punishment_scale = 0.4

            enclosure_cost = -0.04 * enclosure_penalty
            dead_corner_cost = -0.04 * dead_corner_penalty
            danger_cost = -0.05 * danger_penalty
            enclosure_cost *= punishment_scale
            danger_cost *= punishment_scale
            dead_corner_cost *= punishment_scale
            invalid_move_cost = -0.01 * invalid_move_event
            revisit_cost = -0.03 * revisit_penalty
            flash_waste_cost = -0.10 * flash_waste_event
            second_monster_cost = -0.03 * second_monster_penalty

            reward[0] = (
                survival_reward
                + step_score_reward
                + treasure_score_reward
                + buff_reward
                + treasure_approach_reward
                + monster_distance_reward
                + pre_speedup_reward
                + late_survival_reward
                + corridor_reward
                + flash_escape_reward
                + milestone_reward
                + enclosure_cost
                + dead_corner_cost
                + danger_cost
                + invalid_move_cost
                + revisit_cost
                + flash_waste_cost
                + second_monster_cost
            )

        self.last_safety_score = safety_score
        self.last_best_escape = best_escape
        self.last_control_collapse = control_collapse
        self.last_mobility_score = mobility_score
        self.last_best_opportunity = best_opportunity
        self.last_risky_pursuit = risky_pursuit
        self.last_step_score = step_score
        self.last_treasure_score = treasure_score
        self.last_total_score = total_score
        self.last_nearest_treasure_dist_score = nearest_treasure_dist_score
        self.last_nearest_monster_dist_norm = nearest_monster_dist_norm
        self.last_pre_speedup_buffer = pre_speedup_buffer
        self.last_corridor_metric = corridor_metric
        self.last_danger_penalty = danger_penalty
        self.last_flash_cd_norm = flash_cd_norm
        self.last_flash_ready = flash_ready
        self.last_buff_active = buff_active
        self.last_buff_remain_norm = buff_remain_norm
        if action_to_record >= 0:
            self.action_history.append(action_to_record)
            self.action_history = self.action_history[-HISTORY_LIMIT:]
        self.pos_history.append(current_pos_vec)
        self.pos_history = self.pos_history[-HISTORY_LIMIT:]
        self.state_initialized = True

        return feature, legal_action, reward.tolist()
