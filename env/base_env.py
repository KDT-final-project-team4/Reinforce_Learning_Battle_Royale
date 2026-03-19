"""기반 환경 클래스 — 배틀로얄 / 넥서스 모드 공통 로직

이동, 공격, 힐, 아이템 획득, 상대 정책 관리 등
모드에 무관한 메커닉을 공유한다.
"""

from abc import abstractmethod

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from env.agent import Agent, ROLE_TANK, ROLE_DEALER, ROLE_HEALER, ALL_ROLES
from env.map_generator import (
    generate_map, place_agents, get_empty_positions,
    TILE_EMPTY, TILE_WALL, TILE_ZONE,
)
from env.items import ItemManager, ItemType
from env.zone import ZoneManager


# 행동 정의 — Discrete(9) (글로벌 인덱스)
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_MELEE = 4       # 근접 공격 (인접 1칸, 전 역할)
ACTION_RANGED_H = 5    # 원거리 공격 - 가로 (딜러 전용)
ACTION_RANGED_V = 6    # 원거리 공격 - 세로 (딜러 전용)
ACTION_HEAL = 7        # 아군 회복 (힐러 전용)
ACTION_STAY = 8

NUM_ACTIONS = 9

# 역할별 행동 매핑: 역할 로컬 인덱스 → 글로벌 행동 인덱스
ACTION_RANGED = ACTION_RANGED_H  # 자동 조준 (기존 상수 재활용)

ROLE_ACTION_MAP = {
    ROLE_TANK:   [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_MELEE, ACTION_STAY],
    ROLE_DEALER: [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_MELEE, ACTION_RANGED, ACTION_STAY],
    ROLE_HEALER: [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_MELEE, ACTION_HEAL, ACTION_STAY],
}

MOVE_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


class BaseBattleEnv(gym.Env):
    """그리드 기반 전투 환경의 공통 기반 클래스.

    이동, 공격, 힐, 아이템 등 모드 무관 메커닉을 구현한다.
    reset(), step(), _get_observation(), _calculate_reward(), _check_game_over()는
    서브클래스에서 구현한다.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self, config: dict | None = None, render_mode: str | None = None,
                 opponent_policy=None, learning_role: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # 설정 로드
        if config is None:
            with open("config/default.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        self.config = config

        map_cfg = config["map"]
        self.map_width = map_cfg["width"]
        self.map_height = map_cfg["height"]
        self.wall_count = map_cfg["wall_count"]

        agent_cfg = config["agent"]
        self.num_agents = agent_cfg["count"]
        self.view_range = agent_cfg["view_range"]

        game_cfg = config["game"]
        self.max_steps = game_cfg["max_steps"]

        reward_cfg = config["reward"]
        self.reward_cfg = reward_cfg

        # 팀 설정
        team_cfg = config.get("team", {})
        self.num_teams = team_cfg.get("num_teams", 3)
        self.agents_per_team = team_cfg.get("agents_per_team", 2)
        self.friendly_fire = team_cfg.get("friendly_fire", False)

        # 역할 설정
        self.roles_cfg = config.get("roles", {})

        # 역할별 보상 설정 (base reward_cfg를 역할 오버라이드로 병합)
        role_rewards_cfg = config.get("role_rewards", {})
        self._role_reward_cfgs: dict[str, dict] = {}

        # 멀티 정책 지원: learning_role이 설정되면 에이전트 0의 역할을 고정
        self.learning_role = learning_role
        if learning_role and learning_role in ROLE_ACTION_MAP:
            self._role_action_map = ROLE_ACTION_MAP[learning_role]
            self.action_space = spaces.Discrete(len(self._role_action_map))
        else:
            self._role_action_map = None
            self.action_space = spaces.Discrete(NUM_ACTIONS)

        # 역할별 보상 병합
        _base_reward = config["reward"]
        for role in ALL_ROLES:
            merged = dict(_base_reward)
            merged.update(role_rewards_cfg.get(role, {}))
            self._role_reward_cfgs[role] = merged

        # 상대 정책 (기본: 랜덤)
        self.opponent_policy = opponent_policy
        self._opponent_path = None

        # 역할별 상대 정책 (멀티 정책용)
        self._opponent_policies: dict[str, object] = {}

        # 내부 상태
        self.grid = None
        self.agents: list[Agent] = []
        self.current_step = 0
        self.rng = np.random.default_rng()

        # 매니저
        self.item_manager = None
        self.zone_manager = None

    # ─────────────────── 서브클래스에서 구현 ──────────────────

    @abstractmethod
    def reset(self, seed=None, options=None):
        """환경을 초기화하고 (observation, info)를 반환한다."""
        ...

    @abstractmethod
    def step(self, action):
        """한 스텝을 진행하고 (obs, reward, terminated, truncated, info)를 반환한다."""
        ...

    @abstractmethod
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """에이전트의 관측을 반환한다."""
        ...

    @abstractmethod
    def _calculate_reward(self, agent_idx: int) -> float:
        """에이전트의 보상을 계산한다."""
        ...

    @abstractmethod
    def _check_game_over(self) -> bool:
        """게임 종료 여부를 반환한다."""
        ...

    # ────────────────── 상대 정책 관리 ──────────────────────

    def set_opponent_path(self, path: str):
        """Self-play 상대 모델 파일 경로를 설정하고 즉시 로드한다. (단일 정책 호환)"""
        from stable_baselines3 import PPO, DQN
        self._opponent_path = path
        try:
            try:
                self.opponent_policy = PPO.load(path, device="cpu")
            except Exception:
                self.opponent_policy = DQN.load(path, device="cpu")
        except Exception:
            pass

    def set_opponent_paths(self, paths: dict):
        """Self-play 역할별 상대 모델 경로를 설정한다. paths = {role: model_path}"""
        from stable_baselines3 import PPO, DQN
        for role, path in paths.items():
            if not path:
                continue
            try:
                try:
                    self._opponent_policies[role] = PPO.load(path, device="cpu")
                except Exception:
                    self._opponent_policies[role] = DQN.load(path, device="cpu")
            except Exception:
                pass

    def _get_all_opponent_actions(self) -> list[int]:
        """상대 에이전트 행동을 역할별로 배치 predict해 반환한다.
        같은 역할끼리 obs를 묶어 1회 predict → 개별 호출 대비 ~N배 빠름."""
        # 살아있는 상대만 역할별로 그룹화
        role_groups: dict[str, list[int]] = {}
        for i in range(1, self.num_agents):
            if not self.agents[i].alive:
                continue
            role = self.agents[i].role
            if role in self._opponent_policies:
                role_groups.setdefault(role, []).append(i)

        # 역할별 배치 predict
        batched: dict[int, int] = {}
        for role, indices in role_groups.items():
            try:
                obs_list = [self._get_observation(idx) for idx in indices]
                obs_batch = np.stack(obs_list, axis=0)
                role_actions, _ = self._opponent_policies[role].predict(
                    obs_batch, deterministic=False)
                # 단일 에이전트면 스칼라로 올 수 있으므로 배열 보장
                role_actions = np.atleast_1d(role_actions)
                role_map = ROLE_ACTION_MAP[role]
                for j, idx in enumerate(indices):
                    action_idx = int(role_actions[j])
                    batched[idx] = (role_map[action_idx]
                                    if 0 <= action_idx < len(role_map) else ACTION_STAY)
            except Exception:
                # predict 실패 시 해당 역할 에이전트는 랜덤 행동
                for idx in indices:
                    batched[idx] = int(self.rng.integers(0, NUM_ACTIONS))

        # 전체 순서대로 조립
        result = []
        for i in range(1, self.num_agents):
            if not self.agents[i].alive:
                result.append(ACTION_STAY)
            elif i in batched:
                result.append(batched[i])
            elif self.opponent_policy is not None:
                obs = self._get_observation(i)
                action, _ = self.opponent_policy.predict(obs, deterministic=False)
                result.append(int(action))
            else:
                result.append(int(self.rng.integers(0, NUM_ACTIONS)))
        return result

    # ─────────────────────────── 이동 ────────────────────────

    def _process_movements(self, actions: list[int]):
        """모든 에이전트의 이동을 동시에 처리한다."""
        proposed = {}

        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            act = actions[i]
            if act in MOVE_DELTAS:
                dy, dx = MOVE_DELTAS[act]
                new_y = agent.y + dy
                new_x = agent.x + dx

                if not (0 <= new_y < self.map_height and 0 <= new_x < self.map_width):
                    proposed[i] = (agent.y, agent.x)
                    self._step_events[i].append("wall_bump")
                    continue
                if self.grid[new_y, new_x] == TILE_WALL:
                    proposed[i] = (agent.y, agent.x)
                    self._step_events[i].append("wall_bump")
                    continue

                proposed[i] = (new_y, new_x)
            else:
                proposed[i] = (agent.y, agent.x)

        # 충돌 판정
        final_positions = dict(proposed)
        position_counts: dict[tuple[int, int], list[int]] = {}
        for i, pos in proposed.items():
            position_counts.setdefault(pos, []).append(i)

        for pos, agent_ids in position_counts.items():
            if len(agent_ids) > 1:
                for aid in agent_ids:
                    final_positions[aid] = (self.agents[aid].y, self.agents[aid].x)

        for i, (new_y, new_x) in final_positions.items():
            self.agents[i].y = new_y
            self.agents[i].x = new_x

    # ─────────────────────────── 공격 ────────────────────────

    def _process_attacks(self, actions: list[int]):
        """근접 및 원거리 공격을 처리한다."""
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue

            act = actions[i]

            # 근접 공격
            if act == ACTION_MELEE:
                if not agent.can_attack:
                    self._step_events[i].append("attack_cooldown")
                    continue
                agent.attack_count += 1
                agent.attack_cooldown = agent.attack_cooldown_steps
                target = self._find_melee_target(agent)
                if target is None:
                    self._step_events[i].append("attack_miss")
                    continue

                actual_dmg = target.take_damage(agent.attack)
                agent.attack_hits += 1
                self._step_events[i].append(("damage_dealt", actual_dmg))
                self._render_events.append({
                    "type": "melee_hit",
                    "attacker_x": agent.x, "attacker_y": agent.y,
                    "target_x": target.x, "target_y": target.y,
                })

                if not target.alive:
                    agent.kills += 1
                    target.death_step = self.current_step
                    self._step_events[i].append("kill")
                    self._step_events[target.agent_id].append("death")
                    self._render_events.append({
                        "type": "death", "x": target.x, "y": target.y,
                    })

            # 원거리 공격 (딜러 전용 — 자동 조준)
            elif act == ACTION_RANGED_H:  # ACTION_RANGED (자동 조준)
                if not agent.can_ranged_attack:
                    continue  # invalid_action은 별도 처리
                if not agent.can_attack:
                    self._step_events[i].append("attack_cooldown")
                    continue

                agent.attack_count += 1
                agent.attack_cooldown = agent.attack_cooldown_steps
                target = self._find_ranged_target(agent)
                if target is None:
                    self._step_events[i].append("ranged_miss")
                    continue

                # 원거리 데미지 = ATK × multiplier
                multiplier = self.roles_cfg.get("dealer", {}).get(
                    "ranged_damage_multiplier", 0.8)
                ranged_dmg = max(1, int(agent.attack * multiplier))
                actual_dmg = target.take_damage(ranged_dmg)
                agent.attack_hits += 1
                self._step_events[i].append(("damage_dealt", actual_dmg))
                self._render_events.append({
                    "type": "ranged_hit",
                    "attacker_x": agent.x, "attacker_y": agent.y,
                    "target_x": target.x, "target_y": target.y,
                })

                if not target.alive:
                    agent.kills += 1
                    target.death_step = self.current_step
                    self._step_events[i].append("kill")
                    self._step_events[target.agent_id].append("death")
                    self._render_events.append({
                        "type": "death", "x": target.x, "y": target.y,
                    })

    def _find_melee_target(self, attacker: Agent) -> Agent | None:
        """인접 8방향(대각선 포함)에서 가장 가까운 살아있는 적을 찾는다 (아군 제외)."""
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ty, tx = attacker.y + dy, attacker.x + dx
            if not (0 <= ty < self.map_height and 0 <= tx < self.map_width):
                continue
            for agent in self.agents:
                if not agent.alive or agent.agent_id == attacker.agent_id:
                    continue
                # 아군 공격 불가
                if not self.friendly_fire and attacker.is_teammate(agent):
                    continue
                if agent.y == ty and agent.x == tx:
                    return agent
        return None

    def _find_ranged_target(self, attacker: Agent) -> Agent | None:
        """4방향(상하좌우) 모두 스캔하여 가장 가까운 적을 자동 타겟.
        벽에 막히면 해당 방향 스캔 중단. 아군은 타겟 안 됨.
        """
        attack_range = attacker.attack_range
        closest_target = None
        closest_dist = float("inf")

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, attack_range + 1):
                ty = attacker.y + dy * dist
                tx = attacker.x + dx * dist

                if not (0 <= ty < self.map_height and 0 <= tx < self.map_width):
                    break
                if self.grid[ty, tx] == TILE_WALL:
                    break

                for agent in self.agents:
                    if not agent.alive or agent.agent_id == attacker.agent_id:
                        continue
                    if not self.friendly_fire and attacker.is_teammate(agent):
                        continue
                    if agent.y == ty and agent.x == tx:
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_target = agent
                # 적을 찾았으면 이 방향에서는 더 멀리 볼 필요 없음
                if closest_target is not None and closest_dist == dist:
                    break

        return closest_target

    # ─────────────────────────── 힐 ──────────────────────────

    def _process_heals(self, actions: list[int]):
        """힐러의 아군 회복 행동을 처리한다."""
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            if actions[i] != ACTION_HEAL:
                continue
            if not agent.can_heal:
                # 힐러지만 쿨다운 중인 경우
                if agent.role == ROLE_HEALER and agent.heal_cooldown > 0:
                    self._step_events[i].append("heal_cooldown")
                continue  # invalid_action은 별도 처리

            agent.heal_count += 1
            heal_amount = self.roles_cfg.get("healer", {}).get("heal_amount", 20)

            # 인접 4방향에서 팀원 찾기
            target = self._find_heal_target(agent)
            if target is None:
                self._step_events[i].append("attack_miss")  # 힐 빗나감
                continue

            actual_heal = target.heal(heal_amount)
            if actual_heal > 0:
                agent.heal_cooldown = agent.heal_cooldown_steps
                hp_ratio = target.hp / target.max_hp
                self._step_events[i].append(("heal_ally", actual_heal, hp_ratio))
                self._render_events.append({
                    "type": "heal",
                    "healer_x": agent.x, "healer_y": agent.y,
                    "target_x": target.x, "target_y": target.y,
                })
            else:
                self._step_events[i].append("attack_miss")  # 만피 팀원

    def _find_heal_target(self, healer: Agent) -> Agent | None:
        """인접 8방향(대각선 포함)에서 가장 HP가 낮은 살아있는 팀원을 찾는다."""
        best_target = None
        lowest_hp_ratio = 1.0

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ty, tx = healer.y + dy, healer.x + dx
            if not (0 <= ty < self.map_height and 0 <= tx < self.map_width):
                continue
            for agent in self.agents:
                if not agent.alive or agent.agent_id == healer.agent_id:
                    continue
                if not healer.is_teammate(agent):
                    continue
                if agent.y == ty and agent.x == tx:
                    hp_ratio = agent.hp / agent.max_hp
                    if hp_ratio < 1.0 and hp_ratio < lowest_hp_ratio:
                        lowest_hp_ratio = hp_ratio
                        best_target = agent
        return best_target

    # ─────────────────────────── 유효성 검사 ──────────────────

    def _check_invalid_actions(self, actions: list[int]):
        """역할 외 행동을 사용한 에이전트에게 이벤트를 추가한다."""
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            act = actions[i]
            # 딜러가 아닌데 원거리 공격 (ACTION_RANGED_V는 하위호환)
            if act in (ACTION_RANGED_H, ACTION_RANGED_V) and not agent.can_ranged_attack:
                self._step_events[i].append("invalid_action")
            # 힐러가 아닌데 힐 (쿨다운 중인 힐러는 제외)
            elif act == ACTION_HEAL and not agent.can_heal:
                if agent.role != ROLE_HEALER:
                    self._step_events[i].append("invalid_action")

    # ─────────────────────────── 아이템 ──────────────────────

    def _process_items(self):
        """에이전트가 아이템 칸에 있으면 자동 획득한다."""
        if not self.item_manager.enabled:
            return

        for agent in self.agents:
            if not agent.alive:
                continue
            item = self.item_manager.pickup(agent.y, agent.x)
            if item == ItemType.NONE:
                continue

            agent.items_collected += 1
            hp_before = agent.hp / agent.max_hp
            self._step_events[agent.agent_id].append(("item_pickup", item, hp_before))

            if item == ItemType.POTION:
                heal_amount = int(self.config["items"]["potion_heal"] * agent.potion_multiplier)
                agent.heal(heal_amount)
            elif item == ItemType.WEAPON:
                agent.add_attack(self.config["items"]["weapon_attack_bonus"])
            elif item == ItemType.ARMOR:
                agent.add_defense(self.config["items"]["armor_defense_bonus"])

    # ─────────────────────────── 거리 유틸 ────────────────────

    def _get_nearest_enemy_dist(self, agent_idx: int) -> float:
        """가장 가까운 살아있는 적(다른 팀)과의 체비셰프 거리를 반환한다."""
        agent = self.agents[agent_idx]
        min_dist = float("inf")
        for other in self.agents:
            if other.agent_id != agent_idx and other.alive:
                if not agent.is_teammate(other):
                    dist = max(abs(agent.y - other.y), abs(agent.x - other.x))
                    min_dist = min(min_dist, dist)
        return min_dist

    def _get_nearest_low_hp_teammate_dist(self, agent_idx: int) -> float:
        """체력이 만피가 아닌 가장 가까운 살아있는 팀원과의 체비셰프 거리를 반환한다."""
        agent = self.agents[agent_idx]
        min_dist = float("inf")
        for other in self.agents:
            if other.agent_id != agent_idx and other.alive and agent.is_teammate(other):
                if other.hp < other.max_hp:  # 만피 아닌 팀원만
                    dist = max(abs(agent.y - other.y), abs(agent.x - other.x))
                    min_dist = min(min_dist, dist)
        return min_dist

    # ─────────────────────────── 렌더링 ──────────────────────

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_ansi_print()

    def _render_ansi(self) -> str:
        """텍스트 기반 렌더링"""
        lines = [f"Step: {self.current_step}"]
        for r in range(self.map_height):
            row = ""
            for c in range(self.map_width):
                agent_here = None
                for a in self.agents:
                    if a.alive and a.y == r and a.x == c:
                        agent_here = a
                        break

                if agent_here is not None:
                    row += f"A{agent_here.agent_id}"
                elif self.grid[r, c] == TILE_WALL:
                    row += "##"
                elif self.grid[r, c] == TILE_ZONE:
                    row += "~~"
                else:
                    if self.item_manager and self.item_manager.enabled:
                        item = self.item_manager.item_grid[r, c]
                        if item == ItemType.POTION:
                            row += " P"
                        elif item == ItemType.WEAPON:
                            row += " W"
                        elif item == ItemType.ARMOR:
                            row += " A"
                        else:
                            row += " ."
                    else:
                        row += " ."
            lines.append(row)

        for a in self.agents:
            status = "DEAD" if not a.alive else f"HP:{a.hp} ATK:{a.attack} DEF:{a.defense}"
            role_char = a.role[0].upper()
            lines.append(f"  A{a.agent_id}(T{a.team_id}/{role_char}): {status}")

        return "\n".join(lines)

    def _render_ansi_print(self):
        print(self._render_ansi())

    def close(self):
        pass
