"""Gymnasium 커스텀 배틀로얄 환경 (Phase 4)

SB3 호환을 위해 단일 에이전트 인터페이스로 설계.
학습 에이전트(agents[0])가 행동을 선택하고,
상대 에이전트(agents[1..N])는 opponent_policy로 행동한다.
Phase 4: 6인 3팀(2:2:2), 역할 시스템(탱커/딜러/힐러), 원거리 공격/힐.
"""

from gymnasium import spaces
import numpy as np

from env.agent import Agent, ROLE_TANK, ROLE_DEALER, ROLE_HEALER, ALL_ROLES
from env.map_generator import (
    generate_map, place_agents, get_empty_positions,
    TILE_EMPTY, TILE_WALL, TILE_ZONE,
)
from env.items import ItemManager, ItemType
from env.zone import ZoneManager
from env.base_env import (
    BaseBattleEnv,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_MELEE, ACTION_RANGED_H, ACTION_RANGED_V, ACTION_HEAL, ACTION_STAY,
    NUM_ACTIONS, ROLE_ACTION_MAP, MOVE_DELTAS,
)

# 하위 호환 별칭
ACTION_ATTACK = ACTION_MELEE


class BattleRoyaleEnv(BaseBattleEnv):
    """2D 배틀로얄 강화학습 환경 — Phase 4 팀 배틀"""

    def __init__(self, config: dict | None = None, render_mode: str | None = None,
                 opponent_policy=None, learning_role: str | None = None):
        super().__init__(config=config, render_mode=render_mode,
                         opponent_policy=opponent_policy, learning_role=learning_role)

        # 관측 공간 계산
        # 4채널 시야 (벽/독가스, 아군, 적군, 아이템)
        view_size = self.view_range
        num_channels = 4
        grid_obs = view_size * view_size * num_channels  # 9*9*4 = 324

        # 자기 스탯: 7 (hp, atk, def, is_tank, is_dealer, is_healer, range_norm)
        self_stats = 7

        # 가장 가까운 적 방향: 5 (dy, dx, adjacent, in_range, ranged_aligned)
        nearest_enemy = 5

        # 2번째 가까운 적 방향: 5
        second_enemy = 5

        # 팀원 정보: 5 (alive, hp, dy, dx, role_encoded)
        teammate_info = 5

        # 게임 상태: 3 (alive_enemy_ratio, alive_ally_count, visible_enemy_ratio)
        game_state = 3

        obs_size = grid_obs + self_stats + nearest_enemy + second_enemy + teammate_info + game_state
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # 팀 탈락 추적
        self._team_elimination_step: dict[int, int] = {}  # team_id -> elimination step

    def reset(self, seed=None, options=None):
        super(BaseBattleEnv, self).reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 맵 생성
        self.grid = generate_map(
            self.map_width, self.map_height, self.wall_count, self.rng
        )

        # 팀/역할 배정 및 에이전트 배치
        positions = place_agents(self.grid, self.num_agents, self.rng)
        self.agents = []

        # 팀별 역할 배정
        team_roles = {}
        if self.learning_role:
            # 멀티 정책: 팀 0의 첫 번째 에이전트(agent 0)는 learning_role 고정
            other_roles = [r for r in ALL_ROLES if r != self.learning_role]
            teammate_role = str(self.rng.choice(other_roles))
            team_roles[0] = [self.learning_role, teammate_role]
            for t in range(1, self.num_teams):
                roles = list(self.rng.choice(ALL_ROLES, size=self.agents_per_team, replace=False))
                team_roles[t] = roles
        else:
            # 단일 정책: 기존 방식
            for t in range(self.num_teams):
                roles = list(self.rng.choice(ALL_ROLES, size=self.agents_per_team, replace=False))
                team_roles[t] = roles

        for i, (y, x) in enumerate(positions):
            team_id = i // self.agents_per_team
            role_idx = i % self.agents_per_team
            role = team_roles[team_id][role_idx]
            self.agents.append(Agent(i, x, y, self.config, role=role, team_id=team_id))

        # 매니저 초기화
        grid_shape = (self.map_height, self.map_width)
        self.item_manager = ItemManager(self.config, grid_shape, self.rng)
        self.item_manager.reset(self.grid)
        if self.item_manager.enabled:
            self.item_manager._spawn_items(self.grid)
        self.zone_manager = ZoneManager(self.config, grid_shape)
        self.zone_manager.reset()

        self.current_step = 0

        # 팀 탈락 추적 초기화
        self._team_elimination_step = {}

        # 전투 추적
        self._last_combat_step = 0
        self._consecutive_hits = 0

        # 거리 추적 초기화
        self._prev_enemy_dist = float("inf")
        self._prev_teammate_dist = float("inf")

        # 반복 이동 감지용 행동 히스토리
        self._action_history = []

        # 전투 이탈 감지용
        self._recent_combat_steps = []

        # 이벤트 추적
        self._step_events = {}

        # 보상 요소별 기록
        self._last_reward_details = {}

        observation = self._get_observation(0)
        info = self._get_info()
        return observation, info

    def step(self, action):
        """학습 에이전트(0번)의 행동을 받아 한 스텝을 진행한다."""
        self.current_step += 1
        self._step_events = {i: [] for i in range(self.num_agents)}
        self._render_events: list[dict] = []  # 렌더링용 이벤트 (위치 정보 포함)

        # 역할별 행동 매핑: 로컬 인덱스 → 글로벌 인덱스
        if self._role_action_map is not None:
            global_action = self._role_action_map[action]
        else:
            global_action = action

        # 접근 보상 계산을 위해 이전 거리 저장
        self._prev_enemy_dist = self._get_nearest_enemy_dist(0)
        self._prev_teammate_dist = self._get_nearest_low_hp_teammate_dist(0)

        # 상대 에이전트 행동 결정 (역할별 배치 predict)
        actions = [global_action]
        opp_actions = self._get_all_opponent_actions()
        actions.extend(opp_actions)

        # 1. 이동 처리
        self._process_movements(actions)

        # 2. 공격 처리 (근접 + 원거리)
        self._process_attacks(actions)

        # 3. 힐 처리
        self._process_heals(actions)

        # 4. 아이템 획득 처리
        self._process_items()

        # 5. 자기장 업데이트
        self._process_zone()

        # 6. 대기 행동 이벤트
        if global_action == ACTION_STAY:
            self._step_events[0].append("idle")

        # 7. 역할 외 행동 사용 시 invalid_action
        self._check_invalid_actions(actions)

        # 8. 생존 이벤트
        for i, agent in enumerate(self.agents):
            if agent.alive:
                self._step_events[i].append("survive")

        # 9. 팀 탈락 체크
        self._check_team_eliminations()

        # 매니저 스텝
        self.item_manager.step(self.grid)
        self.zone_manager.step(self.grid, self.current_step)

        # 쿨다운 감소 (공격 + 힐)
        for agent in self.agents:
            if agent.alive:
                if agent.attack_cooldown > 0:
                    agent.attack_cooldown -= 1
                if agent.heal_cooldown > 0:
                    agent.heal_cooldown -= 1

        # 행동 히스토리 업데이트
        self._action_history.append(action)
        if len(self._action_history) > 3:
            self._action_history.pop(0)

        # 보상 계산
        reward = self._calculate_reward(0)

        # 종료 조건
        terminated = self._check_game_over()
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation(0)
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    # ─────────────────────────── 자기장 ──────────────────────

    def _process_zone(self):
        """자기장 데미지를 처리한다."""
        if not self.zone_manager.enabled:
            return

        for agent in self.agents:
            if not agent.alive:
                continue
            if self.zone_manager.is_in_zone(self.grid, agent.y, agent.x):
                agent.take_damage(self.zone_manager.damage_per_step)
                self._step_events[agent.agent_id].append("zone_damage")
                if not agent.alive:
                    agent.death_step = self.current_step
                    self._step_events[agent.agent_id].append("death")
                    self._render_events.append({
                        "type": "death", "x": agent.x, "y": agent.y,
                    })

    # ─────────────────────────── 팀 탈락 ─────────────────────

    def _check_team_eliminations(self):
        """팀 전멸 여부를 체크하고 이벤트를 기록한다."""
        for t in range(self.num_teams):
            if t in self._team_elimination_step:
                continue  # 이미 탈락한 팀
            team_alive = any(
                a.alive for a in self.agents if a.team_id == t
            )
            if not team_alive:
                self._team_elimination_step[t] = self.current_step
                # 이 팀의 모든 에이전트에게 탈락 이벤트
                for a in self.agents:
                    if a.team_id == t:
                        self._step_events[a.agent_id].append("team_eliminated")
                    else:
                        pass

        # teammate_death: 같은 팀 멤버가 이번 스텝에 죽었는지 체크
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            for other in self.agents:
                if other.agent_id == agent.agent_id:
                    continue
                if agent.is_teammate(other) and not other.alive:
                    if other.death_step == self.current_step:
                        self._step_events[i].append("teammate_death")

    # ─────────────────────────── 관측 ────────────────────────

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """에이전트의 관측을 반환한다 (Phase 4: 4채널 시야 + 확장 정보)."""
        agent = self.agents[agent_idx]
        view = self.view_range
        half = view // 2

        # 4채널 시야 맵
        local_map = np.zeros((view, view, 4), dtype=np.float32)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy = agent.y + dy
                gx = agent.x + dx
                ly = dy + half
                lx = dx + half

                if not (0 <= gy < self.map_height and 0 <= gx < self.map_width):
                    local_map[ly, lx, 0] = 1.0  # 맵 밖 = 벽
                    continue

                tile = self.grid[gy, gx]
                # 채널 0: 벽/독가스
                if tile == TILE_WALL:
                    local_map[ly, lx, 0] = 1.0
                elif tile == TILE_ZONE:
                    local_map[ly, lx, 0] = 0.5

                # 채널 1: 아군 / 채널 2: 적군
                for other in self.agents:
                    if other.agent_id == agent_idx or not other.alive:
                        continue
                    if other.y == gy and other.x == gx:
                        if agent.is_teammate(other):
                            local_map[ly, lx, 1] = 1.0  # 아군
                        else:
                            local_map[ly, lx, 2] = 1.0  # 적군

                # 채널 3: 아이템
                if self.item_manager and self.item_manager.enabled:
                    item_val = self.item_manager.item_grid[gy, gx]
                    if item_val > 0:
                        local_map[ly, lx, 3] = item_val / 3.0

        # 자기 스탯 (7차원)
        stats = agent.get_stats()

        # 가장 가까운 적 방향: [dy, dx, adjacent, in_range, ranged_aligned]
        enemy_dir_1 = np.zeros(5, dtype=np.float32)
        enemies = []
        for other in self.agents:
            if other.agent_id != agent_idx and other.alive:
                if not agent.is_teammate(other):
                    dist = abs(agent.y - other.y) + abs(agent.x - other.x)
                    enemies.append((dist, other))
        enemies.sort(key=lambda e: e[0])

        max_dim = max(self.map_height, self.map_width)
        if len(enemies) >= 1:
            _, ne = enemies[0]
            enemy_dir_1[0] = (ne.y - agent.y) / max_dim
            enemy_dir_1[1] = (ne.x - agent.x) / max_dim
            dist1 = abs(agent.y - ne.y) + abs(agent.x - ne.x)
            enemy_dir_1[2] = 1.0 if dist1 == 1 else 0.0
            enemy_dir_1[3] = 1.0 if dist1 <= agent.attack_range else 0.0
            # ranged_aligned: 같은 행/열에 있고 사거리 내이면 1.0 (딜러 자동조준 힌트)
            if agent.attack_range > 1 and (ne.y == agent.y or ne.x == agent.x) and dist1 <= agent.attack_range:
                enemy_dir_1[4] = 1.0

        # 2번째 가까운 적 방향
        enemy_dir_2 = np.zeros(5, dtype=np.float32)
        if len(enemies) >= 2:
            _, se = enemies[1]
            enemy_dir_2[0] = (se.y - agent.y) / max_dim
            enemy_dir_2[1] = (se.x - agent.x) / max_dim
            dist2 = abs(agent.y - se.y) + abs(agent.x - se.x)
            enemy_dir_2[2] = 1.0 if dist2 == 1 else 0.0
            enemy_dir_2[3] = 1.0 if dist2 <= agent.attack_range else 0.0
            if agent.attack_range > 1 and (se.y == agent.y or se.x == agent.x) and dist2 <= agent.attack_range:
                enemy_dir_2[4] = 1.0

        # 팀원 정보: [alive, hp, dy, dx, role_encoded]
        teammate_info = np.zeros(5, dtype=np.float32)
        for other in self.agents:
            if other.agent_id != agent_idx and agent.is_teammate(other):
                teammate_info[0] = 1.0 if other.alive else 0.0
                teammate_info[1] = other.hp / other.max_hp if other.alive else 0.0
                teammate_info[2] = (other.y - agent.y) / max_dim
                teammate_info[3] = (other.x - agent.x) / max_dim
                # role encoded: tank=0.33, dealer=0.67, healer=1.0
                if other.role == ROLE_TANK:
                    teammate_info[4] = 0.33
                elif other.role == ROLE_DEALER:
                    teammate_info[4] = 0.67
                elif other.role == ROLE_HEALER:
                    teammate_info[4] = 1.0
                break  # 팀원은 1명

        # 게임 상태: [alive_enemy_ratio, alive_ally_count, visible_enemy_ratio]
        game_state = np.zeros(3, dtype=np.float32)

        # 적 팀 생존 비율
        total_enemies = sum(1 for a in self.agents if not agent.is_teammate(a)
                           and a.agent_id != agent_idx)
        alive_enemies = sum(1 for a in self.agents if not agent.is_teammate(a)
                           and a.agent_id != agent_idx and a.alive)
        game_state[0] = alive_enemies / max(total_enemies, 1)

        # 아군 생존 수 (자신 제외)
        alive_allies = sum(1 for a in self.agents
                          if a.agent_id != agent_idx and agent.is_teammate(a) and a.alive)
        game_state[1] = alive_allies / max(self.agents_per_team - 1, 1)

        # 시야 내 적 수
        visible_enemies = 0
        for other in self.agents:
            if other.agent_id != agent_idx and other.alive:
                if not agent.is_teammate(other):
                    if (abs(other.y - agent.y) <= half and
                            abs(other.x - agent.x) <= half):
                        visible_enemies += 1
        game_state[2] = visible_enemies / max(total_enemies, 1)

        # 평탄화
        obs = np.concatenate([
            local_map.flatten(), stats,
            enemy_dir_1, enemy_dir_2,
            teammate_info, game_state
        ])
        return obs

    # ─────────────────────────── 보상 ────────────────────────

    def _calculate_reward(self, agent_idx: int) -> float:
        """에이전트의 보상을 계산한다."""
        events = self._step_events.get(agent_idx, [])
        reward = 0.0
        agent = self.agents[agent_idx]
        # 역할별 오버라이드 보상 설정 사용
        cfg = self._role_reward_cfgs.get(agent.role, self.reward_cfg)
        dealt_damage_this_step = False

        rd = {}  # reward_details

        for event in events:
            if isinstance(event, tuple) and event[0] == "damage_dealt":
                _, value = event
                r = cfg["damage_dealt"]
                reward += r
                rd["damage_dealt"] = rd.get("damage_dealt", 0.0) + r
                dealt_damage_this_step = True
            elif isinstance(event, tuple) and event[0] == "item_pickup":
                _, item_type, hp_before = event
                threshold = cfg.get("low_hp_potion_threshold", 0.5)
                if item_type == ItemType.POTION and hp_before < threshold:
                    r = cfg.get("low_hp_potion_bonus", cfg["item_pickup"])
                else:
                    r = cfg["item_pickup"]
                reward += r
                rd["item_pickup"] = rd.get("item_pickup", 0.0) + r
            elif isinstance(event, tuple) and event[0] == "heal_ally":
                _, heal_amount, hp_ratio = event
                heal_threshold = cfg.get("heal_hp_threshold", 0.4)
                if hp_ratio < heal_threshold:
                    r = cfg.get("heal_low_hp_ally", 3.0)
                else:
                    r = cfg.get("heal_ally", 1.5)
                reward += r
                rd["heal_ally"] = rd.get("heal_ally", 0.0) + r
            elif event == "kill":
                r = cfg["kill"]
                reward += r
                rd["kill"] = rd.get("kill", 0.0) + r
            elif event == "death":
                r = cfg["death"]
                reward += r
                rd["death"] = r
            elif event == "zone_damage":
                r = cfg["zone_damage"]
                reward += r
                rd["zone_damage"] = rd.get("zone_damage", 0.0) + r
            elif event == "survive":
                r = cfg["survival_per_step"]
                reward += r
                rd["survival"] = r
            elif event == "idle":
                r = cfg["idle_penalty"]
                reward += r
                rd["idle"] = r
            elif event == "wall_bump":
                r = cfg.get("wall_bump", 0.0)
                reward += r
                rd["wall_bump"] = r
            elif event == "attack_miss":
                r = cfg.get("attack_miss", 0.0)
                reward += r
                rd["attack_miss"] = rd.get("attack_miss", 0.0) + r
            elif event == "ranged_miss":
                r = cfg.get("ranged_miss", -0.2)
                reward += r
                rd["ranged_miss"] = rd.get("ranged_miss", 0.0) + r
            elif event == "invalid_action":
                r = cfg.get("invalid_action", -0.1)
                reward += r
                rd["invalid_action"] = rd.get("invalid_action", 0.0) + r
            elif event == "team_eliminated":
                r = cfg.get("team_eliminated", -8.0)
                reward += r
                rd["team_eliminated"] = r
            elif event == "teammate_death":
                r = cfg.get("teammate_death", -3.0)
                reward += r
                rd["teammate_death"] = rd.get("teammate_death", 0.0) + r
            elif event == "attack_cooldown":
                r = cfg.get("attack_cooldown_penalty", cfg.get("attack_miss", -0.3))
                reward += r
                rd["attack_cooldown"] = rd.get("attack_cooldown", 0.0) + r
            elif event == "heal_cooldown":
                r = cfg.get("heal_cooldown_penalty", cfg.get("attack_miss", -0.3))
                reward += r
                rd["heal_cooldown"] = rd.get("heal_cooldown", 0.0) + r

        # 연속 공격 보너스
        if dealt_damage_this_step:
            self._last_combat_step = self.current_step
            self._recent_combat_steps.append(self.current_step)
            self._consecutive_hits += 1
            combo_bonus = cfg.get("combo_bonus", 0.0)
            if self._consecutive_hits >= 2 and combo_bonus > 0:
                reward += combo_bonus
                rd["combo"] = combo_bonus
        else:
            self._consecutive_hits = 0

        # 장기 무전투 패널티
        no_combat_threshold = int(cfg.get("no_combat_threshold", 30))
        no_combat_penalty = cfg.get("no_combat_penalty", 0.0)
        steps_since_combat = self.current_step - self._last_combat_step
        if steps_since_combat > no_combat_threshold and no_combat_penalty != 0.0:
            reward += no_combat_penalty
            rd["no_combat"] = no_combat_penalty

        # 적 접근 보상
        approach_reward = cfg.get("approach_enemy", 0.0)
        curr_dist = self._get_nearest_enemy_dist(agent_idx)
        prev_dist = getattr(self, "_prev_enemy_dist", curr_dist)
        if approach_reward != 0.0 and agent.alive:
            if prev_dist != float("inf") and curr_dist != float("inf"):
                r = approach_reward * (prev_dist - curr_dist)
                reward += r
                rd["approach"] = r

        # 체력 낮은 팀원 접근 보상 (힐러 전용)
        approach_teammate_reward = cfg.get("approach_teammate", 0.0)
        if approach_teammate_reward != 0.0 and agent.alive:
            curr_tm_dist = self._get_nearest_low_hp_teammate_dist(agent_idx)
            prev_tm_dist = getattr(self, "_prev_teammate_dist", curr_tm_dist)
            if prev_tm_dist != float("inf") and curr_tm_dist != float("inf"):
                r = approach_teammate_reward * (prev_tm_dist - curr_tm_dist)
                if r != 0.0:
                    reward += r
                    rd["approach_teammate"] = rd.get("approach_teammate", 0.0) + r

        # 저체력 도주 보너스
        flee_hp_threshold = cfg.get("flee_hp_threshold", 0.0)
        flee_bonus = cfg.get("flee_bonus", 0.0)
        if flee_bonus != 0.0 and agent.alive:
            if (agent.hp / agent.max_hp) < flee_hp_threshold:
                if prev_dist != float("inf") and curr_dist != float("inf"):
                    dist_change = curr_dist - prev_dist
                    if dist_change > 0:
                        r = flee_bonus * dist_change
                        reward += r
                        rd["flee"] = r

        # 전투 이탈 보너스
        disengage_bonus = cfg.get("disengage_bonus", 0.0)
        disengage_hp = cfg.get("disengage_hp_threshold", 0.4)
        disengage_lookback = int(cfg.get("disengage_lookback", 5))
        if disengage_bonus != 0.0 and agent.alive:
            had_recent_combat = any(
                s >= self.current_step - disengage_lookback
                for s in self._recent_combat_steps
            )
            is_low_hp = (agent.hp / agent.max_hp) < disengage_hp
            moving_away = (prev_dist != float("inf") and
                           curr_dist != float("inf") and
                           curr_dist > prev_dist)
            if had_recent_combat and is_low_hp and moving_away:
                reward += disengage_bonus
                rd["disengage"] = disengage_bonus

        # 반복 이동 패널티
        oscillation_penalty = cfg.get("oscillation_penalty", 0.0)
        if oscillation_penalty != 0.0 and len(self._action_history) >= 3:
            h = self._action_history
            opposites = {
                ACTION_UP: ACTION_DOWN, ACTION_DOWN: ACTION_UP,
                ACTION_LEFT: ACTION_RIGHT, ACTION_RIGHT: ACTION_LEFT,
            }
            if (h[-3] == h[-1] and h[-3] in opposites
                    and opposites[h[-3]] == h[-2]):
                reward += oscillation_penalty
                rd["oscillation"] = oscillation_penalty

        # === 종료 보상 ===
        game_over = self._check_game_over()
        is_truncated = self.current_step >= self.max_steps

        if game_over or is_truncated:
            # 팀 순위 보상
            team_ranking_rewards = cfg.get("team_ranking_rewards", None)
            if team_ranking_rewards is not None:
                team_rank = self._get_team_rank(agent.team_id)
                if 0 <= team_rank - 1 < len(team_ranking_rewards):
                    r = float(team_ranking_rewards[team_rank - 1])
                    reward += r
                    rd["ranking"] = r

            # 팀 승리 보상
            if game_over:
                winning_team = self._get_winning_team()
                if winning_team == agent.team_id:
                    r = cfg.get("team_win", 25.0)
                    reward += r
                    rd["team_win"] = r

            # 킬 스트릭 보너스
            kills = agent.kills
            kill_streak_2 = cfg.get("kill_streak_2", 0.0)
            kill_streak_3 = cfg.get("kill_streak_3", 0.0)
            if kills >= 3 and kill_streak_3 > 0:
                reward += kill_streak_3
                rd["kill_streak"] = kill_streak_3
            elif kills >= 2 and kill_streak_2 > 0:
                reward += kill_streak_2
                rd["kill_streak"] = kill_streak_2

            # 낭비 패널티
            wasted_penalty = cfg.get("wasted_advantage_penalty", 0.0)
            wasted_threshold = int(cfg.get("wasted_advantage_threshold", 2))
            if not agent.alive and kills >= wasted_threshold and wasted_penalty != 0.0:
                reward += wasted_penalty
                rd["wasted_advantage"] = wasted_penalty

        self._last_reward_details = rd
        return reward

    # ─────────────────────────── 게임 종료 ───────────────────

    def _check_game_over(self) -> bool:
        """생존 팀이 1개 이하이면 게임 종료."""
        alive_teams = set()
        for a in self.agents:
            if a.alive:
                alive_teams.add(a.team_id)
        return len(alive_teams) <= 1

    def _get_winning_team(self) -> int | None:
        """승리 팀 ID를 반환한다. 없으면 None."""
        for a in self.agents:
            if a.alive:
                return a.team_id
        return None

    def _get_team_rank(self, team_id: int) -> int:
        """팀의 순위를 반환한다 (1등 = 1, 마지막 탈락 = 가장 높은 순위)."""
        # 아직 생존 중인 팀
        alive_teams = set(a.team_id for a in self.agents if a.alive)
        if team_id in alive_teams:
            return 1

        # 탈락 팀: 늦게 탈락할수록 높은 순위
        eliminated = sorted(self._team_elimination_step.items(),
                           key=lambda x: x[1], reverse=True)
        alive_count = len(alive_teams)
        for rank_offset, (tid, _) in enumerate(eliminated):
            if tid == team_id:
                return alive_count + rank_offset + 1

        return self.num_teams  # fallback

    def _get_rank(self, agent_idx: int) -> int:
        """개인 순위 (호환용). 팀 순위와 동일."""
        return self._get_team_rank(self.agents[agent_idx].team_id)

    # ─────────────────────────── 정보 ────────────────────────

    def _get_info(self) -> dict:
        """추가 정보를 반환한다."""
        return {
            "step": self.current_step,
            "agents": [
                {
                    "id": a.agent_id,
                    "hp": a.hp,
                    "alive": a.alive,
                    "kills": a.kills,
                    "position": (a.y, a.x),
                    "attack_count": a.attack_count,
                    "attack_hits": a.attack_hits,
                    "role": a.role,
                    "team_id": a.team_id,
                    "heal_count": a.heal_count,
                }
                for a in self.agents
            ],
            "reward_details": self._last_reward_details,
            "winning_team": self._get_winning_team() if self._check_game_over() else None,
            "render_events": getattr(self, "_render_events", []),
        }
