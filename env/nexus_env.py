"""Gymnasium 넥서스 파괴 모드 환경

3v3 (2팀 × 3명) 넥서스 파괴 모드.
상대 넥서스를 먼저 파괴하면 승리. 사망 시 부활 가능 (대기 시간 증가).
미니언은 규칙 기반 AI로 자동 행동.
"""

from gymnasium import spaces
import numpy as np

from env.agent import Agent, ROLE_TANK, ROLE_DEALER, ROLE_HEALER, ALL_ROLES
from env.map_generator import (
    generate_map, place_agents_near, get_empty_positions,
    TILE_EMPTY, TILE_WALL, TILE_ZONE,
)
from env.items import ItemManager, ItemType
from env.zone import ZoneManager
from env.nexus import Nexus
from env.minion import Minion
from env.minion_ai import MinionAI
from env.base_env import (
    BaseBattleEnv,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_MELEE, ACTION_RANGED_H, ACTION_RANGED_V, ACTION_HEAL, ACTION_STAY,
    NUM_ACTIONS, ROLE_ACTION_MAP, MOVE_DELTAS,
)


class NexusBattleEnv(BaseBattleEnv):
    """3v3 넥서스 파괴 모드 강화학습 환경"""

    def __init__(self, config: dict | None = None, render_mode: str | None = None,
                 opponent_policy=None, learning_role: str | None = None):
        import yaml
        if config is None:
            with open("config/nexus_mode.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        super().__init__(config=config, render_mode=render_mode,
                         opponent_policy=opponent_policy, learning_role=learning_role)

        # 관측 공간 계산
        # 5채널 시야 (벽, 아군, 적군, 아이템, 미니언)
        view_size = self.view_range
        num_channels = 5
        grid_obs = view_size * view_size * num_channels

        self_stats = 7        # hp, atk, def, is_tank, is_dealer, is_healer, range_norm
        nearest_enemy = 4     # dy, dx, adjacent, in_range
        second_enemy = 4
        teammate_1 = 5        # alive, hp, dy, dx, role_encoded
        teammate_2 = 5        # 팀원이 5명
        game_state = 3        # alive_enemy_ratio, alive_ally_ratio, visible_enemies
        nexus_info = 4        # own_hp, enemy_hp, own_dy, own_dx
        respawn_info = 2      # timer, death_count
        nearest_minion = 3    # dy, dx, distance

        obs_size = (grid_obs + self_stats + nearest_enemy + second_enemy
                    + teammate_1 + teammate_2 + game_state
                    + nexus_info + respawn_info + nearest_minion)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # 넥서스 모드 전용
        self.nexuses: list[Nexus] = []
        self.minions: list[Minion] = []
        self.minion_ai = MinionAI(config)
        self._next_minion_id = 0
        self._death_counts: dict[int, int] = {}
        self._respawn_timers: dict[int, int] = {}

        # 넥서스 설정
        nexus_cfg = config.get("nexus", {})
        self._nexus_regions = [
            nexus_cfg.get("team_0_region", [1, 3, 1, 3]),
            nexus_cfg.get("team_1_region", [16, 18, 16, 18]),
        ]

        # 미니언 설정
        minion_cfg = config.get("minion", {})
        self._minion_max_per_team = minion_cfg.get("max_per_team", 8)

        # 부활 설정
        respawn_cfg = config.get("respawn", {})
        self._respawn_base_time = respawn_cfg.get("base_time", 5)
        self._respawn_increment = respawn_cfg.get("increment_per_death", 3)
        self._spawn_radius = respawn_cfg.get("spawn_radius", 3)

    # ─────────────────────────── reset ───────────────────────

    def reset(self, seed=None, options=None):
        super(BaseBattleEnv, self).reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 맵 생성
        self.grid = generate_map(
            self.map_width, self.map_height, self.wall_count, self.rng
        )

        # 넥서스 배치 — 지정 영역 내 빈 칸에 배치, 주변 벽 제거
        self.nexuses = []
        for team_id in range(2):
            region = self._nexus_regions[team_id]
            ny, nx = self._place_nexus_in_region(region)
            self.nexuses.append(Nexus(team_id, nx, ny, self.config))

        # 에이전트 배치 — 각 팀 넥서스 근처에 스폰
        # 역할 강제 배정:
        #   - 기본 3명은 탱커/딜러/힐러 1명씩
        #   - 나머지(예: 팀당 2명)는 역할 랜덤
        self.agents = []
        agent_id = 0
        for team_id in range(2):
            nexus = self.nexuses[team_id]
            positions = place_agents_near(
                self.grid, nexus.y, nexus.x,
                self.agents_per_team, self._spawn_radius, self.rng
            )
            # 역할 배정
            if self.learning_role and team_id == 0:
                # 멀티 정책: 팀0의 첫 에이전트는 learning_role 고정,
                # 나머지 두 기본 역할은 나머지 역할에서 선택
                core_roles = [self.learning_role] + [r for r in ALL_ROLES if r != self.learning_role][:2]
            else:
                # 탱커/딜러/힐러 1명씩
                core_roles = list(ALL_ROLES)
            # 나머지 두 명은 역할 셔플링
            self.rng.shuffle(core_roles)

            team_role_list = []
            # 앞 3명: 탱커/딜러/힐러 각 1명씩 (순서는 랜덤)
            team_role_list.extend(core_roles[:3])
            # 남은 인원: 역할 랜덤 샘플링
            extra = self.agents_per_team - 3
            if extra > 0:
                for _ in range(extra):
                    team_role_list.append(self.rng.choice(ALL_ROLES))

            for idx, (y, x) in enumerate(positions):
                # 멀티 정책: team0의 agent0(인덱스 0)은 learning_role을 강제한다.
                # (기존 코드에서는 core_roles를 shuffle한 뒤 idx별로 role을 배정해서
                #  agent0이 learning_role이 아닐 때가 생겼고, 이 경우 agent0의
                #  action_space/보상/invalid_action이 어긋날 수 있다.)
                if self.learning_role and team_id == 0 and idx == 0:
                    role = self.learning_role
                else:
                    role = team_role_list[idx]
                self.agents.append(
                    Agent(agent_id, x, y, self.config, role=role, team_id=team_id)
                )
                agent_id += 1

        # 매니저 초기화
        grid_shape = (self.map_height, self.map_width)
        self.item_manager = ItemManager(self.config, grid_shape, self.rng)
        self.item_manager.reset(self.grid)
        if self.item_manager.enabled:
            self.item_manager._spawn_items(self.grid)

        # 자기장은 비활성이지만 ZoneManager 인스턴스는 필요 (base_env 호환)
        self.zone_manager = ZoneManager(self.config, grid_shape)
        self.zone_manager.reset()

        self.current_step = 0

        # 미니언 초기화
        self.minions = []
        self._next_minion_id = 0

        # 부활 추적 초기화
        self._death_counts = {i: 0 for i in range(self.num_agents)}
        self._respawn_timers = {}

        # 전투 추적
        self._last_combat_step = 0
        self._consecutive_hits = 0
        self._prev_enemy_dist = float("inf")
        self._prev_teammate_dist = float("inf")
        self._action_history = []
        self._recent_combat_steps = []
        self._step_events = {}
        self._last_reward_details = {}

        observation = self._get_observation(0)
        info = self._get_info()
        return observation, info

    def _place_nexus_in_region(self, region: list[int]) -> tuple[int, int]:
        """지정 영역 내 빈 칸에 넥서스를 배치한다. 주변 벽을 제거한다."""
        r_min, r_max, c_min, c_max = region
        candidates = []
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if 0 <= r < self.map_height and 0 <= c < self.map_width:
                    candidates.append((r, c))

        # 영역 내 벽 제거
        for r, c in candidates:
            if self.grid[r, c] == TILE_WALL:
                self.grid[r, c] = TILE_EMPTY

        # 중앙 배치
        center_y = (r_min + r_max) // 2
        center_x = (c_min + c_max) // 2

        # 주변도 벽 제거 (스폰 공간 확보)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = center_y + dy, center_x + dx
                if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                    if self.grid[ny, nx] == TILE_WALL:
                        self.grid[ny, nx] = TILE_EMPTY

        return center_y, center_x

    # ─────────────────────────── step ────────────────────────

    def step(self, action):
        """학습 에이전트(0번)의 행동을 받아 한 스텝을 진행한다."""
        self.current_step += 1
        self._step_events = {i: [] for i in range(self.num_agents)}
        self._render_events: list[dict] = []

        # 역할별 행동 매핑
        if self._role_action_map is not None:
            global_action = self._role_action_map[action]
        else:
            global_action = action

        # 이전 거리 저장
        self._prev_enemy_dist = self._get_nearest_enemy_dist(0)
        self._prev_teammate_dist = self._get_nearest_low_hp_teammate_dist(0)

        # 상대 행동
        actions = [global_action]
        opp_actions = self._get_all_opponent_actions()
        actions.extend(opp_actions)

        # 사망 중인 에이전트 행동 무시
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                actions[i] = ACTION_STAY

        # 1. 부활 처리
        self._process_respawns()

        # 2. 이동
        self._process_movements(actions)

        # 3. 에이전트 간 공격
        self._process_attacks(actions)

        # 4. 넥서스 공격
        self._process_nexus_attacks(actions)

        # 5. 힐
        self._process_heals(actions)

        # 6. 아이템
        self._process_items()

        # 7. 미니언 스폰
        self._maybe_spawn_minions()

        # 8. 미니언 AI 행동
        self._process_minion_actions()

        # 9. 에이전트 사망 → 부활 대기열
        self._process_agent_deaths()

        # 10. 대기 행동
        if global_action == ACTION_STAY and self.agents[0].alive:
            self._step_events[0].append("idle")

        # 11. 역할 외 행동
        self._check_invalid_actions(actions)

        # 12. 생존 이벤트
        for i, agent in enumerate(self.agents):
            if agent.alive:
                self._step_events[i].append("survive")

        # 13. 아군 넥서스 피격 이벤트 (보상 계산용)
        self._check_own_nexus_damage()

        # 매니저 스텝
        self.item_manager.step(self.grid)

        # 쿨다운 감소
        for agent in self.agents:
            if agent.alive:
                if agent.attack_cooldown > 0:
                    agent.attack_cooldown -= 1
                if agent.heal_cooldown > 0:
                    agent.heal_cooldown -= 1

        # 행동 히스토리
        self._action_history.append(action)
        if len(self._action_history) > 3:
            self._action_history.pop(0)

        # 보상
        reward = self._calculate_reward(0)

        # 종료
        terminated = self._check_game_over()
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation(0)
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    # ─────────────────── 부활 시스템 ──────────────────────────

    def _process_respawns(self):
        """부활 타이머를 감소시키고, 0이 되면 부활시킨다."""
        completed = []
        for agent_id, timer in self._respawn_timers.items():
            if timer <= 1:
                completed.append(agent_id)
            else:
                self._respawn_timers[agent_id] -= 1

        for agent_id in completed:
            del self._respawn_timers[agent_id]
            agent = self.agents[agent_id]
            nexus = self.nexuses[agent.team_id]
            if not nexus.alive:
                continue  # 넥서스가 파괴되면 부활 불가

            # 부활 위치 찾기
            spawn_pos = self._find_respawn_position(nexus.y, nexus.x)
            if spawn_pos is None:
                continue

            agent.y, agent.x = spawn_pos
            agent.alive = True
            self._reset_agent_stats(agent)
            self._step_events[agent_id].append("respawn")

    def _find_respawn_position(self, anchor_y: int, anchor_x: int) -> tuple[int, int] | None:
        """넥서스 근처 빈 칸을 찾는다."""
        candidates = []
        for dy in range(-self._spawn_radius, self._spawn_radius + 1):
            for dx in range(-self._spawn_radius, self._spawn_radius + 1):
                if abs(dy) + abs(dx) > self._spawn_radius:
                    continue
                ny, nx = anchor_y + dy, anchor_x + dx
                if not (0 <= ny < self.map_height and 0 <= nx < self.map_width):
                    continue
                if self.grid[ny, nx] != TILE_EMPTY:
                    continue
                if ny == anchor_y and nx == anchor_x:
                    continue
                # 다른 에이전트/미니언이 없는지 확인
                occupied = False
                for a in self.agents:
                    if a.alive and a.y == ny and a.x == nx:
                        occupied = True
                        break
                if not occupied:
                    for m in self.minions:
                        if m.alive and m.y == ny and m.x == nx:
                            occupied = True
                            break
                if not occupied:
                    candidates.append((ny, nx))

        if not candidates:
            return None
        idx = int(self.rng.integers(0, len(candidates)))
        return candidates[idx]

    def _reset_agent_stats(self, agent: Agent):
        """에이전트 스탯을 역할 기본값으로 초기화한다 (아이템 효과 제거)."""
        role_cfg = self.roles_cfg.get(agent.role, {})
        agent_cfg = self.config.get("agent", {})
        agent.hp = role_cfg.get("hp", agent_cfg.get("initial_hp", 100))
        agent.max_hp = agent.hp
        agent.attack = role_cfg.get("attack", agent_cfg.get("initial_attack", 10))
        agent.defense = role_cfg.get("defense", agent_cfg.get("initial_defense", 0))
        agent.attack_cooldown = 0
        agent.heal_cooldown = 0

    def _process_agent_deaths(self):
        """사망한 에이전트를 부활 대기열에 넣는다."""
        for agent in self.agents:
            if not agent.alive and agent.agent_id not in self._respawn_timers:
                if agent.death_step == self.current_step:
                    self._death_counts[agent.agent_id] += 1
                    death_count = self._death_counts[agent.agent_id]
                    timer = self._respawn_base_time + (death_count - 1) * self._respawn_increment
                    self._respawn_timers[agent.agent_id] = timer

    # ─────────────────── 넥서스 공격 ──────────────────────────

    def _process_nexus_attacks(self, actions: list[int]):
        """에이전트가 적 넥서스 인접 칸에서 근접 공격 시 넥서스에 데미지."""
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            if actions[i] != ACTION_MELEE:
                continue

            for nexus in self.nexuses:
                if nexus.team_id == agent.team_id or not nexus.alive:
                    continue
                dist = abs(agent.y - nexus.y) + abs(agent.x - nexus.x)
                if dist == 1:
                    # 에이전트 간 전투에서 이미 타겟을 찾았다면 넥서스 공격 스킵
                    # (damage_dealt 이벤트가 이미 있는지 확인)
                    already_hit = any(
                        isinstance(e, tuple) and e[0] == "damage_dealt"
                        for e in self._step_events[i]
                    )
                    if already_hit:
                        # 마무리 구간에서는 넥서스를 우선 타격하도록 예외 허용
                        thr = float(
                            self.config.get("reward", {}).get(
                                "finisher_enemy_nexus_hp_threshold", 0.25
                            )
                        )
                        hp_ratio = nexus.hp / max(nexus.max_hp, 1)
                        if hp_ratio > thr:
                            continue

                    actual_dmg = nexus.take_damage(agent.attack)
                    self._step_events[i].append(("nexus_damage", actual_dmg))
                    self._render_events.append({
                        "type": "nexus_hit",
                        "attacker_x": agent.x, "attacker_y": agent.y,
                        "target_x": nexus.x, "target_y": nexus.y,
                    })

    def _check_own_nexus_damage(self):
        """아군 넥서스가 이번 스텝에 데미지를 받았는지 체크한다."""
        agent0 = self.agents[0]
        own_nexus = self.nexuses[agent0.team_id]

        # 적 에이전트나 미니언이 아군 넥서스를 공격했는지 확인
        # nexus_damage 이벤트를 가진 적 에이전트가 있으면 own_nexus_damaged 추가
        for i, agent in enumerate(self.agents):
            if agent.team_id == agent0.team_id:
                continue
            for event in self._step_events.get(i, []):
                if isinstance(event, tuple) and event[0] == "nexus_damage":
                    self._step_events[0].append("own_nexus_damaged")
                    return

        # 미니언에 의한 넥서스 데미지도 체크
        if hasattr(self, "_own_nexus_damaged_this_step") and self._own_nexus_damaged_this_step:
            self._step_events[0].append("own_nexus_damaged")

    # ─────────────────── 미니언 시스템 ────────────────────────

    def _maybe_spawn_minions(self):
        """주기적으로 미니언 웨이브를 스폰한다."""
        minion_cfg = self.config.get("minion", {})
        start = minion_cfg.get("spawn_start_step", 15)
        interval = minion_cfg.get("spawn_interval", 25)
        per_wave = minion_cfg.get("per_wave", 2)

        if self.current_step < start:
            return
        if (self.current_step - start) % interval != 0:
            return

        for team_id in range(2):
            # 팀 미니언 수 제한
            alive_count = sum(1 for m in self.minions
                             if m.alive and m.team_id == team_id)
            can_spawn = min(per_wave, self._minion_max_per_team - alive_count)
            if can_spawn <= 0:
                continue

            nexus = self.nexuses[team_id]
            for _ in range(can_spawn):
                pos = self._find_respawn_position(nexus.y, nexus.x)
                if pos is None:
                    break
                minion = Minion(self._next_minion_id, team_id, pos[1], pos[0], self.config)
                self.minions.append(minion)
                self._next_minion_id += 1

    def _process_minion_actions(self):
        """모든 살아있는 미니언의 규칙 기반 AI 행동을 실행한다."""
        self._own_nexus_damaged_this_step = False

        # 미니언 이동 속도 더 감소: 3스텝에 한 번만 이동/공격 처리
        if self.current_step % 3 != 0:
            return

        for minion in self.minions:
            if not minion.alive:
                continue

            enemy_nexus = self.nexuses[1 - minion.team_id]
            action = self.minion_ai.get_action(
                minion, self.grid, self.agents, self.minions, enemy_nexus
            )

            if action[0] == "attack_agent":
                target = action[1]
                target.take_damage(minion.attack)
                if not target.alive:
                    target.death_step = self.current_step
                    self._step_events[target.agent_id].append("death")
                    self._render_events.append({
                        "type": "death", "x": target.x, "y": target.y,
                    })

            elif action[0] == "attack_nexus":
                actual_dmg = enemy_nexus.take_damage(minion.attack)
                # 학습 에이전트 팀의 넥서스가 피해를 받았는지 체크
                if enemy_nexus.team_id == self.agents[0].team_id:
                    self._own_nexus_damaged_this_step = True
                self._render_events.append({
                    "type": "nexus_hit",
                    "attacker_x": minion.x, "attacker_y": minion.y,
                    "target_x": enemy_nexus.x, "target_y": enemy_nexus.y,
                })

            elif action[0] == "move":
                _, dy, dx = action
                new_y = minion.y + dy
                new_x = minion.x + dx
                if (0 <= new_y < self.map_height and
                        0 <= new_x < self.map_width and
                        self.grid[new_y, new_x] != TILE_WALL):
                    minion.y = new_y
                    minion.x = new_x

    # ─────────────────── 관측 ────────────────────────────────

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """에이전트의 관측을 반환한다 (넥서스 모드: 5채널 + 넥서스/미니언/부활 정보)."""
        agent = self.agents[agent_idx]
        view = self.view_range
        half = view // 2
        max_dim = max(self.map_height, self.map_width)

        # 사망 중인 에이전트: 넥서스 중심으로 관측
        if not agent.alive:
            obs_center_y = self.nexuses[agent.team_id].y
            obs_center_x = self.nexuses[agent.team_id].x
        else:
            obs_center_y = agent.y
            obs_center_x = agent.x

        # 5채널 시야 맵
        local_map = np.zeros((view, view, 5), dtype=np.float32)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                gy = obs_center_y + dy
                gx = obs_center_x + dx
                ly = dy + half
                lx = dx + half

                if not (0 <= gy < self.map_height and 0 <= gx < self.map_width):
                    local_map[ly, lx, 0] = 1.0
                    continue

                tile = self.grid[gy, gx]
                if tile == TILE_WALL:
                    local_map[ly, lx, 0] = 1.0

                # 채널 1: 아군 에이전트 / 채널 2: 적 에이전트
                for other in self.agents:
                    if other.agent_id == agent_idx or not other.alive:
                        continue
                    if other.y == gy and other.x == gx:
                        if agent.is_teammate(other):
                            local_map[ly, lx, 1] = 1.0
                        else:
                            local_map[ly, lx, 2] = 1.0

                # 채널 3: 아이템
                if self.item_manager and self.item_manager.enabled:
                    item_val = self.item_manager.item_grid[gy, gx]
                    if item_val > 0:
                        local_map[ly, lx, 3] = item_val / 3.0

                # 채널 4: 미니언 (아군 0.5, 적 1.0)
                for m in self.minions:
                    if not m.alive:
                        continue
                    if m.y == gy and m.x == gx:
                        if m.team_id == agent.team_id:
                            local_map[ly, lx, 4] = 0.5
                        else:
                            local_map[ly, lx, 4] = 1.0

                # 넥서스도 시야에 표시 (채널 1/2에 넥서스를 표시)
                for nexus in self.nexuses:
                    if not nexus.alive:
                        continue
                    if nexus.y == gy and nexus.x == gx:
                        if nexus.team_id == agent.team_id:
                            local_map[ly, lx, 1] = max(local_map[ly, lx, 1], 0.7)
                        else:
                            local_map[ly, lx, 2] = max(local_map[ly, lx, 2], 0.7)

        # 자기 스탯 (7차원)
        if agent.alive:
            stats = agent.get_stats()
        else:
            stats = np.zeros(7, dtype=np.float32)

        # 가장 가까운 적 에이전트 방향
        enemy_dir_1 = np.zeros(4, dtype=np.float32)
        enemy_dir_2 = np.zeros(4, dtype=np.float32)
        if agent.alive:
            enemies = []
            for other in self.agents:
                if other.agent_id != agent_idx and other.alive:
                    if not agent.is_teammate(other):
                        dist = abs(agent.y - other.y) + abs(agent.x - other.x)
                        enemies.append((dist, other))
            enemies.sort(key=lambda e: e[0])

            if len(enemies) >= 1:
                _, ne = enemies[0]
                enemy_dir_1[0] = (ne.y - agent.y) / max_dim
                enemy_dir_1[1] = (ne.x - agent.x) / max_dim
                d = abs(agent.y - ne.y) + abs(agent.x - ne.x)
                enemy_dir_1[2] = 1.0 if d == 1 else 0.0
                enemy_dir_1[3] = 1.0 if d <= agent.attack_range else 0.0

            if len(enemies) >= 2:
                _, se = enemies[1]
                enemy_dir_2[0] = (se.y - agent.y) / max_dim
                enemy_dir_2[1] = (se.x - agent.x) / max_dim
                d = abs(agent.y - se.y) + abs(agent.x - se.x)
                enemy_dir_2[2] = 1.0 if d == 1 else 0.0
                enemy_dir_2[3] = 1.0 if d <= agent.attack_range else 0.0

        # 팀원 정보 (2명)
        teammate_1 = np.zeros(5, dtype=np.float32)
        teammate_2 = np.zeros(5, dtype=np.float32)
        tm_idx = 0
        for other in self.agents:
            if other.agent_id != agent_idx and agent.is_teammate(other):
                tm = teammate_1 if tm_idx == 0 else teammate_2
                tm[0] = 1.0 if other.alive else 0.0
                tm[1] = other.hp / other.max_hp if other.alive else 0.0
                if agent.alive:
                    tm[2] = (other.y - agent.y) / max_dim
                    tm[3] = (other.x - agent.x) / max_dim
                if other.role == ROLE_TANK:
                    tm[4] = 0.33
                elif other.role == ROLE_DEALER:
                    tm[4] = 0.67
                elif other.role == ROLE_HEALER:
                    tm[4] = 1.0
                tm_idx += 1

        # 게임 상태
        game_state = np.zeros(3, dtype=np.float32)
        total_enemies = sum(1 for a in self.agents if not agent.is_teammate(a)
                           and a.agent_id != agent_idx)
        alive_enemies = sum(1 for a in self.agents if not agent.is_teammate(a)
                           and a.agent_id != agent_idx and a.alive)
        game_state[0] = alive_enemies / max(total_enemies, 1)
        alive_allies = sum(1 for a in self.agents
                          if a.agent_id != agent_idx and agent.is_teammate(a) and a.alive)
        game_state[1] = alive_allies / max(self.agents_per_team - 1, 1)
        visible_enemies = 0
        for other in self.agents:
            if other.agent_id != agent_idx and other.alive and not agent.is_teammate(other):
                if (abs(other.y - obs_center_y) <= half and
                        abs(other.x - obs_center_x) <= half):
                    visible_enemies += 1
        game_state[2] = visible_enemies / max(total_enemies, 1)

        # 넥서스 정보: [아군HP, 적HP, 아군넥서스_dy, 아군넥서스_dx]
        nexus_info = np.zeros(4, dtype=np.float32)
        own_nexus = self.nexuses[agent.team_id]
        enemy_nexus = self.nexuses[1 - agent.team_id]
        nexus_info[0] = own_nexus.hp / own_nexus.max_hp
        nexus_info[1] = enemy_nexus.hp / enemy_nexus.max_hp
        if agent.alive:
            nexus_info[2] = (own_nexus.y - agent.y) / max_dim
            nexus_info[3] = (own_nexus.x - agent.x) / max_dim

        # 부활 정보: [timer_normalized, death_count_normalized]
        respawn_info = np.zeros(2, dtype=np.float32)
        if not agent.alive:
            max_timer = self._respawn_base_time + 5 * self._respawn_increment
            timer = self._respawn_timers.get(agent_idx, 0)
            respawn_info[0] = min(timer / max(max_timer, 1), 1.0)
            respawn_info[1] = min(self._death_counts.get(agent_idx, 0) / 5.0, 1.0)

        # 가장 가까운 적 미니언: [dy, dx, distance_normalized]
        nearest_minion = np.zeros(3, dtype=np.float32)
        if agent.alive:
            closest_m_dist = float("inf")
            closest_m = None
            for m in self.minions:
                if not m.alive or m.team_id == agent.team_id:
                    continue
                d = abs(m.y - agent.y) + abs(m.x - agent.x)
                if d < closest_m_dist:
                    closest_m_dist = d
                    closest_m = m
            if closest_m is not None:
                nearest_minion[0] = (closest_m.y - agent.y) / max_dim
                nearest_minion[1] = (closest_m.x - agent.x) / max_dim
                nearest_minion[2] = min(closest_m_dist / max_dim, 1.0)

        obs = np.concatenate([
            local_map.flatten(), stats,
            enemy_dir_1, enemy_dir_2,
            teammate_1, teammate_2,
            game_state, nexus_info, respawn_info, nearest_minion,
        ])
        return obs

    # ─────────────────── 정책 정규화 (진영 대칭) ─────────────────────────

    def transform_obs_for_policy(self, agent_idx: int, obs: np.ndarray) -> np.ndarray:
        """Team1(레드)이 Team0과 같은 좌표계로 보이도록 관측을 180도 회전/부호 반전한다.

        - local_map(view×view×5): 180도 회전 (상하/좌우 뒤집기)
        - dy/dx 형태의 상대 좌표: 부호 반전
        """
        if not getattr(self, "_policy_team_normalize", False):
            return obs
        if agent_idx < 0 or agent_idx >= len(self.agents):
            return obs
        agent = self.agents[agent_idx]
        if getattr(agent, "team_id", 0) != 1:
            return obs

        view = self.view_range
        num_channels = 5
        grid_obs = view * view * num_channels

        # 안전하게 복사
        o = np.array(obs, copy=True)

        # 1) 로컬 맵 180도 회전
        lm = o[:grid_obs].reshape((view, view, num_channels))
        lm = lm[::-1, ::-1, :]
        o[:grid_obs] = lm.reshape(-1)

        # 오프셋 계산 (NexusBattleEnv.__init__ 정의와 동일)
        off = grid_obs
        self_stats = 7
        nearest_enemy = 4
        second_enemy = 4
        teammate = 5
        game_state = 3
        nexus_info = 4
        respawn_info = 2
        nearest_minion = 3

        off_stats = off
        off += self_stats
        off_enemy1 = off
        off += nearest_enemy
        off_enemy2 = off
        off += second_enemy
        off_tm1 = off
        off += teammate
        off_tm2 = off
        off += teammate
        off_game = off
        off += game_state
        off_nexus = off
        off += nexus_info
        off_respawn = off
        off += respawn_info
        off_minion = off
        off += nearest_minion

        # 2) dy/dx 성분 부호 반전
        # enemy_dir_1: [dy, dx, adjacent, in_range]
        o[off_enemy1 + 0] *= -1.0
        o[off_enemy1 + 1] *= -1.0
        # enemy_dir_2
        o[off_enemy2 + 0] *= -1.0
        o[off_enemy2 + 1] *= -1.0
        # teammate_1: [alive, hp, dy, dx, role]
        o[off_tm1 + 2] *= -1.0
        o[off_tm1 + 3] *= -1.0
        # teammate_2
        o[off_tm2 + 2] *= -1.0
        o[off_tm2 + 3] *= -1.0
        # nexus_info: [own_hp, enemy_hp, own_dy, own_dx]
        o[off_nexus + 2] *= -1.0
        o[off_nexus + 3] *= -1.0
        # nearest_minion: [dy, dx, dist]
        o[off_minion + 0] *= -1.0
        o[off_minion + 1] *= -1.0

        return o

    def transform_action_from_policy(self, agent_idx: int, action: int) -> int:
        """Team1(레드)이 낸 action을 환경 좌표계로 되돌린다 (180도 회전 역변환).

        이동만 반전:
          UP <-> DOWN, LEFT <-> RIGHT
        나머지(공격/힐/대기/원거리 방향)는 동일하게 사용.
        """
        if not getattr(self, "_policy_team_normalize", False):
            return action
        if agent_idx < 0 or agent_idx >= len(self.agents):
            return action
        agent = self.agents[agent_idx]
        if getattr(agent, "team_id", 0) != 1:
            return action

        if action == ACTION_UP:
            return ACTION_DOWN
        if action == ACTION_DOWN:
            return ACTION_UP
        if action == ACTION_LEFT:
            return ACTION_RIGHT
        if action == ACTION_RIGHT:
            return ACTION_LEFT
        return action

    # ─────────────────── 보상 ────────────────────────────────

    def _enemy_agents_visible_in_view(self, agent_idx: int) -> int:
        """시야(뷰 사각형) 안에 살아 있는 적 에이전트 수. `_get_observation`의 visible_enemies와 동일 기준."""
        agent = self.agents[agent_idx]
        if not agent.alive:
            return 0
        view = self.view_range
        half = view // 2
        cy, cx = agent.y, agent.x
        n = 0
        for other in self.agents:
            if other.agent_id == agent_idx or not other.alive:
                continue
            if agent.is_teammate(other):
                continue
            if abs(other.y - cy) <= half and abs(other.x - cx) <= half:
                n += 1
        return n

    def _calculate_reward(self, agent_idx: int) -> float:
        """넥서스 모드 보상을 계산한다."""
        events = self._step_events.get(agent_idx, [])
        reward = 0.0
        agent = self.agents[agent_idx]
        cfg = self._role_reward_cfgs.get(agent.role, self.reward_cfg)
        dealt_damage_this_step = False
        rd = {}

        for event in events:
            if isinstance(event, tuple) and event[0] == "damage_dealt":
                _, value = event
                r = cfg.get("damage_dealt", 0.8)
                reward += r
                rd["damage_dealt"] = rd.get("damage_dealt", 0.0) + r
                dealt_damage_this_step = True

            elif isinstance(event, tuple) and event[0] == "nexus_damage":
                _, dmg = event
                r = cfg.get("nexus_damage", 3.0)

                # 넥서스 마무리(매우 낮은 HP)는 교전으로 취급하여
                # no_combat 패널티 누적 때문에 넥서스를 포기하는 부작용을 줄인다.
                fin_thr = float(cfg.get("finisher_enemy_nexus_hp_threshold", 0.25))
                fin_mult = float(cfg.get("finisher_nexus_damage_multiplier", 1.0))
                enemy_nexus = self.nexuses[1 - agent.team_id]
                hp_ratio = enemy_nexus.hp / max(enemy_nexus.max_hp, 1)
                if fin_mult != 1.0 and hp_ratio <= fin_thr:
                    r *= fin_mult
                reward += r
                rd["nexus_damage"] = rd.get("nexus_damage", 0.0) + r
                if hp_ratio <= fin_thr:
                    dealt_damage_this_step = True

            elif isinstance(event, tuple) and event[0] == "minion_kill":
                _, mx, my = event
                own_n = self.nexuses[agent.team_id]
                # "아군 넥서스 근처에서" 처치 보너스 적용
                minion_radius = int(cfg.get("minion_kill_nexus_radius", 3))
                if own_n.alive:
                    dist_to_own_nexus = abs(my - own_n.y) + abs(mx - own_n.x)
                    if dist_to_own_nexus <= minion_radius:
                        r = cfg.get("minion_kill_near_nexus", cfg.get("minion_kill", 0.0))
                    else:
                        r = cfg.get("minion_kill", 0.0)
                else:
                    r = cfg.get("minion_kill", 0.0)

                reward += r
                rd["minion_kill"] = rd.get("minion_kill", 0.0) + r
                # 미니언 처치도 전투로 간주해서 no_combat 타이머 갱신
                dealt_damage_this_step = True

            elif isinstance(event, tuple) and event[0] == "item_pickup":
                _, item_type, hp_before = event
                threshold = cfg.get("low_hp_potion_threshold", 0.5)
                if item_type == ItemType.POTION and hp_before < threshold:
                    r = cfg.get("low_hp_potion_bonus", cfg.get("item_pickup", 1.5))
                else:
                    r = cfg.get("item_pickup", 1.5)
                reward += r
                rd["item_pickup"] = rd.get("item_pickup", 0.0) + r

            elif isinstance(event, tuple) and event[0] == "heal_ally":
                _, heal_amount, hp_ratio = event
                heal_threshold = cfg.get("heal_hp_threshold", 0.5)
                if hp_ratio < heal_threshold:
                    r = cfg.get("heal_low_hp_ally", 4.0)
                else:
                    r = cfg.get("heal_ally", 1.5)
                reward += r
                rd["heal_ally"] = rd.get("heal_ally", 0.0) + r

            elif event == "kill":
                r = cfg.get("kill", 5.0)
                reward += r
                rd["kill"] = rd.get("kill", 0.0) + r

            elif event == "death":
                r = cfg.get("death", -5.0)
                reward += r
                rd["death"] = r

            elif event == "own_nexus_damaged":
                r = cfg.get("own_nexus_damaged", -2.0)
                reward += r
                rd["own_nexus_damaged"] = rd.get("own_nexus_damaged", 0.0) + r

            elif event == "survive":
                r = cfg.get("survival_per_step", 0.02)
                reward += r
                rd["survival"] = r

            elif event == "idle":
                r = cfg.get("idle_penalty", -0.03)
                reward += r
                rd["idle"] = r

            elif event == "wall_bump":
                r = cfg.get("wall_bump", -0.1)
                reward += r
                rd["wall_bump"] = r

            elif event == "attack_miss":
                r = cfg.get("attack_miss", -0.2)
                reward += r
                rd["attack_miss"] = rd.get("attack_miss", 0.0) + r

            elif event == "ranged_miss":
                r = cfg.get("ranged_miss", -0.1)
                reward += r
                rd["ranged_miss"] = rd.get("ranged_miss", 0.0) + r

            elif event == "invalid_action":
                r = cfg.get("invalid_action", -0.1)
                reward += r
                rd["invalid_action"] = rd.get("invalid_action", 0.0) + r

            elif event == "teammate_death":
                r = cfg.get("teammate_death", -2.0)
                reward += r
                rd["teammate_death"] = rd.get("teammate_death", 0.0) + r

            elif event == "attack_cooldown":
                r = cfg.get("attack_cooldown_penalty", -0.2)
                reward += r
                rd["attack_cooldown"] = rd.get("attack_cooldown", 0.0) + r

            elif event == "heal_cooldown":
                r = cfg.get("heal_cooldown_penalty", -0.2)
                reward += r
                rd["heal_cooldown"] = rd.get("heal_cooldown", 0.0) + r

        # 저체력인데 이번 스텝에 공격(딜)을 넣은 경우 패널티 → 후퇴·포션 유도
        if agent.alive and dealt_damage_this_step:
            thr = float(cfg.get("low_hp_combat_threshold", 0.0))
            pen = float(cfg.get("low_hp_combat_penalty", 0.0))
            if thr > 0.0 and pen != 0.0 and (agent.hp / max(agent.max_hp, 1)) < thr:
                reward += pen
                rd["low_hp_combat"] = rd.get("low_hp_combat", 0.0) + pen

        # 연속 공격 보너스
        if dealt_damage_this_step:
            self._last_combat_step = self.current_step
            self._recent_combat_steps.append(self.current_step)
            self._consecutive_hits += 1
            combo = cfg.get("combo_bonus", 0.3)
            if self._consecutive_hits >= 2 and combo > 0:
                reward += combo
                rd["combo"] = combo
        else:
            self._consecutive_hits = 0

        # 장기 무전투 패널티
        no_combat_threshold = int(cfg.get("no_combat_threshold", 40))
        no_combat_penalty = cfg.get("no_combat_penalty", -0.1)
        steps_since = self.current_step - self._last_combat_step
        if steps_since > no_combat_threshold and no_combat_penalty != 0.0 and agent.alive:
            reward += no_combat_penalty
            rd["no_combat"] = no_combat_penalty

        # 적 접근 보상
        if agent.alive:
            visible_enemy_count = self._enemy_agents_visible_in_view(agent_idx)
            approach_reward = cfg.get("approach_enemy", 0.05)
            curr_dist = self._get_nearest_enemy_dist(agent_idx)
            prev_dist = getattr(self, "_prev_enemy_dist", curr_dist)
            dist_change = curr_dist - prev_dist
            if approach_reward != 0.0:
                if prev_dist != float("inf") and curr_dist != float("inf"):
                    r = approach_reward * (prev_dist - curr_dist)
                    if visible_enemy_count > 0:
                        vm = float(cfg.get("visible_enemy_approach_mult", 1.0))
                        r *= vm
                    # 치고 빠지기 모드(연속으로 때리는 중)에서는
                    # 후퇴로 인해 `approach_enemy`가 크게 깎이지 않도록 음수 방향을 중화한다.
                    hr_thr = int(cfg.get("hit_and_run_consecutive_hits_threshold", 2))
                    if (
                        dealt_damage_this_step
                        and self._consecutive_hits >= hr_thr
                        and dist_change > 0
                        and r < 0.0
                    ):
                        r = 0.0
                    reward += r
                    rd["approach"] = r

            # 적팀 미니언 접근 보상 
            approach_minion_reward = cfg.get("approach_minion", 0.0)
            if approach_minion_reward != 0.0:
                curr_minion_dist = float("inf")
                for m in self.minions:
                    if not m.alive or m.team_id == agent.team_id:
                        continue
                    d = abs(agent.y - m.y) + abs(agent.x - m.x)
                    if d < curr_minion_dist:
                        curr_minion_dist = d

                prev_minion_dist = getattr(self, "_prev_enemy_minion_dist", curr_minion_dist)
                if prev_minion_dist != float("inf") and curr_minion_dist != float("inf"):
                    r = approach_minion_reward * (prev_minion_dist - curr_minion_dist)
                    reward += r
                    rd["approach_minion"] = r

                # 다음 스텝을 위한 거리 갱신
                self._prev_enemy_minion_dist = curr_minion_dist

            # 저체력 도주 보너스: 체력이 낮을 때 적과의 거리를 벌리면 보상
            flee_hp_threshold = cfg.get("flee_hp_threshold", 0.0)
            flee_bonus = cfg.get("flee_bonus", 0.0)
            if flee_bonus != 0.0 and (agent.hp / agent.max_hp) < flee_hp_threshold:
                if prev_dist != float("inf") and curr_dist != float("inf"):
                    dist_change = curr_dist - prev_dist
                    if dist_change > 0:
                        r = flee_bonus * dist_change
                        if visible_enemy_count > 0:
                            r *= float(cfg.get("flee_visible_mult", 1.0))
                        reward += r
                        rd["flee"] = rd.get("flee", 0.0) + r

            # 치고 빠지기(무한 교전 억제): 연속으로 맞추는데도 거리가 늘지 않으면 패널티
            # - 거리가 늘면(후퇴면) 보상
            # - 시야에 적이 있을 때만 적용하여 넥서스 질주/전투 외 상황을 덜 방해
            hr_thr = int(cfg.get("hit_and_run_consecutive_hits_threshold", 2))
            hr_pen = float(cfg.get("hit_and_run_disengage_penalty", 0.0))
            hr_rew = float(cfg.get("hit_and_run_disengage_reward", 0.0))
            if (
                dealt_damage_this_step
                and visible_enemy_count > 0
                and self._consecutive_hits >= hr_thr
                and prev_dist != float("inf")
                and curr_dist != float("inf")
            ):
                if dist_change > 0:
                    if hr_rew != 0.0:
                        reward += hr_rew
                        rd["hit_and_run_disengage"] = rd.get("hit_and_run_disengage", 0.0) + hr_rew
                else:
                    if hr_pen != 0.0:
                        reward += hr_pen
                        rd["hit_and_run_disengage"] = rd.get("hit_and_run_disengage", 0.0) + hr_pen

            # 다음 스텝 거리 변화 계산을 위해 갱신
            self._prev_enemy_dist = curr_dist

            # 적 넥서스 접근 보상
            approach_nexus = cfg.get("approach_enemy_nexus", 0.05)
            if approach_nexus != 0.0:
                own_n = self.nexuses[agent.team_id]
                own_ratio = (own_n.hp / max(own_n.max_hp, 1)) if own_n.alive else 0.0
                own_low_thr = float(cfg.get("own_nexus_low_hp_threshold", 0.35))
                own_low_mult = float(cfg.get("own_nexus_low_approach_enemy_nexus_mult", 0.15))
                en = self.nexuses[1 - agent.team_id]
                if en.alive:
                    curr_n_dist = abs(agent.y - en.y) + abs(agent.x - en.x)
                    prev_n_dist = getattr(self, "_prev_nexus_dist", curr_n_dist)
                    r = approach_nexus * (prev_n_dist - curr_n_dist)
                    if visible_enemy_count > 0:
                        nm = float(cfg.get("visible_enemy_nexus_mult", 1.0))
                        r *= nm
                    if own_ratio <= own_low_thr:
                        r *= own_low_mult
                    reward += r
                    rd["approach_nexus"] = r
                    self._prev_nexus_dist = curr_n_dist

            # 적 넥서스 인접 유지 보상: 넥서스 앞까지 왔는데 타격을 못하는 현상 완화
            adjacent_nexus_bonus = float(cfg.get("adjacent_enemy_nexus_bonus", 0.0))
            if adjacent_nexus_bonus != 0.0:
                own_n = self.nexuses[agent.team_id]
                own_ratio = (own_n.hp / max(own_n.max_hp, 1)) if own_n.alive else 0.0
                own_low_thr = float(cfg.get("own_nexus_low_hp_threshold", 0.35))
                own_low_mult = float(cfg.get("own_nexus_low_approach_enemy_nexus_mult", 0.15))
                en = self.nexuses[1 - agent.team_id]
                if en.alive:
                    dist_to_enemy_nexus = abs(agent.y - en.y) + abs(agent.x - en.x)
                    if dist_to_enemy_nexus == 1:
                        bonus = adjacent_nexus_bonus
                        if own_ratio <= own_low_thr:
                            bonus *= own_low_mult
                        reward += bonus
                        rd["adjacent_enemy_nexus"] = rd.get("adjacent_enemy_nexus", 0.0) + bonus

            # 내 넥서스가 부실할 때(수비 필요): 내 넥서스에서 너무 멀어지면 페널티
            own_n = self.nexuses[agent.team_id]
            own_ratio = (own_n.hp / max(own_n.max_hp, 1)) if own_n.alive else 0.0
            if agent.alive and own_ratio <= float(cfg.get("own_nexus_low_hp_threshold", 0.35)):
                dist_to_own = abs(agent.y - own_n.y) + abs(agent.x - own_n.x)
                stay_radius = int(cfg.get("stay_near_radius", 4))
                excess = dist_to_own - stay_radius
                if excess > 0:
                    pen_per_tile = float(cfg.get("own_nexus_low_far_penalty_per_tile", -0.04))
                    pen = pen_per_tile * excess
                    reward += pen
                    rd["own_nexus_low_far"] = rd.get("own_nexus_low_far", 0.0) + pen

            # 힐러 팀원 접근 보상
            approach_tm = cfg.get("approach_teammate", 0.0)
            if approach_tm != 0.0:
                curr_tm = self._get_nearest_low_hp_teammate_dist(agent_idx)
                prev_tm = getattr(self, "_prev_teammate_dist", curr_tm)
                if prev_tm != float("inf") and curr_tm != float("inf"):
                    r = approach_tm * (prev_tm - curr_tm)
                    if r != 0.0:
                        reward += r
                        rd["approach_teammate"] = r

            # 자기 넥서스 근처에만 머무르는 패널티 (적이 멀리 있을 때)
            stay_penalty = cfg.get("stay_near_own_nexus_penalty", 0.0)
            if stay_penalty != 0.0:
                own_n = self.nexuses[agent.team_id]
                dist_to_own = abs(agent.y - own_n.y) + abs(agent.x - own_n.x)
                own_ratio = (own_n.hp / max(own_n.max_hp, 1)) if own_n.alive else 0.0
                own_low_thr = float(cfg.get("own_nexus_low_hp_threshold", 0.35))
                stay_radius = int(cfg.get("stay_near_radius", 4))
                enemy_dist = self._get_nearest_enemy_dist(agent_idx)
                enemy_far_threshold = int(cfg.get("stay_near_enemy_far", 6))
                # 내 넥서스가 부실하면 '농성' 패널티를 끄고 복귀를 유도한다.
                if own_ratio > own_low_thr and dist_to_own <= stay_radius and enemy_dist > enemy_far_threshold:
                    reward += stay_penalty
                    rd["stay_near_own_nexus"] = stay_penalty

            # 넥서스 방어 보상 (아군 넥서스 근처에 있을 때)
            defend_bonus = cfg.get("defend_nexus_bonus", 0.0)
            if defend_bonus != 0.0:
                own_n = self.nexuses[agent.team_id]
                defend_radius = int(cfg.get("defend_nexus_radius", 4))
                dist_to_own = abs(agent.y - own_n.y) + abs(agent.x - own_n.x)
                # 근처에 적이 있을 때만 방어 보상
                if dist_to_own <= defend_radius:
                    enemy_near = any(
                        not a.is_teammate(agent) and a.alive and a.agent_id != agent_idx
                        and abs(a.y - own_n.y) + abs(a.x - own_n.x) <= defend_radius + 2
                        for a in self.agents
                    )
                    if enemy_near:
                        reward += defend_bonus
                        rd["defend_nexus"] = defend_bonus

        # 반복 이동 패널티
        osc = cfg.get("oscillation_penalty", -0.05)
        if osc != 0.0 and len(self._action_history) >= 3:
            h = self._action_history
            opposites = {
                ACTION_UP: ACTION_DOWN, ACTION_DOWN: ACTION_UP,
                ACTION_LEFT: ACTION_RIGHT, ACTION_RIGHT: ACTION_LEFT,
            }
            if h[-3] == h[-1] and h[-3] in opposites and opposites[h[-3]] == h[-2]:
                reward += osc
                rd["oscillation"] = osc

        # === 종료 보상 ===
        game_over = self._check_game_over()
        is_truncated = self.current_step >= self.max_steps

        if game_over:
            # 어느 넥서스가 파괴됐는지 확인
            own_nexus = self.nexuses[agent.team_id]
            enemy_nexus = self.nexuses[1 - agent.team_id]
            if not enemy_nexus.alive:
                r = cfg.get("nexus_destroyed_win", 30.0)
                reward += r
                rd["nexus_destroyed_win"] = r
            elif not own_nexus.alive:
                r = cfg.get("nexus_destroyed_loss", -20.0)
                reward += r
                rd["nexus_destroyed_loss"] = r

        elif is_truncated:
            # 시간 초과: 넥서스 HP 비율로 판정
            own_ratio = self.nexuses[agent.team_id].hp / self.nexuses[agent.team_id].max_hp
            enemy_ratio = self.nexuses[1 - agent.team_id].hp / self.nexuses[1 - agent.team_id].max_hp
            if own_ratio > enemy_ratio:
                r = cfg.get("timeout_nexus_advantage", 10.0)
                reward += r
                rd["timeout_advantage"] = r
            elif own_ratio < enemy_ratio:
                r = cfg.get("timeout_nexus_disadvantage", -8.0)
                reward += r
                rd["timeout_disadvantage"] = r

        self._last_reward_details = rd
        return reward

    # ─────────────────── 게임 종료 ───────────────────────────

    def _check_game_over(self) -> bool:
        """넥서스가 파괴되면 게임 종료."""
        return any(not n.alive for n in self.nexuses)

    def _get_winning_team(self) -> int | None:
        """승리 팀을 반환한다."""
        for n in self.nexuses:
            if not n.alive:
                return 1 - n.team_id  # 파괴된 넥서스의 상대 팀이 승리
        return None

    # ─────────────────── 정보 ────────────────────────────────

    def _get_info(self) -> dict:
        """추가 정보를 반환한다."""
        return {
            "step": self.current_step,
            "mode": "nexus",
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
                    "death_count": self._death_counts.get(a.agent_id, 0),
                    "respawn_timer": self._respawn_timers.get(a.agent_id, 0),
                }
                for a in self.agents
            ],
            "nexuses": [
                {
                    "team_id": n.team_id,
                    "hp": n.hp,
                    "max_hp": n.max_hp,
                    "alive": n.alive,
                    "position": (n.y, n.x),
                }
                for n in self.nexuses
            ],
            "minions_alive": sum(1 for m in self.minions if m.alive),
            "reward_details": self._last_reward_details,
            "winning_team": self._get_winning_team() if self._check_game_over() else None,
            "render_events": getattr(self, "_render_events", []),
        }
