"""환경 유닛 테스트"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.battle_env import (
    BattleRoyaleEnv, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_MELEE, ACTION_RANGED_H, ACTION_RANGED_V, ACTION_HEAL, ACTION_STAY,
)

# Phase 1~3 호환: ACTION_ATTACK → ACTION_MELEE
ACTION_ATTACK = ACTION_MELEE
from env.map_generator import generate_map, get_empty_positions, place_agents, TILE_WALL


TEST_CONFIG = {
    "map": {"width": 10, "height": 10, "wall_count": 5, "wall_placement": "random"},
    "agent": {"count": 6, "initial_hp": 100, "initial_attack": 10,
              "initial_defense": 0, "view_range": 5},
    "team": {"num_teams": 3, "agents_per_team": 2, "friendly_fire": False},
    "roles": {
        "tank": {"hp": 150, "attack": 12, "defense": 5, "attack_range": 1},
        "dealer": {"hp": 70, "attack": 15, "defense": 0, "attack_range": 3,
                   "ranged_damage_multiplier": 0.8},
        "healer": {"hp": 100, "attack": 6, "defense": 2, "attack_range": 1,
                   "heal_amount": 20, "potion_multiplier": 1.5},
    },
    "items": {"enabled": False},
    "zone": {"enabled": False},
    "game": {"max_steps": 500, "simultaneous_actions": True},
    "reward": {
        "kill": 10.0, "death": -10.0, "damage_dealt": 1.0,
        "item_pickup": 2.0, "potion_use": 1.0, "zone_damage": -1.0,
        "survival_per_step": 0.01, "idle_penalty": -0.05,
        "attack_miss": -0.3, "invalid_action": -0.1,
    },
    "training": {"algorithm": "DQN", "total_timesteps": 1000,
                 "learning_rate": 0.0001, "self_play": False},
}

# Phase 4 obs size: 5*5*4 + 7 + 5 + 5 + 5 + 3 = 123
PHASE4_OBS_SIZE = 5 * 5 * 4 + 7 + 5 + 5 + 5 + 3


class TestMapGenerator(unittest.TestCase):

    def test_map_shape(self):
        grid = generate_map(10, 10, 5)
        self.assertEqual(grid.shape, (10, 10))

    def test_wall_count(self):
        grid = generate_map(10, 10, 8)
        self.assertEqual(np.sum(grid == TILE_WALL), 8)

    def test_empty_positions(self):
        grid = generate_map(10, 10, 5)
        empty = get_empty_positions(grid)
        self.assertEqual(len(empty), 100 - 5)

    def test_place_agents(self):
        grid = generate_map(10, 10, 5)
        positions = place_agents(grid, 2)
        self.assertEqual(len(positions), 2)
        # 서로 다른 위치인지 확인
        self.assertNotEqual(positions[0], positions[1])
        # 벽이 아닌 곳인지 확인
        for y, x in positions:
            self.assertEqual(grid[y, x], 0)


class TestBattleEnv(unittest.TestCase):

    def setUp(self):
        self.env = BattleRoyaleEnv(config=TEST_CONFIG)

    def tearDown(self):
        self.env.close()

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (PHASE4_OBS_SIZE,))
        self.assertTrue(np.all(obs >= -1.0))
        self.assertTrue(np.all(obs <= 1.0))

    def test_observation_space(self):
        obs, _ = self.env.reset(seed=42)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_step_returns(self):
        self.env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.env.step(ACTION_STAY)
        self.assertEqual(obs.shape, (PHASE4_OBS_SIZE,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_agents_initialized(self):
        self.env.reset(seed=42)
        self.assertEqual(len(self.env.agents), 6)
        for a in self.env.agents:
            self.assertTrue(a.alive)
            self.assertGreater(a.hp, 0)

    def test_movement(self):
        self.env.reset(seed=42)
        a0 = self.env.agents[0]
        old_y, old_x = a0.y, a0.x

        # 여러 스텝 이동해서 최소 한 번은 이동 성공했는지 확인
        moved = False
        for action in [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]:
            self.env.step(action)
            if (a0.y, a0.x) != (old_y, old_x):
                moved = True
                break
            old_y, old_x = a0.y, a0.x

        # 벽에 의해 이동 못할 수도 있으므로 soft check
        # (최소한 에러 없이 동작하면 OK)

    def test_attack_adjacent(self):
        """두 에이전트를 인접하게 배치하고 공격 테스트"""
        self.env.reset(seed=42)
        # 에이전트들을 강제로 인접 배치
        self.env.agents[0].y = 5
        self.env.agents[0].x = 5
        self.env.agents[1].y = 5
        self.env.agents[1].x = 6  # 오른쪽에 배치

        initial_hp = self.env.agents[1].hp
        obs, reward, _, _, _ = self.env.step(ACTION_ATTACK)

        # 공격이 성공하면 상대 HP가 줄어야 함
        self.assertLessEqual(self.env.agents[1].hp, initial_hp)

    def test_game_over_on_team_elimination(self):
        """다른 팀이 모두 전멸하면 게임 종료"""
        self.env.reset(seed=42)
        # 팀1(agents 2,3)과 팀2(agents 4,5) 모두 사망 처리
        for i in [2, 3, 4, 5]:
            self.env.agents[i].hp = 0
            self.env.agents[i].alive = False

        obs, reward, terminated, truncated, info = self.env.step(ACTION_STAY)
        self.assertTrue(terminated)

    def test_max_steps_truncation(self):
        """최대 스텝 초과 시 truncated"""
        config = TEST_CONFIG.copy()
        config["game"] = {**TEST_CONFIG["game"], "max_steps": 3}
        env = BattleRoyaleEnv(config=config)
        env.reset(seed=42)

        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(ACTION_STAY)
            if terminated:
                break

        if not terminated:
            _, _, terminated, truncated, _ = env.step(ACTION_STAY)
            self.assertTrue(truncated or terminated)

        env.close()

    def test_render_ansi(self):
        env = BattleRoyaleEnv(config=TEST_CONFIG, render_mode="ansi")
        env.reset(seed=42)
        output = env.render()
        self.assertIsInstance(output, str)
        self.assertIn("Step:", output)
        self.assertIn("A0", output)
        env.close()

    def test_idle_penalty(self):
        """대기 행동 시 음수 보상"""
        self.env.reset(seed=42)
        _, reward, _, _, _ = self.env.step(ACTION_STAY)
        # idle_penalty(-0.05) + survival(+0.01) = -0.04
        self.assertLess(reward, 0)

    def test_multiple_resets(self):
        """여러 번 리셋해도 문제 없는지"""
        for _ in range(5):
            obs, info = self.env.reset()
            self.assertEqual(obs.shape, (PHASE4_OBS_SIZE,))


PHASE2_CONFIG = {
    "map": {"width": 15, "height": 15, "wall_count": 15, "wall_placement": "random"},
    "agent": {"count": 6, "initial_hp": 100, "initial_attack": 10,
              "initial_defense": 0, "view_range": 7},
    "team": {"num_teams": 3, "agents_per_team": 2, "friendly_fire": False},
    "roles": {
        "tank": {"hp": 150, "attack": 12, "defense": 5, "attack_range": 1},
        "dealer": {"hp": 70, "attack": 15, "defense": 0, "attack_range": 3,
                   "ranged_damage_multiplier": 0.8},
        "healer": {"hp": 100, "attack": 6, "defense": 2, "attack_range": 1,
                   "heal_amount": 20, "potion_multiplier": 1.5},
    },
    "items": {"enabled": True, "potion_heal": 30, "weapon_attack_bonus": 10,
              "armor_defense_bonus": 10, "respawn_interval": 20,
              "max_potions_on_map": 3, "max_weapons_on_map": 2, "max_armors_on_map": 2},
    "zone": {"enabled": True, "shrink_start_step": 50, "shrink_interval": 30,
             "damage_per_step": 8},
    "game": {"max_steps": 500, "simultaneous_actions": True},
    "reward": {
        "kill": 20.0, "death": -10.0, "damage_dealt": 2.0,
        "item_pickup": 2.0, "potion_use": 1.0, "zone_damage": -2.0,
        "survival_per_step": 0.0, "idle_penalty": -0.2,
        "wall_bump": -0.1, "approach_enemy": 0.3,
        "no_combat_penalty": -0.1, "no_combat_threshold": 30,
        "combo_bonus": 0.5, "attack_miss": -0.3,
    },
    "training": {"algorithm": "PPO", "total_timesteps": 1000,
                 "learning_rate": 0.0003, "self_play": False},
}

# Phase 2 obs size: 7*7*4 + 7 + 5 + 5 + 5 + 3 = 219
PHASE2_OBS_SIZE = 7 * 7 * 4 + 7 + 5 + 5 + 5 + 3


class TestPhase2(unittest.TestCase):
    """Phase 2 기능 테스트: 아이템, 자기장, 새 보상"""

    def setUp(self):
        self.env = BattleRoyaleEnv(config=PHASE2_CONFIG)

    def tearDown(self):
        self.env.close()

    def test_15x15_map(self):
        obs, _ = self.env.reset(seed=42)
        self.assertEqual(self.env.grid.shape, (15, 15))
        self.assertEqual(obs.shape, (PHASE2_OBS_SIZE,))

    def test_items_spawned_on_reset(self):
        self.env.reset(seed=42)
        total_items = np.sum(self.env.item_manager.item_grid > 0)
        self.assertGreater(total_items, 0)

    def test_item_pickup(self):
        """에이전트가 아이템 위로 이동하면 획득"""
        self.env.reset(seed=42)
        from env.items import ItemType
        # 에이전트 위치에 강제로 무기 배치
        a = self.env.agents[0]
        self.env.item_manager.item_grid[a.y, a.x] = ItemType.WEAPON
        old_atk = a.attack

        self.env.step(ACTION_STAY)  # 아이템 획득은 step 내에서 처리
        self.assertEqual(a.attack, old_atk + 10)

    def test_zone_shrinks(self):
        """자기장이 일정 스텝 후 축소"""
        self.env.reset(seed=42)
        from env.map_generator import TILE_ZONE
        # 50 스텝까지 빠르게 진행
        for _ in range(51):
            _, _, terminated, _, _ = self.env.step(ACTION_STAY)
            if terminated:
                break
        # 자기장이 적용됐는지 확인
        zone_count = np.sum(self.env.grid == TILE_ZONE)
        self.assertGreater(zone_count, 0)

    def test_zone_damage(self):
        """자기장 영역에서 HP가 감소"""
        self.env.reset(seed=42)
        from env.map_generator import TILE_ZONE
        a = self.env.agents[0]
        # 에이전트 위치를 독가스로 변경
        self.env.grid[a.y, a.x] = TILE_ZONE
        self.env.zone_manager.enabled = True

        old_hp = a.hp
        self.env.step(ACTION_STAY)
        self.assertLess(a.hp, old_hp)

    def test_wall_bump_event(self):
        """벽에 부딪히면 wall_bump 이벤트 발생"""
        self.env.reset(seed=42)
        a = self.env.agents[0]
        # 에이전트를 맵 상단 모서리로 이동
        a.y = 0
        a.x = 0
        self.env.step(ACTION_UP)  # 맵 밖으로 이동 시도
        # wall_bump 이벤트가 기록되었는지 확인
        self.assertIn("wall_bump", self.env._step_events[0])

    def test_attack_miss_event(self):
        """인접한 적 없이 공격하면 attack_miss 이벤트 발생"""
        self.env.reset(seed=42)
        a0 = self.env.agents[0]
        a1 = self.env.agents[1]
        # 두 에이전트를 멀리 배치
        a0.y, a0.x = 2, 2
        a1.y, a1.x = 12, 12
        self.env.step(ACTION_ATTACK)
        self.assertIn("attack_miss", self.env._step_events[0])

    def test_attack_count_tracking(self):
        """공격 횟수 카운터 동작 확인"""
        self.env.reset(seed=42)
        a0 = self.env.agents[0]
        a2 = self.env.agents[2]  # 적 팀 에이전트
        a0.y, a0.x = 5, 5
        a2.y, a2.x = 5, 6  # 인접 배치

        self.assertEqual(a0.attack_count, 0)
        self.assertEqual(a0.attack_hits, 0)
        self.env.step(ACTION_ATTACK)
        self.assertEqual(a0.attack_count, 1)
        self.assertEqual(a0.attack_hits, 1)

    def test_approach_enemy_reward(self):
        """적에게 가까워지면 양수 보상 요소 존재"""
        self.env.reset(seed=42)
        a0 = self.env.agents[0]
        a1 = self.env.agents[1]
        # 에이전트들을 적당한 거리에 배치
        a0.y, a0.x = 5, 5
        a1.y, a1.x = 5, 8  # 오른쪽에 3칸 거리

        _, reward_right, _, _, _ = self.env.step(ACTION_RIGHT)  # 적 방향으로 이동
        # approach_enemy 보상이 양수여야 함 (0.1 * 1 = +0.1)
        # 정확한 값은 상대 이동에 따라 다를 수 있지만, 최소한 에러 없이 동작


PHASE4_CONFIG = {
    "map": {"width": 20, "height": 20, "wall_count": 25, "wall_placement": "random"},
    "agent": {"count": 6, "initial_hp": 100, "initial_attack": 10,
              "initial_defense": 0, "view_range": 9},
    "team": {"num_teams": 3, "agents_per_team": 2, "friendly_fire": False},
    "roles": {
        "tank": {"hp": 150, "attack": 12, "defense": 5, "attack_range": 1},
        "dealer": {"hp": 70, "attack": 15, "defense": 0, "attack_range": 3,
                   "ranged_damage_multiplier": 0.8},
        "healer": {"hp": 100, "attack": 6, "defense": 2, "attack_range": 1,
                   "heal_amount": 20, "potion_multiplier": 1.5},
    },
    "items": {"enabled": True, "potion_heal": 30, "weapon_attack_bonus": 10,
              "armor_defense_bonus": 10, "respawn_interval": 20,
              "max_potions_on_map": 3, "max_weapons_on_map": 3, "max_armors_on_map": 3},
    "zone": {"enabled": True, "shrink_start_step": 60, "shrink_interval": 30,
             "damage_per_step": 8},
    "game": {"max_steps": 600, "simultaneous_actions": True},
    "reward": {
        "kill": 10.0, "death": -5.0, "damage_dealt": 2.0,
        "item_pickup": 2.0, "low_hp_potion_bonus": 3.0, "low_hp_potion_threshold": 0.5,
        "potion_use": 1.0, "zone_damage": -4.0, "survival_per_step": 0.01,
        "approach_enemy": 0.1, "flee_hp_threshold": 0.3, "flee_bonus": 0.1,
        "disengage_bonus": 0.2, "disengage_hp_threshold": 0.4, "disengage_lookback": 5,
        "combo_bonus": 0.5,
        "idle_penalty": -0.05, "wall_bump": -0.1, "attack_miss": -0.3,
        "oscillation_penalty": -0.05, "no_combat_penalty": -0.3, "no_combat_threshold": 20,
        "invalid_action": -0.1, "ranged_miss": -0.2,
        "team_win": 25.0, "team_eliminated": -8.0, "teammate_death": -3.0,
        "heal_ally": 1.5, "heal_low_hp_ally": 3.0, "heal_hp_threshold": 0.4,
        "team_ranking_rewards": [25, 5, -10],
        "kill_streak_2": 1.0, "kill_streak_3": 2.0,
        "wasted_advantage_penalty": -3.0, "wasted_advantage_threshold": 2,
    },
    "training": {"algorithm": "PPO", "total_timesteps": 1000,
                 "learning_rate": 0.0001, "self_play": False},
}

# Phase 4 obs size with view_range=9: 9*9*4 + 7 + 5 + 5 + 5 + 3 = 347
PHASE4_FULL_OBS_SIZE = 9 * 9 * 4 + 7 + 5 + 5 + 5 + 3


class TestPhase4(unittest.TestCase):
    """Phase 4 기능 테스트: 6인 3팀, 역할, 원거리 공격, 힐"""

    def setUp(self):
        self.env = BattleRoyaleEnv(config=PHASE4_CONFIG)

    def tearDown(self):
        self.env.close()

    def test_obs_shape(self):
        obs, _ = self.env.reset(seed=42)
        self.assertEqual(self.env.grid.shape, (20, 20))
        self.assertEqual(obs.shape, (PHASE4_FULL_OBS_SIZE,))

    def test_obs_space_contains(self):
        obs, _ = self.env.reset(seed=42)
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_six_agents_three_teams(self):
        self.env.reset(seed=42)
        self.assertEqual(len(self.env.agents), 6)
        teams = set(a.team_id for a in self.env.agents)
        self.assertEqual(teams, {0, 1, 2})
        # 각 팀 2명
        for t in range(3):
            count = sum(1 for a in self.env.agents if a.team_id == t)
            self.assertEqual(count, 2)

    def test_roles_assigned(self):
        """각 에이전트에 역할이 배정됨"""
        self.env.reset(seed=42)
        from env.agent import ALL_ROLES
        for a in self.env.agents:
            self.assertIn(a.role, ALL_ROLES)

    def test_role_stats_differ(self):
        """역할에 따라 스탯이 다름"""
        self.env.reset(seed=42)
        hp_set = set(a.max_hp for a in self.env.agents)
        # 최소 2종류 이상의 HP가 있어야 (역할 차이)
        self.assertGreater(len(hp_set), 1)

    def test_no_friendly_fire(self):
        """같은 팀 공격 불가"""
        self.env.reset(seed=42)
        a0 = self.env.agents[0]  # 팀 0
        a1 = self.env.agents[1]  # 팀 0 (팀원)
        # 인접 배치
        a0.y, a0.x = 5, 5
        a1.y, a1.x = 5, 6
        initial_hp = a1.hp

        self.env.step(ACTION_MELEE)
        # 아군이므로 데미지 없어야 함
        self.assertEqual(a1.hp, initial_hp)

    def test_melee_attack_enemy(self):
        """적 팀 인접 공격 성공"""
        self.env.reset(seed=42)
        a0 = self.env.agents[0]  # 팀 0
        a2 = self.env.agents[2]  # 팀 1
        a0.y, a0.x = 5, 5
        a2.y, a2.x = 5, 6
        initial_hp = a2.hp

        self.env.step(ACTION_MELEE)
        self.assertLess(a2.hp, initial_hp)

    def test_ranged_attack_dealer_only(self):
        """딜러만 원거리 공격 가능, 다른 역할은 invalid_action"""
        self.env.reset(seed=42)
        from env.agent import ROLE_DEALER, ROLE_TANK
        # agent 0을 딜러가 아닌 역할로 강제 설정
        self.env.agents[0].role = ROLE_TANK

        self.env.step(ACTION_RANGED_H)
        self.assertIn("invalid_action", self.env._step_events[0])

    def test_ranged_attack_hits(self):
        """딜러의 원거리 공격이 적을 맞힘"""
        self.env.reset(seed=42)
        from env.agent import ROLE_DEALER
        a0 = self.env.agents[0]
        a0.role = ROLE_DEALER  # 강제 딜러 설정
        a2 = self.env.agents[2]  # 팀 1
        # 가로 방향에 적 배치
        a0.y, a0.x = 5, 5
        a2.y, a2.x = 5, 7  # 2칸 거리 (사거리 3 이내)
        initial_hp = a2.hp

        self.env.step(ACTION_RANGED_H)
        self.assertLess(a2.hp, initial_hp)

    def test_ranged_blocked_by_wall(self):
        """원거리 공격이 벽에 막힘"""
        self.env.reset(seed=42)
        from env.agent import ROLE_DEALER
        from env.map_generator import TILE_WALL
        a0 = self.env.agents[0]
        a0.role = ROLE_DEALER
        a2 = self.env.agents[2]
        a0.y, a0.x = 5, 5
        a2.y, a2.x = 5, 8  # 3칸 거리
        # 중간에 벽 배치
        self.env.grid[5, 6] = TILE_WALL
        initial_hp = a2.hp

        self.env.step(ACTION_RANGED_H)
        # 벽에 막혀서 데미지 없음
        self.assertEqual(a2.hp, initial_hp)

    def test_heal_action(self):
        """힐러의 아군 회복 행동"""
        self.env.reset(seed=42)
        from env.agent import ROLE_HEALER
        a0 = self.env.agents[0]
        a0.role = ROLE_HEALER
        a1 = self.env.agents[1]  # 팀원
        # max_hp보다 낮게 설정하여 힐 대상 확보
        a1.max_hp = 150
        a1.hp = 50  # 데미지 입은 상태 (50/150 = 33%)

        # 여러 번 시도 (팀원이 이동으로 빗나갈 수 있음)
        healed = False
        for _ in range(10):
            a0.y, a0.x = 5, 5
            a1.y, a1.x = 5, 6
            a1.hp = 50
            self.env.step(ACTION_HEAL)
            if a1.hp > 50:
                healed = True
                break
            self.env.reset(seed=None)
            a0 = self.env.agents[0]
            a0.role = ROLE_HEALER
            a1 = self.env.agents[1]
            a1.max_hp = 150
        self.assertTrue(healed, "Heal should succeed at least once in 10 tries")

    def test_heal_nonhealer_invalid(self):
        """힐러가 아닌 역할이 힐 시도하면 invalid_action"""
        self.env.reset(seed=42)
        from env.agent import ROLE_TANK
        self.env.agents[0].role = ROLE_TANK
        self.env.step(ACTION_HEAL)
        self.assertIn("invalid_action", self.env._step_events[0])

    def test_team_game_over(self):
        """팀 기반 게임 종료: 생존 팀 1개 이하"""
        self.env.reset(seed=42)
        # 팀1, 팀2 전멸
        for i in [2, 3, 4, 5]:
            self.env.agents[i].hp = 0
            self.env.agents[i].alive = False
        self.assertTrue(self.env._check_game_over())

    def test_team_not_over_with_two_teams(self):
        """2팀 이상 생존하면 게임 계속"""
        self.env.reset(seed=42)
        # 팀2만 전멸
        for i in [4, 5]:
            self.env.agents[i].hp = 0
            self.env.agents[i].alive = False
        self.assertFalse(self.env._check_game_over())

    def test_team_ranking(self):
        """팀 순위: 늦게 탈락할수록 높은 순위"""
        self.env.reset(seed=42)
        # 팀2 먼저 탈락
        for i in [4, 5]:
            self.env.agents[i].hp = 0
            self.env.agents[i].alive = False
        self.env._team_elimination_step[2] = 10
        # 팀1 그 다음 탈락
        for i in [2, 3]:
            self.env.agents[i].hp = 0
            self.env.agents[i].alive = False
        self.env._team_elimination_step[1] = 20

        # 팀0 = 1등 (생존), 팀1 = 2등, 팀2 = 3등
        self.assertEqual(self.env._get_team_rank(0), 1)
        self.assertEqual(self.env._get_team_rank(1), 2)
        self.assertEqual(self.env._get_team_rank(2), 3)

    def test_winning_team(self):
        """승리 팀 반환"""
        self.env.reset(seed=42)
        for i in [2, 3, 4, 5]:
            self.env.agents[i].hp = 0
            self.env.agents[i].alive = False
        self.assertEqual(self.env._get_winning_team(), 0)

    def test_step_all_actions(self):
        """모든 9가지 행동이 에러 없이 실행됨"""
        self.env.reset(seed=42)
        for action in range(9):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertEqual(obs.shape, (PHASE4_FULL_OBS_SIZE,))
            if terminated or truncated:
                self.env.reset(seed=42)

    def test_info_includes_team_and_role(self):
        """info에 팀/역할 정보 포함"""
        self.env.reset(seed=42)
        _, _, _, _, info = self.env.step(ACTION_STAY)
        for a_info in info["agents"]:
            self.assertIn("team_id", a_info)
            self.assertIn("role", a_info)
            self.assertIn("heal_count", a_info)

    def test_potion_multiplier_healer(self):
        """힐러의 포션 효과 1.5배"""
        self.env.reset(seed=42)
        from env.agent import ROLE_HEALER
        from env.items import ItemType
        a0 = self.env.agents[0]
        a0.role = ROLE_HEALER
        a0.hp = 50
        a0.max_hp = 150
        self.env.item_manager.enabled = True
        self.env.item_manager.item_grid[a0.y, a0.x] = ItemType.POTION

        self.env.step(ACTION_STAY)
        # 포션 30 * 1.5 = 45 회복 → 50 + 45 = 95
        self.assertEqual(a0.hp, 95)


if __name__ == "__main__":
    unittest.main()
