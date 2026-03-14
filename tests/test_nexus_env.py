"""넥서스 환경 통합 테스트"""

import unittest
import numpy as np

from env.nexus_env import NexusBattleEnv
from env.base_env import ACTION_STAY, ACTION_MELEE, ROLE_ACTION_MAP


class TestNexusEnvReset(unittest.TestCase):
    """reset() 관련 테스트."""

    def _make_env(self, **config_overrides):
        import yaml
        with open("config/nexus_mode.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config.update(config_overrides)
        return NexusBattleEnv(config=config)

    def test_reset_returns_obs_and_info(self):
        env = self._make_env()
        obs, info = env.reset(seed=42)
        self.assertIsNotNone(obs)
        self.assertIsInstance(info, dict)
        env.close()

    def test_obs_shape(self):
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, env.observation_space.shape)
        env.close()

    def test_nexuses_created(self):
        env = self._make_env()
        env.reset(seed=42)
        self.assertEqual(len(env.nexuses), 2)
        self.assertEqual(env.nexuses[0].team_id, 0)
        self.assertEqual(env.nexuses[1].team_id, 1)
        self.assertTrue(env.nexuses[0].alive)
        self.assertTrue(env.nexuses[1].alive)
        self.assertEqual(env.nexuses[0].hp, 500)
        env.close()

    def test_agents_created_with_roles(self):
        env = self._make_env()
        env.reset(seed=42)
        self.assertEqual(len(env.agents), 6)  # 2팀 x 3명

        # 각 팀의 역할 확인: 탱커, 딜러, 힐러 각 1명
        for team_id in range(2):
            team_agents = [a for a in env.agents if a.team_id == team_id]
            self.assertEqual(len(team_agents), 3)
            roles = {a.role for a in team_agents}
            self.assertEqual(roles, {"tank", "dealer", "healer"})
        env.close()

    def test_agents_spawn_near_nexus(self):
        env = self._make_env()
        env.reset(seed=42)
        spawn_radius = env._spawn_radius
        for team_id in range(2):
            nexus = env.nexuses[team_id]
            team_agents = [a for a in env.agents if a.team_id == team_id]
            for a in team_agents:
                dist = abs(a.y - nexus.y) + abs(a.x - nexus.x)
                self.assertLessEqual(dist, spawn_radius,
                    f"Agent {a.agent_id} too far from nexus: dist={dist}")
        env.close()

    def test_no_zone_damage(self):
        """자기장이 비활성화되어 있는지 확인."""
        env = self._make_env()
        env.reset(seed=42)
        self.assertFalse(env.zone_manager.enabled)
        env.close()

    def test_minions_empty_on_reset(self):
        env = self._make_env()
        env.reset(seed=42)
        self.assertEqual(len(env.minions), 0)
        env.close()

    def test_death_counts_initialized(self):
        env = self._make_env()
        env.reset(seed=42)
        for i in range(6):
            self.assertEqual(env._death_counts[i], 0)
        env.close()

    def test_info_contains_nexus_mode(self):
        env = self._make_env()
        _, info = env.reset(seed=42)
        self.assertEqual(info["mode"], "nexus")
        self.assertIn("nexuses", info)
        env.close()


class TestNexusEnvStep(unittest.TestCase):
    """step() 동작 관련 테스트."""

    def _make_env(self):
        import yaml
        with open("config/nexus_mode.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return NexusBattleEnv(config=config)

    def test_step_returns_correct_format(self):
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        action = 0  # ACTION_UP
        obs2, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(obs2.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        env.close()

    def test_run_multiple_steps(self):
        """여러 스텝을 에러 없이 실행할 수 있는지 확인."""
        env = self._make_env()
        env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    def test_nexus_destroyed_ends_game(self):
        """넥서스가 파괴되면 게임이 끝나는지 확인."""
        env = self._make_env()
        env.reset(seed=42)
        # 적 넥서스 HP를 1로 설정
        env.nexuses[1].hp = 1
        # 강제 게임 종료 확인
        env.nexuses[1].take_damage(1)
        self.assertFalse(env.nexuses[1].alive)
        self.assertTrue(env._check_game_over())
        env.close()

    def test_game_truncated_at_max_steps(self):
        """max_steps에 도달하면 truncated."""
        env = self._make_env()
        env.reset(seed=42)
        terminated = False
        truncated = False
        for i in range(env.max_steps + 10):
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, info = env.step(ACTION_STAY)
        self.assertTrue(terminated or truncated,
                        "Game should end by max_steps")
        env.close()

    def test_reward_details_in_info(self):
        env = self._make_env()
        env.reset(seed=42)
        _, _, _, _, info = env.step(ACTION_STAY)
        self.assertIn("reward_details", info)
        env.close()

    def test_learning_role(self):
        """learning_role 설정 시 agent 0의 역할이 고정되는지 확인."""
        import yaml
        with open("config/nexus_mode.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        env = NexusBattleEnv(config=config, learning_role="healer")
        env.reset(seed=42)
        self.assertEqual(env.agents[0].role, "healer")
        self.assertEqual(env.agents[0].team_id, 0)
        env.close()


class TestNexusEnvRespawn(unittest.TestCase):
    """부활 시스템 테스트."""

    def _make_env(self):
        import yaml
        with open("config/nexus_mode.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return NexusBattleEnv(config=config)

    def test_agent_death_adds_respawn_timer(self):
        """에이전트 사망 시 부활 타이머가 설정되는지 확인."""
        env = self._make_env()
        env.reset(seed=42)
        # 에이전트를 강제 사망 (death_step을 current_step에 맞춰야 함)
        agent = env.agents[1]
        agent.hp = 0
        agent.alive = False
        agent.death_step = env.current_step
        env._process_agent_deaths()
        self.assertIn(agent.agent_id, env._respawn_timers)
        self.assertGreater(env._respawn_timers[agent.agent_id], 0)
        env.close()

    def test_respawn_timer_increases_with_deaths(self):
        """사망 횟수가 늘수록 부활 시간이 늘어나는지 확인."""
        env = self._make_env()
        env.reset(seed=42)
        agent = env.agents[1]
        base_time = env._respawn_base_time
        increment = env._respawn_increment

        # 첫 번째 사망
        agent.hp = 0
        agent.alive = False
        agent.death_step = env.current_step
        env._process_agent_deaths()
        first_timer = env._respawn_timers[agent.agent_id]
        self.assertEqual(first_timer, base_time)  # base_time + 0 * increment

        # 부활 시키고 다시 사망
        agent.alive = True
        agent.hp = 100
        del env._respawn_timers[agent.agent_id]
        env.current_step += 1
        agent.hp = 0
        agent.alive = False
        agent.death_step = env.current_step
        env._process_agent_deaths()
        second_timer = env._respawn_timers[agent.agent_id]
        self.assertEqual(second_timer, base_time + increment)
        env.close()


class TestNexusEnvMinions(unittest.TestCase):
    """미니언 스폰 테스트."""

    def _make_env(self):
        import yaml
        with open("config/nexus_mode.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return NexusBattleEnv(config=config)

    def test_minions_spawn_after_start_step(self):
        """spawn_start_step 이후 미니언이 생성되는지 확인."""
        env = self._make_env()
        env.reset(seed=42)
        minion_cfg = env.config.get("minion", {})
        spawn_start = minion_cfg.get("spawn_start_step", 15)

        # spawn_start 스텝까지 진행
        for _ in range(spawn_start):
            obs, _, terminated, truncated, _ = env.step(ACTION_STAY)
            if terminated or truncated:
                break

        # 미니언이 생성되었는지 확인
        self.assertGreater(len(env.minions), 0,
            f"Minions should spawn by step {spawn_start}")
        env.close()

    def test_minion_max_per_team_respected(self):
        """팀당 미니언 최대 수가 지켜지는지 확인."""
        env = self._make_env()
        env.reset(seed=42)
        max_per_team = env._minion_max_per_team

        # 많은 스텝 진행
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(ACTION_STAY)
            if terminated or truncated:
                break
            for tid in range(2):
                team_count = sum(1 for m in env.minions if m.team_id == tid and m.alive)
                self.assertLessEqual(team_count, max_per_team)
        env.close()


if __name__ == "__main__":
    unittest.main()
