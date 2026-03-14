"""미니언 + MinionAI 유닛 테스트"""

import unittest
import numpy as np

from env.minion import Minion
from env.minion_ai import MinionAI
from env.agent import Agent
from env.nexus import Nexus
from env.map_generator import TILE_WALL


def _make_config(**overrides):
    cfg = {
        "minion": {"hp": 30, "attack": 5, "detection_range": 5},
        "nexus": {"hp": 500},
        "agent": {"initial_hp": 100, "initial_attack": 10, "initial_defense": 0},
        "roles": {
            "tank": {"hp": 100, "attack": 10, "defense": 5, "attack_range": 1, "attack_cooldown_steps": 0},
            "dealer": {"hp": 70, "attack": 15, "defense": 0, "attack_range": 3, "attack_cooldown_steps": 0},
            "healer": {"hp": 100, "attack": 6, "defense": 2, "attack_range": 1, "heal_amount": 20,
                        "heal_cooldown_steps": 0},
        },
        "items": {"enabled": False},
    }
    cfg.update(overrides)
    return cfg


class TestMinion(unittest.TestCase):
    """미니언 클래스 기본 동작 테스트."""

    def test_init(self):
        config = _make_config()
        m = Minion(0, team_id=0, x=3, y=4, config=config)
        self.assertEqual(m.hp, 30)
        self.assertEqual(m.attack, 5)
        self.assertTrue(m.alive)
        self.assertEqual(m.position, (4, 3))

    def test_take_damage(self):
        m = Minion(0, 0, 0, 0, _make_config())
        actual = m.take_damage(10)
        self.assertEqual(actual, 10)
        self.assertEqual(m.hp, 20)
        self.assertTrue(m.alive)

    def test_take_damage_kill(self):
        m = Minion(0, 0, 0, 0, _make_config())
        m.take_damage(30)
        self.assertEqual(m.hp, 0)
        self.assertFalse(m.alive)

    def test_take_damage_overkill(self):
        m = Minion(0, 0, 0, 0, _make_config())
        actual = m.take_damage(100)
        self.assertEqual(actual, 30)
        self.assertEqual(m.hp, 0)
        self.assertFalse(m.alive)


class TestMinionAI(unittest.TestCase):
    """미니언 AI 행동 결정 테스트."""

    def _make_grid(self, w=10, h=10):
        return np.zeros((h, w), dtype=np.int32)

    def _make_agent(self, agent_id, team_id, x, y, role="tank"):
        config = _make_config()
        return Agent(agent_id, x, y, config, role=role, team_id=team_id)

    def _make_minion(self, minion_id, team_id, x, y):
        return Minion(minion_id, team_id, x, y, _make_config())

    def _make_nexus(self, team_id, x, y):
        return Nexus(team_id, x, y, _make_config())

    def test_attack_adjacent_enemy_agent(self):
        """인접 적 에이전트가 있으면 공격."""
        ai = MinionAI(_make_config())
        grid = self._make_grid()
        minion = self._make_minion(0, team_id=0, x=5, y=5)
        enemy = self._make_agent(0, team_id=1, x=5, y=4, role="tank")
        nexus = self._make_nexus(1, 9, 9)

        action = ai.get_action(minion, grid, [enemy], [minion], nexus)
        self.assertEqual(action[0], "attack_agent")
        self.assertIs(action[1], enemy)

    def test_attack_adjacent_nexus(self):
        """인접 적 넥서스 공격 (적 에이전트가 인접하지 않을 때)."""
        ai = MinionAI(_make_config())
        grid = self._make_grid()
        minion = self._make_minion(0, team_id=0, x=5, y=5)
        nexus = self._make_nexus(1, 5, 4)  # x=5, y=4 → 인접

        action = ai.get_action(minion, grid, [], [minion], nexus)
        self.assertEqual(action, ("attack_nexus",))

    def test_move_toward_enemy_in_detection_range(self):
        """감지 범위 내 적 에이전트 쪽으로 이동."""
        ai = MinionAI(_make_config())
        grid = self._make_grid()
        minion = self._make_minion(0, team_id=0, x=5, y=5)
        enemy = self._make_agent(0, team_id=1, x=5, y=2, role="tank")  # 거리 3
        nexus = self._make_nexus(1, 9, 9)

        action = ai.get_action(minion, grid, [enemy], [minion], nexus)
        self.assertEqual(action[0], "move")
        # y=2로 가야 하므로 dy=-1
        dy, dx = action[1], action[2]
        self.assertEqual(dy, -1)
        self.assertEqual(dx, 0)

    def test_move_toward_nexus_when_no_enemy(self):
        """적이 감지 범위 밖이면 넥서스로 이동."""
        ai = MinionAI(_make_config())
        grid = self._make_grid()
        minion = self._make_minion(0, team_id=0, x=1, y=1)
        nexus = self._make_nexus(1, 8, 8)

        action = ai.get_action(minion, grid, [], [minion], nexus)
        self.assertEqual(action[0], "move")

    def test_idle_when_blocked(self):
        """이동 경로가 막혀있으면 idle."""
        ai = MinionAI(_make_config())
        grid = self._make_grid(3, 3)
        # 미니언을 (1,1)에 놓고, 사방을 벽으로 막음
        grid[0, 1] = TILE_WALL
        grid[2, 1] = TILE_WALL
        grid[1, 0] = TILE_WALL
        grid[1, 2] = TILE_WALL
        minion = self._make_minion(0, team_id=0, x=1, y=1)
        nexus = self._make_nexus(1, 0, 0)
        nexus.y = 0; nexus.x = 0

        action = ai.get_action(minion, grid, [], [minion], nexus)
        self.assertEqual(action, ("idle",))

    def test_agent_priority_over_nexus(self):
        """적 에이전트와 넥서스가 모두 인접해도 에이전트 우선 공격."""
        ai = MinionAI(_make_config())
        grid = self._make_grid()
        minion = self._make_minion(0, team_id=0, x=5, y=5)
        enemy = self._make_agent(0, team_id=1, x=5, y=4, role="dealer")
        nexus = self._make_nexus(1, 5, 6)  # x=5, y=6 → 인접

        action = ai.get_action(minion, grid, [enemy], [minion], nexus)
        self.assertEqual(action[0], "attack_agent")

    def test_ignores_dead_agents(self):
        """사망한 에이전트는 무시."""
        ai = MinionAI(_make_config())
        grid = self._make_grid()
        minion = self._make_minion(0, team_id=0, x=5, y=5)
        dead_enemy = self._make_agent(0, team_id=1, x=5, y=4, role="tank")
        dead_enemy.alive = False
        dead_enemy.hp = 0
        nexus = self._make_nexus(1, 9, 9)

        action = ai.get_action(minion, grid, [dead_enemy], [minion], nexus)
        # 사망한 에이전트 무시 → 넥서스로 이동
        self.assertNotEqual(action[0], "attack_agent")

    def test_ignores_friendly_agents(self):
        """같은 팀 에이전트는 공격 대상이 아님."""
        ai = MinionAI(_make_config())
        grid = self._make_grid()
        minion = self._make_minion(0, team_id=0, x=5, y=5)
        ally = self._make_agent(0, team_id=0, x=5, y=4, role="tank")  # 같은 팀
        nexus = self._make_nexus(1, 9, 9)

        action = ai.get_action(minion, grid, [ally], [minion], nexus)
        self.assertNotEqual(action[0], "attack_agent")


if __name__ == "__main__":
    unittest.main()
