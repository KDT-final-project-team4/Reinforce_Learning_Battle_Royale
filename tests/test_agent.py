"""에이전트 로직 유닛 테스트"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agent import Agent


DEFAULT_CONFIG = {
    "agent": {
        "initial_hp": 100,
        "initial_attack": 10,
        "initial_defense": 0,
        "view_range": 5,
    }
}


class TestAgent(unittest.TestCase):

    def test_init(self):
        agent = Agent(0, 3, 4, DEFAULT_CONFIG)
        self.assertEqual(agent.agent_id, 0)
        self.assertEqual(agent.x, 3)
        self.assertEqual(agent.y, 4)
        self.assertEqual(agent.hp, 100)
        self.assertEqual(agent.attack, 10)
        self.assertEqual(agent.defense, 0)
        self.assertTrue(agent.alive)

    def test_take_damage_basic(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        actual = agent.take_damage(10)
        self.assertEqual(actual, 10)  # defense=0이므로 그대로
        self.assertEqual(agent.hp, 90)

    def test_take_damage_with_defense(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        agent.defense = 5
        actual = agent.take_damage(10)
        # 비율 감소: reduction = 5/(5+20) = 0.2, damage = int(10*0.8) = 8
        self.assertEqual(actual, 8)
        self.assertEqual(agent.hp, 92)

    def test_take_damage_min_1(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        agent.defense = 100  # 방어력이 공격력보다 높아도 최소 1
        actual = agent.take_damage(10)
        self.assertEqual(actual, 1)
        self.assertEqual(agent.hp, 99)

    def test_death(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        agent.take_damage(100)
        self.assertEqual(agent.hp, 0)
        self.assertFalse(agent.alive)

    def test_overkill(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        agent.take_damage(200)
        self.assertEqual(agent.hp, 0)  # 음수가 되면 안 됨
        self.assertFalse(agent.alive)

    def test_heal(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        agent.take_damage(50)
        healed = agent.heal(30)
        self.assertEqual(healed, 30)
        self.assertEqual(agent.hp, 80)

    def test_heal_cap(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        agent.take_damage(10)
        healed = agent.heal(50)
        self.assertEqual(healed, 10)  # 최대 HP 초과 안됨
        self.assertEqual(agent.hp, 100)

    def test_move(self):
        agent = Agent(0, 5, 5, DEFAULT_CONFIG)
        agent.move(0, -1)  # 위로
        self.assertEqual(agent.x, 5)
        self.assertEqual(agent.y, 4)

    def test_stats_normalized(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        stats = agent.get_stats()
        self.assertAlmostEqual(stats[0], 1.0)       # hp=100/100
        self.assertAlmostEqual(stats[1], 10/30.0)   # atk=10/30
        self.assertAlmostEqual(stats[2], 0.0)        # def=0/20

    def test_add_attack_defense(self):
        agent = Agent(0, 0, 0, DEFAULT_CONFIG)
        agent.add_attack(10)
        self.assertEqual(agent.attack, 20)
        agent.add_defense(5)
        self.assertEqual(agent.defense, 5)

    def test_position_property(self):
        agent = Agent(0, 3, 7, DEFAULT_CONFIG)
        self.assertEqual(agent.position, (3, 7))


if __name__ == "__main__":
    unittest.main()
