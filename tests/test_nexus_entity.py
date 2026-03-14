"""Nexus 엔티티 유닛 테스트"""

import unittest

from env.nexus import Nexus


class TestNexus(unittest.TestCase):
    """넥서스 클래스 기본 동작 테스트."""

    def _make_nexus(self, **kwargs):
        config = {"nexus": {"hp": kwargs.get("hp", 500)}}
        return Nexus(team_id=0, x=5, y=5, config=config)

    def test_init(self):
        n = self._make_nexus()
        self.assertEqual(n.team_id, 0)
        self.assertEqual(n.hp, 500)
        self.assertEqual(n.max_hp, 500)
        self.assertTrue(n.alive)
        self.assertEqual(n.position, (5, 5))

    def test_take_damage(self):
        n = self._make_nexus(hp=100)
        actual = n.take_damage(30)
        self.assertEqual(actual, 30)
        self.assertEqual(n.hp, 70)
        self.assertTrue(n.alive)

    def test_take_damage_overkill(self):
        n = self._make_nexus(hp=50)
        actual = n.take_damage(100)
        self.assertEqual(actual, 50)  # HP를 넘지 않는 실제 데미지
        self.assertEqual(n.hp, 0)
        self.assertFalse(n.alive)

    def test_take_damage_exact_kill(self):
        n = self._make_nexus(hp=30)
        actual = n.take_damage(30)
        self.assertEqual(actual, 30)
        self.assertEqual(n.hp, 0)
        self.assertFalse(n.alive)

    def test_take_zero_damage(self):
        n = self._make_nexus(hp=100)
        actual = n.take_damage(0)
        self.assertEqual(actual, 0)
        self.assertEqual(n.hp, 100)
        self.assertTrue(n.alive)

    def test_take_negative_damage_ignored(self):
        n = self._make_nexus(hp=100)
        actual = n.take_damage(-10)
        self.assertEqual(actual, 0)
        self.assertEqual(n.hp, 100)

    def test_repr(self):
        n = self._make_nexus()
        r = repr(n)
        self.assertIn("Nexus", r)
        self.assertIn("team=0", r)


if __name__ == "__main__":
    unittest.main()
