"""미니언 클래스 — 규칙 기반 NPC"""


class Minion:
    """규칙 기반 AI로 제어되는 NPC 유닛.
    적 에이전트를 우선 공격하고, 없으면 적 넥서스를 공격한다.
    """

    def __init__(self, minion_id: int, team_id: int, x: int, y: int, config: dict):
        self.minion_id = minion_id
        self.team_id = team_id
        self.x = x
        self.y = y
        minion_cfg = config.get("minion", {})
        self.hp = minion_cfg.get("hp", 30)
        self.max_hp = self.hp
        self.attack = minion_cfg.get("attack", 5)
        self.alive = True

    @property
    def position(self) -> tuple[int, int]:
        return (self.y, self.x)

    def take_damage(self, damage: int) -> int:
        """데미지를 받는다. 실제 데미지를 반환한다."""
        actual = min(self.hp, max(0, damage))
        self.hp -= actual
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
        return actual

    def __repr__(self):
        return (f"Minion(id={self.minion_id}, team={self.team_id}, "
                f"pos=({self.x},{self.y}), hp={self.hp}, alive={self.alive})")
