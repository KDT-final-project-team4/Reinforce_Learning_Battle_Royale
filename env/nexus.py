"""넥서스 클래스 — 파괴 가능 팀 구조물"""


class Nexus:
    """팀의 넥서스. HP가 0이 되면 해당 팀이 패배한다."""

    def __init__(self, team_id: int, x: int, y: int, config: dict):
        self.team_id = team_id
        self.x = x
        self.y = y
        nexus_cfg = config.get("nexus", {})
        self.hp = nexus_cfg.get("hp", 500)
        self.max_hp = self.hp
        self.alive = True

    @property
    def position(self) -> tuple[int, int]:
        return (self.y, self.x)

    def take_damage(self, damage: int) -> int:
        """넥서스에 데미지를 입힌다. 실제 데미지를 반환한다."""
        actual = min(self.hp, max(0, damage))
        self.hp -= actual
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
        return actual

    def __repr__(self):
        return (f"Nexus(team={self.team_id}, pos=({self.x},{self.y}), "
                f"hp={self.hp}/{self.max_hp}, alive={self.alive})")
