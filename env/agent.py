"""에이전트 클래스 — HP, 장비, 상태, 역할, 팀 관리"""

import numpy as np

# 역할 정의
ROLE_TANK = "tank"
ROLE_DEALER = "dealer"
ROLE_HEALER = "healer"
ALL_ROLES = [ROLE_TANK, ROLE_DEALER, ROLE_HEALER]


class Agent:
    """배틀로얄 에이전트. 위치, 체력, 공격력, 방어력, 역할, 팀을 관리한다."""

    def __init__(self, agent_id: int, x: int, y: int, config: dict,
                 role: str = ROLE_TANK, team_id: int = 0):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.role = role
        self.team_id = team_id

        agent_cfg = config.get("agent", {})
        self.view_range = agent_cfg.get("view_range", 5)

        # 역할별 스탯 초기화
        roles_cfg = config.get("roles", {})
        role_cfg = roles_cfg.get(role, {})

        self.hp = role_cfg.get("hp", agent_cfg.get("initial_hp", 100))
        self.max_hp = self.hp
        self.attack = role_cfg.get("attack", agent_cfg.get("initial_attack", 10))
        self.defense = role_cfg.get("defense", agent_cfg.get("initial_defense", 0))

        self.alive = True
        self.kills = 0
        self.items_collected = 0
        self.attack_count = 0
        self.attack_hits = 0
        self.heal_count = 0
        self.death_step = None  # 사망한 스텝 (순위 계산용)

        # 공격 쿨다운 (탱커/딜러)
        self.attack_cooldown = 0
        self.attack_cooldown_steps = role_cfg.get("attack_cooldown_steps", 0)

        # 힐 쿨다운 (힐러 전용)
        self.heal_cooldown = 0
        self.heal_cooldown_steps = role_cfg.get("heal_cooldown_steps", 0)

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def attack_range(self) -> int:
        """공격 사거리. 딜러는 원거리(3), 나머지는 근접(1)."""
        if self.role == ROLE_DEALER:
            return 3
        return 1

    @property
    def can_ranged_attack(self) -> bool:
        return self.role == ROLE_DEALER

    @property
    def can_attack(self) -> bool:
        return self.attack_cooldown == 0

    @property
    def can_heal(self) -> bool:
        return self.role == ROLE_HEALER and self.heal_cooldown == 0

    @property
    def potion_multiplier(self) -> float:
        """힐러는 포션 효과 1.5배."""
        if self.role == ROLE_HEALER:
            return 1.5
        return 1.0

    def is_teammate(self, other: 'Agent') -> bool:
        return self.team_id == other.team_id

    def take_damage(self, damage: int) -> int:
        """데미지를 받는다. 비율 감소 공식으로 방어력 적용."""
        reduction = self.defense / (self.defense + 20)
        actual_damage = max(1, int(damage * (1 - reduction)))
        self.hp -= actual_damage
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
        return actual_damage

    def heal(self, amount: int) -> int:
        """HP를 회복한다. 실제 회복량을 반환한다."""
        old_hp = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - old_hp

    def move(self, dx: int, dy: int):
        """상대 좌표만큼 이동한다."""
        self.x += dx
        self.y += dy

    def add_attack(self, bonus: int):
        self.attack += bonus

    def add_defense(self, bonus: int):
        self.defense += bonus

    def get_stats(self) -> np.ndarray:
        """정규화된 스탯 벡터를 반환한다.
        [hp, attack, defense, is_tank, is_dealer, is_healer, range_norm]
        """
        return np.array([
            self.hp / self.max_hp,
            self.attack / 30.0,
            self.defense / 20.0,
            1.0 if self.role == ROLE_TANK else 0.0,
            1.0 if self.role == ROLE_DEALER else 0.0,
            1.0 if self.role == ROLE_HEALER else 0.0,
            self.attack_range / 3.0,
        ], dtype=np.float32)

    def __repr__(self):
        return (f"Agent(id={self.agent_id}, team={self.team_id}, role={self.role}, "
                f"pos=({self.x},{self.y}), hp={self.hp}, atk={self.attack}, "
                f"def={self.defense}, alive={self.alive})")
