"""아이템 정의 모듈 — 무기, 방어구, 포션

Phase 1에서는 아이템이 비활성화 상태. Phase 2에서 활성화한다.
"""

from enum import IntEnum

import numpy as np


class ItemType(IntEnum):
    NONE = 0
    POTION = 1   # HP 포션
    WEAPON = 2   # 무기 (공격력 증가)
    ARMOR = 3    # 방어구 (방어력 증가)


class ItemManager:
    """맵 위의 아이템 배치 및 리젠을 관리한다."""

    def __init__(self, config: dict, grid_shape: tuple[int, int],
                 rng: np.random.Generator | None = None):
        self.enabled = config.get("items", {}).get("enabled", False)
        self.cfg = config.get("items", {})
        self.grid_shape = grid_shape
        self.rng = rng or np.random.default_rng()

        # 아이템 맵: 0이면 아이템 없음, ItemType 값이면 해당 아이템 존재
        self.item_grid = np.zeros(grid_shape, dtype=np.int32)
        self.item_age_grid = np.zeros(grid_shape, dtype=np.int32)
        self.expire_steps = self.cfg.get("item_expire_steps", 0)  # 0 = 만료 없음
        self.steps_since_spawn = 0

    def reset(self, grid: np.ndarray):
        """아이템 맵을 초기화한다."""
        self.item_grid = np.zeros(self.grid_shape, dtype=np.int32)
        self.item_age_grid = np.zeros(self.grid_shape, dtype=np.int32)
        self.steps_since_spawn = 0

    def step(self, grid: np.ndarray):
        """매 스텝마다 호출. 아이템 만료/자기장 정리 후 리젠 주기에 따라 생성."""
        if not self.enabled:
            return

        from env.map_generator import TILE_ZONE

        # 기존 아이템 나이 증가
        has_items = self.item_grid > 0
        self.item_age_grid[has_items] += 1

        # 만료된 아이템 제거
        if self.expire_steps > 0:
            expired = has_items & (self.item_age_grid >= self.expire_steps)
            self.item_grid[expired] = 0
            self.item_age_grid[expired] = 0

        # 자기장에 덮인 아이템 제거
        in_zone = (grid == TILE_ZONE) & (self.item_grid > 0)
        self.item_grid[in_zone] = 0
        self.item_age_grid[in_zone] = 0

        self.steps_since_spawn += 1
        interval = self.cfg.get("respawn_interval", 20)
        if self.steps_since_spawn >= interval:
            self.steps_since_spawn = 0
            self._spawn_items(grid)

    def pickup(self, y: int, x: int) -> ItemType:
        """(y, x) 위치의 아이템을 획득한다. 아이템 타입을 반환한다."""
        item = ItemType(self.item_grid[y, x])
        if item != ItemType.NONE:
            self.item_grid[y, x] = 0
            self.item_age_grid[y, x] = 0
        return item

    def get_item_at(self, y: int, x: int) -> ItemType:
        """(y, x) 위치의 아이템 타입을 반환한다."""
        return ItemType(self.item_grid[y, x])

    def _spawn_items(self, grid: np.ndarray):
        """빈 칸에 아이템을 스폰한다."""
        from env.map_generator import TILE_EMPTY

        empty = list(zip(*np.where(
            (grid == TILE_EMPTY) & (self.item_grid == 0)
        )))
        if not empty:
            return

        self.rng.shuffle(empty)
        idx = 0

        # 포션 스폰
        current_potions = np.sum(self.item_grid == ItemType.POTION)
        max_potions = self.cfg.get("max_potions_on_map", 3)
        while current_potions < max_potions and idx < len(empty):
            r, c = empty[idx]
            self.item_grid[r, c] = ItemType.POTION
            current_potions += 1
            idx += 1

        # 무기 스폰
        current_weapons = np.sum(self.item_grid == ItemType.WEAPON)
        max_weapons = self.cfg.get("max_weapons_on_map", 2)
        while current_weapons < max_weapons and idx < len(empty):
            r, c = empty[idx]
            self.item_grid[r, c] = ItemType.WEAPON
            current_weapons += 1
            idx += 1

        # 방어구 스폰
        current_armors = np.sum(self.item_grid == ItemType.ARMOR)
        max_armors = self.cfg.get("max_armors_on_map", 2)
        while current_armors < max_armors and idx < len(empty):
            r, c = empty[idx]
            self.item_grid[r, c] = ItemType.ARMOR
            current_armors += 1
            idx += 1
