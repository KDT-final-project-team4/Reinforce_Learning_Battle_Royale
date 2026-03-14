"""자기장(독가스) 축소 로직

Phase 1에서는 비활성화 상태. Phase 2에서 활성화한다.
"""

import numpy as np

from env.map_generator import TILE_ZONE


class ZoneManager:
    """맵 외곽부터 독가스를 축소하는 메커니즘을 관리한다."""

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        self.enabled = config.get("zone", {}).get("enabled", False)
        self.cfg = config.get("zone", {})
        self.grid_shape = grid_shape

        self.shrink_start = self.cfg.get("shrink_start_step", 100)
        self.shrink_interval = self.cfg.get("shrink_interval", 30)
        self.damage_per_step = self.cfg.get("damage_per_step", 5)

        # 안전지대 최소 반경: 이 이상으로 축소 불가 (기본값: 맵 단변의 1/5, 최소 3)
        h, w = grid_shape
        default_min = max(3, min(h, w) // 5)
        self.min_safe_radius = self.cfg.get("min_safe_radius", default_min)

        self.current_shrink_level = 0  # 외곽에서 몇 줄까지 독가스화됐는지

    def reset(self):
        self.current_shrink_level = 0

    def step(self, grid: np.ndarray, current_step: int) -> bool:
        """매 스텝 호출. 독가스 축소가 발생했으면 True 반환."""
        if not self.enabled:
            return False

        if current_step < self.shrink_start:
            return False

        steps_since_start = current_step - self.shrink_start
        target_level = (steps_since_start // self.shrink_interval) + 1

        if target_level > self.current_shrink_level:
            self.current_shrink_level = target_level
            self._apply_zone(grid)
            return True
        return False

    def is_in_zone(self, grid: np.ndarray, y: int, x: int) -> bool:
        """해당 위치가 독가스 영역인지 확인한다."""
        return grid[y, x] == TILE_ZONE

    def _apply_zone(self, grid: np.ndarray):
        """현재 shrink_level에 따라 외곽을 독가스로 전환한다."""
        h, w = self.grid_shape
        level = self.current_shrink_level

        # 최대 축소 범위 제한 (안전지대 최소 반경 보장)
        max_level = min(h, w) // 2 - self.min_safe_radius
        level = min(level, max(0, max_level))

        for l in range(level):
            # 상단 행
            grid[l, :] = TILE_ZONE
            # 하단 행
            grid[h - 1 - l, :] = TILE_ZONE
            # 좌측 열
            grid[:, l] = TILE_ZONE
            # 우측 열
            grid[:, w - 1 - l] = TILE_ZONE
