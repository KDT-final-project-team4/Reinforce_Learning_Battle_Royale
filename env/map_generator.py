"""맵 생성 모듈 — 벽, 아이템 배치"""

import numpy as np

# 타일 타입
TILE_EMPTY = 0
TILE_WALL = 1
TILE_ZONE = 2  # 독가스 (Phase 2)


def generate_map(width: int, height: int, wall_count: int,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """그리드 맵을 생성한다.

    Args:
        width: 맵 가로 크기
        height: 맵 세로 크기
        wall_count: 배치할 벽 수
        rng: 난수 생성기 (재현성 위해)

    Returns:
        (height, width) 크기의 정수 배열. 0=빈공간, 1=벽.
    """
    if rng is None:
        rng = np.random.default_rng()

    grid = np.zeros((height, width), dtype=np.int32)

    # 빈 공간 좌표 목록에서 랜덤 선택하여 벽 배치
    empty_positions = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(empty_positions)

    placed = 0
    for r, c in empty_positions:
        if placed >= wall_count:
            break
        grid[r, c] = TILE_WALL
        placed += 1

    return grid


def get_empty_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    """맵에서 빈 공간(값 0)의 좌표 리스트를 반환한다."""
    rows, cols = np.where(grid == TILE_EMPTY)
    return list(zip(rows.tolist(), cols.tolist()))


def place_agents_near(grid: np.ndarray, anchor_y: int, anchor_x: int,
                      count: int, radius: int,
                      rng: np.random.Generator | None = None) -> list[tuple[int, int]]:
    """앵커 좌표 근처에 에이전트를 배치한다.

    Args:
        anchor_y, anchor_x: 중심 좌표 (넥서스 위치 등)
        count: 배치할 수
        radius: 앵커로부터 맨해튼 거리 최대값

    Returns:
        [(y, x), ...] 에이전트 위치 리스트
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = grid.shape
    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dy) + abs(dx) > radius:
                continue
            ny, nx = anchor_y + dy, anchor_x + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == TILE_EMPTY:
                # 앵커 칸 자체는 제외 (넥서스가 차지)
                if ny == anchor_y and nx == anchor_x:
                    continue
                candidates.append((ny, nx))

    if len(candidates) < count:
        raise ValueError(
            f"앵커({anchor_y},{anchor_x}) 반경 {radius} 내 빈 공간({len(candidates)})이 "
            f"에이전트 수({count})보다 부족합니다."
        )

    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[i] for i in indices]


def place_agents(grid: np.ndarray, num_agents: int,
                 rng: np.random.Generator | None = None) -> list[tuple[int, int]]:
    """에이전트들의 시작 위치를 랜덤으로 결정한다.

    Returns:
        [(y, x), ...] 형태의 에이전트 시작 위치 리스트
    """
    if rng is None:
        rng = np.random.default_rng()

    empty = get_empty_positions(grid)
    if len(empty) < num_agents:
        raise ValueError(
            f"빈 공간({len(empty)}개)이 에이전트 수({num_agents})보다 부족합니다."
        )

    indices = rng.choice(len(empty), size=num_agents, replace=False)
    return [empty[i] for i in indices]
