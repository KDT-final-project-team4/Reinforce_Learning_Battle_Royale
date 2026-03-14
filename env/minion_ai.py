"""미니언 규칙 기반 AI — BFS 경로 탐색 + 공격 우선순위"""

from __future__ import annotations

from collections import deque

import numpy as np

from env.map_generator import TILE_WALL

if False:  # TYPE_CHECKING
    from env.agent import Agent
    from env.minion import Minion
    from env.nexus import Nexus


class MinionAI:
    """미니언의 행동을 결정하는 규칙 기반 AI.

    공격 우선순위:
    1. 인접한 적 에이전트 → 공격
    2. 인접한 적 넥서스 → 공격
    3. 감지 범위 내 가장 가까운 적 에이전트로 이동
    4. 적 넥서스로 이동
    """

    def __init__(self, config: dict):
        minion_cfg = config.get("minion", {})
        self.detection_range = minion_cfg.get("detection_range", 5)

    def get_action(
        self,
        minion: Minion,
        grid: np.ndarray,
        agents: list[Agent],
        minions: list[Minion],
        enemy_nexus: Nexus,
    ) -> tuple:
        """미니언의 다음 행동을 결정한다.

        Returns:
            ("attack_agent", agent) — 인접 적 에이전트 공격
            ("attack_nexus",)       — 인접 적 넥서스 공격
            ("move", dy, dx)        — 한 칸 이동
            ("idle",)               — 이동 불가
        """
        my, mx = minion.y, minion.x

        # 1. 인접 4방향에 적 에이전트가 있으면 공격
        adj_enemy = self._find_adjacent_enemy_agent(my, mx, minion.team_id, agents)
        if adj_enemy is not None:
            return ("attack_agent", adj_enemy)

        # 2. 인접 4방향에 적 넥서스가 있으면 공격
        if enemy_nexus.alive:
            dist_to_nexus = abs(my - enemy_nexus.y) + abs(mx - enemy_nexus.x)
            if dist_to_nexus == 1:
                return ("attack_nexus",)

        # 3. 감지 범위 내 가장 가까운 적 에이전트 쪽으로 이동
        closest_enemy = self._find_closest_enemy_in_range(
            my, mx, minion.team_id, agents, self.detection_range
        )
        if closest_enemy is not None:
            step = self._bfs_next_step(
                grid, my, mx, closest_enemy.y, closest_enemy.x,
                agents, minions, minion.minion_id
            )
            if step is not None:
                return ("move", step[0] - my, step[1] - mx)

        # 4. 적 넥서스 쪽으로 이동
        if enemy_nexus.alive:
            step = self._bfs_next_step(
                grid, my, mx, enemy_nexus.y, enemy_nexus.x,
                agents, minions, minion.minion_id
            )
            if step is not None:
                return ("move", step[0] - my, step[1] - mx)

        return ("idle",)

    # ────────────────── 헬퍼 ──────────────────

    @staticmethod
    def _find_adjacent_enemy_agent(
        y: int, x: int, team_id: int, agents: list[Agent]
    ) -> Agent | None:
        """인접 4방향에서 살아있는 적 에이전트를 찾는다 (HP가 가장 낮은 우선)."""
        best = None
        best_hp = float("inf")
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ty, tx = y + dy, x + dx
            for agent in agents:
                if not agent.alive or agent.team_id == team_id:
                    continue
                if agent.y == ty and agent.x == tx:
                    if agent.hp < best_hp:
                        best_hp = agent.hp
                        best = agent
        return best

    @staticmethod
    def _find_closest_enemy_in_range(
        y: int, x: int, team_id: int, agents: list[Agent], detection_range: int
    ) -> Agent | None:
        """감지 범위 내에서 가장 가까운 살아있는 적 에이전트를 찾는다."""
        closest = None
        closest_dist = float("inf")
        for agent in agents:
            if not agent.alive or agent.team_id == team_id:
                continue
            dist = abs(y - agent.y) + abs(x - agent.x)
            if dist <= detection_range and dist < closest_dist:
                closest_dist = dist
                closest = agent
        return closest

    @staticmethod
    def _bfs_next_step(
        grid: np.ndarray,
        start_y: int, start_x: int,
        goal_y: int, goal_x: int,
        agents: list[Agent],
        minions: list[Minion],
        self_minion_id: int,
    ) -> tuple[int, int] | None:
        """BFS로 목표까지의 최단 경로 중 첫 번째 스텝을 반환한다.

        벽과 다른 유닛이 차지한 칸은 통행 불가로 처리한다.
        목표 칸 자체는 도달 가능으로 취급한다 (인접까지만 가면 됨).
        """
        h, w = grid.shape

        if start_y == goal_y and start_x == goal_x:
            return None  # 이미 도착

        # 점유 칸 집합 (자기 자신 제외)
        occupied = set()
        for a in agents:
            if a.alive:
                occupied.add((a.y, a.x))
        for m in minions:
            if m.alive and m.minion_id != self_minion_id:
                occupied.add((m.y, m.x))

        # BFS
        visited = set()
        visited.add((start_y, start_x))
        # (y, x, first_step_y, first_step_x)
        queue = deque()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = start_y + dy, start_x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if grid[ny, nx] == TILE_WALL:
                continue
            # 목표 칸은 점유되어 있어도 인접이므로 도달 가능
            if ny == goal_y and nx == goal_x:
                return (ny, nx)
            if (ny, nx) in occupied:
                continue
            if (ny, nx) not in visited:
                visited.add((ny, nx))
                queue.append((ny, nx, ny, nx))

        while queue:
            cy, cx, fy, fx = queue.popleft()

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if grid[ny, nx] == TILE_WALL:
                    continue
                if (ny, nx) in visited:
                    continue
                # 목표 도달 (인접)
                if ny == goal_y and nx == goal_x:
                    return (fy, fx)
                if (ny, nx) in occupied:
                    continue
                visited.add((ny, nx))
                queue.append((ny, nx, fy, fx))

        return None  # 경로 없음
