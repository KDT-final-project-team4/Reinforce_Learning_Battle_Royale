"""Pygame 기반 배틀로얄 시각화 렌더러"""

import os
import pygame

from env.map_generator import TILE_WALL, TILE_ZONE
from env.items import ItemType

# 색상 정의
COLOR_BG = (30, 30, 30)
COLOR_EMPTY = (50, 50, 50)
COLOR_WALL = (100, 100, 100)
COLOR_ZONE = (120, 40, 120)
COLOR_GRID_LINE = (40, 40, 40)

# 팀별 색상 (팀 내 첫째/둘째 에이전트 밝기 차이)
TEAM_COLORS = {
    0: [(60, 140, 255), (100, 170, 255)],    # 파랑팀
    1: [(255, 70, 70), (255, 120, 120)],      # 빨강팀
    2: [(70, 220, 70), (130, 240, 130)],      # 초록팀
}
# 역할 없는 환경용 fallback
AGENT_COLORS = [(60, 140, 255), (255, 70, 70),
                (70, 220, 70), (255, 200, 50),
                (255, 130, 50), (180, 70, 255)]

COLOR_POTION = (50, 220, 50)     # 초록 - HP포션
COLOR_WEAPON = (255, 200, 50)    # 노랑 - 무기
COLOR_ARMOR = (100, 200, 255)    # 하늘 - 방어구

COLOR_HP_BAR_BG = (60, 60, 60)
COLOR_HP_BAR_FG = (50, 200, 50)
COLOR_HP_BAR_LOW = (220, 50, 50)

COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 140)

# 넥서스 모드 색상
COLOR_NEXUS_BLUE = (40, 100, 220)
COLOR_NEXUS_RED = (220, 40, 40)
COLOR_NEXUS_OUTLINE = (255, 215, 0)      # 금색 테두리
COLOR_MINION_BLUE = (80, 130, 200)
COLOR_MINION_RED = (200, 80, 80)
COLOR_RESPAWN_TEXT = (255, 180, 50)


def _lerp(a: float, b: float, t: float) -> float:
    """선형 보간: a에서 b까지 t(0.0~1.0) 비율로 이동한 값을 반환한다."""
    return a + (b - a) * t


class PygameRenderer:
    """Pygame을 이용한 배틀로얄 시각화 렌더러."""

    def __init__(self, map_width: int, map_height: int,
                 cell_size: int = 80, panel_width: int = 400,
                 interp_frames: int = 0):
        """
        Args:
            interp_frames: 스텝 사이 보간 프레임 수. 0이면 보간 없이 기존 방식.
                           예: 6이면 한 스텝을 6프레임에 걸쳐 부드럽게 보여준다.
        """
        self.map_width = map_width
        self.map_height = map_height
        self.cell_size = cell_size
        self.panel_width = panel_width
        self.interp_frames = interp_frames
        self.scale = cell_size / 40  # UI 비례 스케일링 계수

        self.screen_w = map_width * cell_size + panel_width
        self.screen_h = map_height * cell_size

        pygame.init()
        pygame.display.set_caption("BattleRL Arena")
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()
        sc = self.scale
        self.font = pygame.font.SysFont("consolas", int(17 * sc))
        self.font_large = pygame.font.SysFont("consolas", int(22 * sc), bold=True)
        self.font_small = pygame.font.SysFont("consolas", int(14 * sc))

        # 보간용 상태 저장
        self._prev_positions = {}   # agent_id → (x, y)
        self._prev_hps = {}         # agent_id → hp
        self._prev_alive = {}       # agent_id → alive(bool)
        self._last_agents = []      # _save_state에서 alive 스냅샷 저장용
        self._paused = False        # 스페이스바 일시정지
        self._death_positions: dict[int, tuple[int, int]] = {}  # agent_id → 마지막 (x, y)

        # 스프라이트 캐시
        self._sprites: dict[str, pygame.Surface] = {}          # 에이전트
        self._tile_sprites: dict[str, pygame.Surface] = {}     # 타일
        self._item_sprites: dict[str, pygame.Surface] = {}     # 아이템
        self._effect_sprites: dict[str, pygame.Surface] = {}   # 이펙트
        self._dead_sprite: pygame.Surface | None = None         # 사망 마커

        # 이펙트 큐: [{"sprite", "x", "y", "frames_left", "max_frames"}]
        self._active_effects: list[dict] = []

        self._load_sprites()

    # ── 팀 색상명 매핑 ────────────────────────────────────────────────────
    _TEAM_COLOR_NAME = {0: "blue", 1: "red", 2: "green"}

    def _load_sprites(self):
        """rendering/assets/ 에서 모든 스프라이트를 로드한다. 파일이 없으면 조용히 건너뛴다."""
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        cs = self.cell_size
        item_size = int(cs * 0.6)

        def _load(filename: str, size: int) -> pygame.Surface | None:
            path = os.path.join(assets_dir, filename)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, (size, size))
            return None

        # 에이전트 스프라이트
        for role in ("tank", "dealer", "healer"):
            for team_color in ("blue", "red", "green"):
                key = f"{role}_{team_color}"
                spr = _load(f"agent_{key}.png", cs)
                if spr:
                    self._sprites[key] = spr

        # 사망 마커
        self._dead_sprite = _load("agent_dead.png", cs)

        # 타일 스프라이트
        for name in ("floor", "wall", "zone"):
            spr = _load(f"tile_{name}.png", cs)
            if spr:
                self._tile_sprites[name] = spr

        # 아이템 스프라이트
        for name in ("potion", "weapon", "armor"):
            spr = _load(f"item_{name}.png", item_size)
            if spr:
                self._item_sprites[name] = spr

        # 이펙트 스프라이트
        for name in ("melee", "ranged", "hit", "heal", "death"):
            spr = _load(f"effect_{name}.png", cs)
            if spr:
                self._effect_sprites[name] = spr

    def _get_sprite(self, agent) -> "pygame.Surface | None":
        """에이전트 역할/팀에 맞는 스프라이트를 반환한다. 없으면 None."""
        role = getattr(agent, "role", None)
        team_id = getattr(agent, "team_id", None)
        if role is None or team_id is None:
            return None
        team_color = self._TEAM_COLOR_NAME.get(team_id)
        if team_color is None:
            return None
        return self._sprites.get(f"{role}_{team_color}")

    def render(self, grid, agents, item_manager, zone_manager, step, fps=10,
                events=None, nexuses=None, minions=None, respawn_timers=None):
        """한 스텝을 렌더링한다. 보간이 설정되어 있으면 여러 프레임에 걸쳐 부드럽게 표시한다.

        Args:
            nexuses: 넥서스 리스트 (넥서스 모드 전용, None이면 무시)
            minions: 미니언 리스트 (넥서스 모드 전용, None이면 무시)
            respawn_timers: {agent_id: remaining_steps} 부활 타이머 (넥서스 모드 전용)
        """
        # 새 이펙트 등록 (첫 프레임에서만)
        if events:
            self._enqueue_effects(events, agents)

        n = self.interp_frames if self.interp_frames > 0 else 1
        render_fps = fps * n if self.interp_frames > 0 else fps

        # 현재 위치/HP 수집
        curr_positions = {}
        curr_hps = {}
        for a in agents:
            curr_positions[a.agent_id] = (a.x, a.y)
            curr_hps[a.agent_id] = a.hp

        for frame_i in range(n):
            # ── 이벤트 처리 ──────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._save_state(curr_positions, curr_hps)
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self._paused = not self._paused

            # ── 일시정지 중이면 화면만 유지하고 스텝 진행 안 함 ────────
            if self._paused:
                self._draw_pause_overlay()
                pygame.display.flip()
                self.clock.tick(30)
                while self._paused:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            self._save_state(curr_positions, curr_hps)
                            return False
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                            self._paused = False
                    # 맵은 마지막 프레임 유지, 패널만 다시 그려서 스탯 가독성 확보
                    self._draw_panel(agents, step, nexuses=nexuses,
                                     respawn_timers=respawn_timers)
                    self._draw_pause_overlay()
                    pygame.display.flip()
                    self.clock.tick(30)
                continue  # 이 보간 프레임은 skip (시간 흐름 없이 재개)

            t = (frame_i + 1) / n  # 0이 아닌 1/n부터 시작, 마지막은 1.0

            self.screen.fill(COLOR_BG)

            # 1. 타일 렌더링
            self._draw_tiles(grid)

            # 1.5. 시야 오버레이 (타일 위, 아이템/에이전트 아래)
            # self._draw_view_ranges(agents)

            # 2. 아이템 렌더링
            if item_manager and item_manager.enabled:
                self._draw_items(item_manager)

            # 2.5. 넥서스 렌더링
            if nexuses:
                self._draw_nexuses(nexuses)

            # 2.6. 미니언 렌더링
            if minions:
                self._draw_minions(minions)

            # 3. 에이전트 렌더링 (보간 적용)
            if self.interp_frames > 0:
                self._draw_agents_interpolated(agents, curr_positions, curr_hps, t)
            else:
                self._draw_agents(agents)

            # 4. 이펙트 렌더링
            self._draw_effects()

            # 5. 그리드 선
            self._draw_grid_lines()

            # 6. 우측 정보 패널
            self._draw_panel(agents, step, nexuses=nexuses,
                             respawn_timers=respawn_timers)

            pygame.display.flip()
            self.clock.tick(render_fps)

        self._save_state(curr_positions, curr_hps)
        return True

    def _draw_pause_overlay(self):
        """맵 영역만 반투명 어둡게, 우측 패널은 그대로 유지한다."""
        map_px = self.map_width * self.cell_size
        overlay = pygame.Surface((map_px, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))
        pause_txt = self.font_large.render("⏸  PAUSED  (SPACE to resume)",
                                           True, (255, 220, 50))
        cx = map_px // 2 - pause_txt.get_width() // 2
        cy = self.screen_h // 2 - pause_txt.get_height() // 2
        self.screen.blit(pause_txt, (cx, cy))

    def _save_state(self, positions, hps):
        """다음 스텝 보간을 위해 현재 상태를 저장한다."""
        self._prev_positions = dict(positions)
        self._prev_hps = dict(hps)
        self._prev_alive = {a.agent_id: a.alive for a in self._last_agents}

    def _draw_view_ranges(self, agents):
        """각 에이전트의 관측 시야를 팀 색상 반투명 오버레이로 표시."""
        cs = self.cell_size
        cell_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)

        for a in agents:
            if not a.alive:
                continue
            view = getattr(a, "view_range", 9)
            half = view // 2

            # 팀 색상 + 낮은 알파
            team_colors = TEAM_COLORS.get(a.team_id, [(150, 150, 150)])
            base_color = team_colors[0]
            cell_surf.fill((*base_color, 30))

            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    gy = a.y + dy
                    gx = a.x + dx
                    if not (0 <= gy < self.map_height and 0 <= gx < self.map_width):
                        continue
                    self.screen.blit(cell_surf, (gx * cs, gy * cs))

    def _draw_tiles(self, grid):
        cs = self.cell_size
        for r in range(self.map_height):
            for c in range(self.map_width):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                tile = grid[r, c]
                if tile == TILE_WALL:
                    spr = self._tile_sprites.get("wall")
                    if spr:
                        self.screen.blit(spr, rect)
                    else:
                        pygame.draw.rect(self.screen, COLOR_WALL, rect)
                        lw = max(1, int(2 * self.scale))
                        pygame.draw.line(self.screen, (80, 80, 80),
                                         (c*cs+lw, r*cs+cs//2), (c*cs+cs-lw, r*cs+cs//2), lw)
                        pygame.draw.line(self.screen, (80, 80, 80),
                                         (c*cs+cs//2, r*cs+lw), (c*cs+cs//2, r*cs+cs-lw), lw)
                elif tile == TILE_ZONE:
                    spr = self._tile_sprites.get("zone")
                    if spr:
                        self.screen.blit(spr, rect)
                    else:
                        pygame.draw.rect(self.screen, COLOR_ZONE, rect)
                        s = pygame.Surface((cs, cs), pygame.SRCALPHA)
                        s.fill((120, 40, 120, 60))
                        self.screen.blit(s, rect)
                else:
                    spr = self._tile_sprites.get("floor")
                    if spr:
                        self.screen.blit(spr, rect)
                    else:
                        pygame.draw.rect(self.screen, COLOR_EMPTY, rect)

    def _draw_items(self, item_manager):
        cs = self.cell_size
        _ITEM_SPRITE_KEY = {ItemType.POTION: "potion", ItemType.WEAPON: "weapon", ItemType.ARMOR: "armor"}
        for r in range(self.map_height):
            for c in range(self.map_width):
                item = item_manager.item_grid[r, c]
                if item == 0:
                    continue
                cx = c * cs + cs // 2
                cy = r * cs + cs // 2

                spr_key = _ITEM_SPRITE_KEY.get(item)
                spr = self._item_sprites.get(spr_key) if spr_key else None
                if spr:
                    sx = cx - spr.get_width() // 2
                    sy = cy - spr.get_height() // 2
                    self.screen.blit(spr, (sx, sy))
                else:
                    # 폴백: 도형 렌더링
                    radius = cs // 4
                    if item == ItemType.POTION:
                        pygame.draw.circle(self.screen, COLOR_POTION, (cx, cy), radius)
                        txt = self.font.render("+", True, (255, 255, 255))
                        self.screen.blit(txt, (cx - txt.get_width()//2, cy - txt.get_height()//2))
                    elif item == ItemType.WEAPON:
                        points = [(cx, cy - radius), (cx - radius, cy + radius),
                                  (cx + radius, cy + radius)]
                        pygame.draw.polygon(self.screen, COLOR_WEAPON, points)
                    elif item == ItemType.ARMOR:
                        armor_rect = pygame.Rect(cx - radius, cy - radius,
                                                 radius * 2, radius * 2)
                        pygame.draw.rect(self.screen, COLOR_ARMOR, armor_rect)

    def _get_agent_color(self, agent):
        """에이전트의 팀/역할 기반 색상을 반환한다."""
        team_id = getattr(agent, "team_id", None)
        if team_id is not None and team_id in TEAM_COLORS:
            slot = agent.agent_id % 2  # 팀 내 0번/1번
            return TEAM_COLORS[team_id][slot]
        return AGENT_COLORS[agent.agent_id % len(AGENT_COLORS)]

    def _get_role_char(self, agent):
        """역할 첫 글자를 반환한다 (T/D/H)."""
        role = getattr(agent, "role", None)
        if role:
            return role[0].upper()
        return str(agent.agent_id)

    def _draw_agents(self, agents):
        self._last_agents = agents
        cs = self.cell_size
        for a in agents:
            # 사망 상태는 맵에서 숨기고(패널에서만 표시), 리스폰 시에만 다시 보인다.
            if not a.alive:
                continue
            self._death_positions.pop(a.agent_id, None)

            cx = a.x * cs + cs // 2
            cy = a.y * cs + cs // 2
            radius = cs // 3

            sprite = self._get_sprite(a)

            if sprite:
                sx = cx - sprite.get_width() // 2
                sy = cy - sprite.get_height() // 2
                self.screen.blit(sprite, (sx, sy))
            else:
                # 폴백: 도형 렌더링
                color = self._get_agent_color(a)
                pygame.draw.circle(self.screen, color, (cx, cy), radius)
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius, max(2, int(2 * self.scale)))
                role_char = self._get_role_char(a)
                id_txt = self.font.render(role_char, True, (255, 255, 255))
                self.screen.blit(id_txt, (cx - id_txt.get_width()//2,
                                           cy - id_txt.get_height()//2))

            # HP 바 (스프라이트/도형 공통)
            sc = self.scale
            bar_w = cs - int(8 * sc)
            bar_h = int(4 * sc)
            bar_x = a.x * cs + int(4 * sc)
            bar_y = a.y * cs + int(2 * sc)

            pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                             (bar_x, bar_y, bar_w, bar_h))
            hp_ratio = a.hp / a.max_hp
            hp_color = COLOR_HP_BAR_FG if hp_ratio > 0.3 else COLOR_HP_BAR_LOW
            pygame.draw.rect(self.screen, hp_color,
                             (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def _draw_agents_interpolated(self, agents, curr_positions, curr_hps, t):
        """보간된 위치와 HP로 에이전트를 그린다."""
        self._last_agents = agents
        cs = self.cell_size
        for a in agents:
            # 사망 상태는 맵에서 숨기고(패널에서만 표시), 리스폰 시에만 다시 보인다.
            if not a.alive:
                continue
            self._death_positions.pop(a.agent_id, None)

            aid = a.agent_id
            curr_x, curr_y = curr_positions[aid]
            was_dead = self._prev_alive.get(aid, False) is False and aid in self._prev_alive

            # 이전 위치가 있으면 보간, 없으면 (첫 프레임) 현재 위치 사용
            if aid in self._prev_positions and not was_dead:
                prev_x, prev_y = self._prev_positions[aid]
                draw_x = _lerp(prev_x, curr_x, t)
                draw_y = _lerp(prev_y, curr_y, t)
            else:
                draw_x, draw_y = float(curr_x), float(curr_y)

            # HP 보간
            if aid in self._prev_hps and not was_dead:
                draw_hp = _lerp(self._prev_hps[aid], curr_hps[aid], t)
            else:
                draw_hp = float(curr_hps[aid])

            # 픽셀 좌표 계산 (float 기반)
            cx = int(draw_x * cs + cs / 2)
            cy = int(draw_y * cs + cs / 2)
            radius = cs // 3

            sprite = self._get_sprite(a)

            if sprite:
                sx = cx - sprite.get_width() // 2
                sy = cy - sprite.get_height() // 2
                self.screen.blit(sprite, (sx, sy))
            else:
                color = self._get_agent_color(a)
                pygame.draw.circle(self.screen, color, (cx, cy), radius)
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius, max(2, int(2 * self.scale)))
                role_char = self._get_role_char(a)
                id_txt = self.font.render(role_char, True, (255, 255, 255))
                self.screen.blit(id_txt, (cx - id_txt.get_width() // 2,
                                           cy - id_txt.get_height() // 2))

            # HP 바 (스프라이트/도형 공통, 보간된 위치 기준)
            sc = self.scale
            bar_w = cs - int(8 * sc)
            bar_h = int(4 * sc)
            bar_x = int(draw_x * cs) + int(4 * sc)
            bar_y = int(draw_y * cs) + int(2 * sc)

            pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                             (bar_x, bar_y, bar_w, bar_h))
            hp_ratio = draw_hp / a.max_hp
            hp_color = COLOR_HP_BAR_FG if hp_ratio > 0.3 else COLOR_HP_BAR_LOW
            pygame.draw.rect(self.screen, hp_color,
                             (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def _draw_nexuses(self, nexuses):
        """넥서스를 큰 색상 사각형 + HP 바로 렌더링한다."""
        cs = self.cell_size
        sc = self.scale
        for nexus in nexuses:
            if not nexus.alive:
                continue
            px_x = nexus.x * cs
            px_y = nexus.y * cs
            color = COLOR_NEXUS_BLUE if nexus.team_id == 0 else COLOR_NEXUS_RED

            # 넥서스 사각형 (셀 크기의 80%)
            margin = int(cs * 0.1)
            nex_rect = pygame.Rect(px_x + margin, px_y + margin,
                                   cs - 2 * margin, cs - 2 * margin)
            pygame.draw.rect(self.screen, color, nex_rect)
            pygame.draw.rect(self.screen, COLOR_NEXUS_OUTLINE, nex_rect,
                             max(2, int(2 * sc)))

            # "N" 글자
            n_txt = self.font_large.render("N", True, (255, 255, 255))
            self.screen.blit(n_txt, (px_x + cs // 2 - n_txt.get_width() // 2,
                                      px_y + cs // 2 - n_txt.get_height() // 2))

            # HP 바 (넥서스 아래쪽)
            bar_w = cs - int(8 * sc)
            bar_h = int(5 * sc)
            bar_x = px_x + int(4 * sc)
            bar_y = px_y + cs - int(8 * sc)
            pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                             (bar_x, bar_y, bar_w, bar_h))
            hp_ratio = nexus.hp / nexus.max_hp
            hp_color = COLOR_HP_BAR_FG if hp_ratio > 0.3 else COLOR_HP_BAR_LOW
            pygame.draw.rect(self.screen, hp_color,
                             (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def _draw_minions(self, minions):
        """미니언을 작은 삼각형으로 렌더링한다."""
        cs = self.cell_size
        sc = self.scale
        for m in minions:
            if not m.alive:
                continue
            cx = m.x * cs + cs // 2
            cy = m.y * cs + cs // 2
            color = COLOR_MINION_BLUE if m.team_id == 0 else COLOR_MINION_RED
            size = int(cs * 0.25)

            # 삼각형 (위로 향하는)
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (255, 255, 255), points,
                                max(1, int(1 * sc)))

            # 미니언 HP 바 (작게)
            bar_w = int(cs * 0.5)
            bar_h = int(2 * sc)
            bar_x = cx - bar_w // 2
            bar_y = m.y * cs + int(2 * sc)
            pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                             (bar_x, bar_y, bar_w, bar_h))
            hp_ratio = m.hp / m.max_hp if m.max_hp > 0 else 0
            hp_color = COLOR_HP_BAR_FG if hp_ratio > 0.3 else COLOR_HP_BAR_LOW
            pygame.draw.rect(self.screen, hp_color,
                             (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def _draw_grid_lines(self):
        # 타일 스프라이트가 로드되어 있으면 그리드 선 생략 (스프라이트가 셀 경계를 자체 제공)
        if self._tile_sprites:
            return
        cs = self.cell_size
        map_h_px = self.map_height * cs
        map_w_px = self.map_width * cs
        for r in range(self.map_height + 1):
            pygame.draw.line(self.screen, COLOR_GRID_LINE,
                             (0, r * cs), (map_w_px, r * cs))
        for c in range(self.map_width + 1):
            pygame.draw.line(self.screen, COLOR_GRID_LINE,
                             (c * cs, 0), (c * cs, map_h_px))

    def _draw_panel(self, agents, step, nexuses=None, respawn_timers=None):
        px = self.map_width * self.cell_size  # 패널 시작 X
        pw = self.panel_width
        ph = self.screen_h
        sc = self.scale
        respawn_timers = respawn_timers or {}

        # 패널 배경 + 구분선
        pygame.draw.rect(self.screen, (18, 18, 18), (px, 0, pw, ph))
        pygame.draw.line(self.screen, (80, 80, 80), (px, 0), (px, ph), 2)

        # ── 헤더: Step ──────────────────────────────────────────
        step_txt = self.font_large.render(f"Step: {step}", True, COLOR_TEXT)
        self.screen.blit(step_txt, (px + int(10 * sc), int(8 * sc)))
        hdr_y = int(32 * sc)
        pygame.draw.line(self.screen, (55, 55, 55),
                         (px + int(4 * sc), hdr_y), (px + pw - int(4 * sc), hdr_y), 1)

        y = int(40 * sc)
        lx = px + int(8 * sc)   # 텍스트 left x
        bar_x = px + int(8 * sc)
        bar_w = pw - int(16 * sc)

        # ── 넥서스 HP 바 (넥서스 모드) ──────────────────────────
        if nexuses:
            for nexus in nexuses:
                n_color = COLOR_NEXUS_BLUE if nexus.team_id == 0 else COLOR_NEXUS_RED
                status = "" if nexus.alive else " DESTROYED"
                n_txt = self.font.render(
                    f"Nexus T{nexus.team_id}: {nexus.hp}/{nexus.max_hp}{status}",
                    True, n_color)
                self.screen.blit(n_txt, (lx, y))
                y += int(18 * sc)

                # HP 바
                n_bar_h = int(8 * sc)
                pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                                 (bar_x, y, bar_w, n_bar_h))
                if nexus.alive and nexus.max_hp > 0:
                    hp_ratio = nexus.hp / nexus.max_hp
                    hp_color = n_color if hp_ratio > 0.3 else COLOR_HP_BAR_LOW
                    pygame.draw.rect(self.screen, hp_color,
                                     (bar_x, y, int(bar_w * hp_ratio), n_bar_h))
                y += int(14 * sc)

            y += int(6 * sc)
            pygame.draw.line(self.screen, (55, 55, 55),
                             (px + int(4 * sc), y), (px + pw - int(4 * sc), y), 1)
            y += int(6 * sc)

        # ── 팀별 에이전트 카드 ───────────────────────────────────
        teams: dict[int, list] = {}
        for a in agents:
            tid = getattr(a, "team_id", 0)
            teams.setdefault(tid, []).append(a)

        for tid in sorted(teams.keys()):
            team_color = TEAM_COLORS.get(tid, [(200, 200, 200)])[0]
            team_hdr = self.font_large.render(f"▶ Team {tid}", True, team_color)
            self.screen.blit(team_hdr, (lx, y))
            y += int(26 * sc)

            for a in teams[tid]:
                role_name = getattr(a, "role", "?").upper()
                dead = not a.alive

                # 사망 시: 색상을 어둡게, 생존 시: 팀 색상
                respawn_t = respawn_timers.get(a.agent_id, 0)
                if dead and respawn_t > 0:
                    name_color = COLOR_RESPAWN_TEXT
                    stat_color = (120, 100, 50)
                    dead_tag = f"  Respawn:{respawn_t}"
                elif dead:
                    name_color = (100, 100, 100)
                    stat_color = (75, 75, 75)
                    dead_tag = "  ✕ DEAD"
                else:
                    name_color = self._get_agent_color(a)
                    stat_color = COLOR_TEXT_DIM
                    dead_tag = ""

                # 줄1: 역할 + ID + HP 수치
                l1 = f"  [{role_name}] A{a.agent_id}{dead_tag}   HP {a.hp}/{a.max_hp}"
                self.screen.blit(self.font.render(l1, True, name_color), (lx, y))
                y += int(20 * sc)

                # HP 바 (사망 시 빈 바)
                panel_bar_h = int(9 * sc)
                pygame.draw.rect(self.screen, COLOR_HP_BAR_BG,
                                 (bar_x, y, bar_w, panel_bar_h))
                if not dead:
                    hp_ratio = max(0.0, a.hp / a.max_hp)
                    hp_color = COLOR_HP_BAR_FG if hp_ratio > 0.3 else COLOR_HP_BAR_LOW
                    pygame.draw.rect(self.screen, hp_color,
                                     (bar_x, y, int(bar_w * hp_ratio), panel_bar_h))
                y += int(14 * sc)

                # 줄2: ATK / DEF / 킬
                l2 = f"  ATK:{a.attack:3d}  DEF:{a.defense:3d}  K:{a.kills}"
                self.screen.blit(
                    self.font_small.render(l2, True, stat_color), (lx, y))
                y += int(18 * sc)

                # 줄3: 공격 성공/전체 (정확도)
                atk_total = getattr(a, "attack_count", 0)
                atk_hits  = getattr(a, "attack_hits",  0)
                if atk_total > 0:
                    acc = atk_hits / atk_total * 100
                    l3 = f"  Atk {atk_hits}/{atk_total} ({acc:.0f}%)"
                else:
                    l3 = "  Atk 0/0  (--%)"
                self.screen.blit(
                    self.font_small.render(l3, True, stat_color), (lx, y))
                y += int(18 * sc)

                # 줄4: 힐러 전용 — 힐 횟수
                if getattr(a, "role", None) == "healer":
                    heal_cnt = getattr(a, "heal_count", 0)
                    heal_color = (50, 130, 80) if dead else (80, 220, 130)
                    l4 = f"  Heals: {heal_cnt}"
                    self.screen.blit(
                        self.font_small.render(l4, True, heal_color), (lx, y))
                    y += int(18 * sc)

                # 줄5: 아이템 수집
                items = getattr(a, "items_collected", 0)
                if items > 0:
                    item_color = (120, 95, 40) if dead else (255, 200, 80)
                    l5 = f"  Items: {items}"
                    self.screen.blit(
                        self.font_small.render(l5, True, item_color), (lx, y))
                    y += int(18 * sc)

                y += int(8 * sc)  # 에이전트 간 여백

            y += int(10 * sc)  # 팀 간 여백
            pygame.draw.line(self.screen, (45, 45, 45),
                             (px + int(4 * sc), y - int(5 * sc)), (px + pw - int(4 * sc), y - int(5 * sc)), 1)

    # ── 이펙트 시스템 ──────────────────────────────────────────────────

    def _enqueue_effects(self, events: list[dict], agents):
        """렌더링 이벤트를 이펙트 큐에 등록한다."""
        n = max(self.interp_frames, 3)  # 이펙트 지속 프레임 수
        for ev in events:
            etype = ev.get("type")

            if etype == "melee_hit":
                ax, ay = ev.get("attacker_x", 0), ev.get("attacker_y", 0)
                tx, ty = ev.get("target_x", 0), ev.get("target_y", 0)
                mid_x = (ax + tx) / 2.0
                mid_y = (ay + ty) / 2.0
                # melee(슬래시): 공격자·대상 중간 지점
                melee_spr = self._effect_sprites.get("melee")
                if melee_spr:
                    self._active_effects.append({
                        "sprite": melee_spr, "x": mid_x, "y": mid_y,
                        "frames_left": n, "max_frames": n,
                    })
                # hit(버스트): 피격 에이전트 위치
                hit_spr = self._effect_sprites.get("hit")
                if hit_spr:
                    self._active_effects.append({
                        "sprite": hit_spr, "x": tx, "y": ty,
                        "frames_left": n, "max_frames": n,
                    })

            elif etype == "ranged_hit":
                # 화살이 공격자 → 대상으로 이동하는 투사체 이펙트
                ax, ay = ev.get("attacker_x", 0), ev.get("attacker_y", 0)
                tx, ty = ev.get("target_x", 0), ev.get("target_y", 0)
                arrow_spr = self._effect_sprites.get("ranged")
                hit_spr = self._effect_sprites.get("hit")
                if arrow_spr:
                    self._active_effects.append({
                        "sprite": arrow_spr,
                        "start_x": ax, "start_y": ay,
                        "end_x": tx, "end_y": ty,
                        "x": ax, "y": ay,
                        "projectile": True,
                        "frames_left": n, "max_frames": n,
                    })
                if hit_spr:
                    # 피격 이펙트는 투사체 도달 후 표시
                    self._active_effects.append({
                        "sprite": hit_spr, "x": tx, "y": ty,
                        "delay": n // 2,
                        "frames_left": n, "max_frames": n,
                    })

            elif etype == "heal":
                # 힐러와 대상 사이 중간 지점에 표시
                hx, hy = ev.get("healer_x", 0), ev.get("healer_y", 0)
                tx, ty = ev.get("target_x", 0), ev.get("target_y", 0)
                mid_x = (hx + tx) / 2.0
                mid_y = (hy + ty) / 2.0
                spr = self._effect_sprites.get("heal")
                if spr:
                    self._active_effects.append({
                        "sprite": spr, "x": mid_x, "y": mid_y,
                        "frames_left": n, "max_frames": n,
                    })

            elif etype == "death":
                tx, ty = ev.get("x", 0), ev.get("y", 0)
                spr = self._effect_sprites.get("death")
                if spr:
                    self._active_effects.append({
                        "sprite": spr, "x": tx, "y": ty,
                        "frames_left": n, "max_frames": n,
                    })

    def _draw_effects(self):
        """활성 이펙트를 알파 페이드와 함께 렌더링한다."""
        cs = self.cell_size
        still_active = []
        for eff in self._active_effects:
            if eff["frames_left"] <= 0:
                continue

            # 딜레이가 있으면 대기
            delay = eff.get("delay", 0)
            if delay > 0:
                eff["delay"] = delay - 1
                still_active.append(eff)
                continue

            # 투사체 이펙트: 매 프레임 위치 보간 (시작 → 끝)
            if eff.get("projectile"):
                progress = 1.0 - (eff["frames_left"] / eff["max_frames"])
                eff["x"] = _lerp(eff["start_x"], eff["end_x"], progress)
                eff["y"] = _lerp(eff["start_y"], eff["end_y"], progress)

            # 알파 계산: 시작 255 → 0으로 선형 감소
            alpha = int(255 * eff["frames_left"] / eff["max_frames"])
            spr = eff["sprite"].copy()
            spr.set_alpha(alpha)
            px_x = int(eff["x"] * cs)
            px_y = int(eff["y"] * cs)
            self.screen.blit(spr, (px_x, px_y))
            eff["frames_left"] -= 1
            if eff["frames_left"] > 0:
                still_active.append(eff)
        self._active_effects = still_active

    def close(self):
        pygame.quit()
