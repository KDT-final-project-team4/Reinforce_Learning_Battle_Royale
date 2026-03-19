"""학습 중 커스텀 콜백 — 로깅, 체크포인트, Self-Play 상대 갱신"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env.agent import ALL_ROLES


# 보상 요소 이름 목록 (TensorBoard 로깅용)
REWARD_KEYS = [
    "kill", "death", "damage_dealt", "item_pickup", "zone_damage",
    "survival", "idle", "wall_bump", "attack_miss",
    "combo", "no_combat", "approach", "flee", "disengage", "oscillation",
    "ranking", "kill_streak", "wasted_advantage",
    # Phase 4 팀/역할 보상
    "heal_ally", "ranged_miss", "invalid_action",
    "team_win", "team_eliminated", "teammate_death",
    "approach_teammate",
    "proximity",
]

# 넥서스 모드 보상 요소 이름 목록
NEXUS_REWARD_KEYS = [
    "kill", "death", "damage_dealt", "item_pickup",
    "survival", "idle", "wall_bump", "attack_miss",
    "combo", "no_combat", "approach", "flee", "oscillation",
    "heal_ally", "ranged_miss", "invalid_action",
    "teammate_death", "approach_teammate",
    "attack_cooldown", "heal_cooldown",
    # 넥서스 모드 전용
    "nexus_damage", "own_nexus_damaged",
    "nexus_destroyed_win", "nexus_destroyed_loss",
    "approach_nexus", "defend_nexus", "stay_near_own_nexus",
    "timeout_advantage", "timeout_disadvantage",
    "minion_kill",
]


class BattleRoyaleCallback(BaseCallback):
    """학습 중 에피소드 통계를 기록하고 모델을 주기적으로 저장하는 콜백."""

    def __init__(self, save_dir: str = "models",
                 save_freq: int = 50000, verbose: int = 1,
                 role: str | None = None):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.role = role  # 현재 학습 역할 (멀티 정책용)
        self._role_prefix = f"[{role.upper()}] " if role else ""
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.episode_count = 0

        # 보상 요소별 에피소드 누적 (현재 에피소드)
        self._ep_reward_sums = {k: 0.0 for k in REWARD_KEYS}
        # 최근 100 에피소드의 보상 요소별 합산
        self._reward_history = {k: [] for k in REWARD_KEYS}

    def _on_step(self) -> bool:
        # 매 스텝마다 보상 요소 누적
        infos = self.locals.get("infos", [])
        for info in infos:
            rd = info.get("reward_details", {})
            for k in REWARD_KEYS:
                self._ep_reward_sums[k] += rd.get(k, 0.0)

            # 에피소드 완료 체크 (Monitor wrapper에서 제공)
            if "episode" in info:
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # 승리 판정: 학습 에이전트의 팀이 승리했으면 승리
                winning_team = info.get("winning_team", None)
                agents_info = info.get("agents", [])
                if agents_info:
                    my_team = agents_info[0].get("team_id", 0)
                    if winning_team is not None and winning_team == my_team:
                        self.win_count += 1

                # 이 에피소드(게임 한 판) 결과 간단 출력
                if self.verbose:
                    # 팀0(학습 에이전트 팀) 기준 kill/death/heal 평균
                    team0_kills = 0
                    team0_deaths = 0
                    team0_heals = 0
                    team0_count = 0
                    if agents_info:
                        my_team = agents_info[0].get("team_id", 0)
                        for a in agents_info:
                            if a.get("team_id", 0) == my_team:
                                team0_count += 1
                                team0_kills += a.get("kills", 0)
                                team0_deaths += a.get("death_count", 0)
                                team0_heals += a.get("heal_count", 0)
                    avg_k = team0_kills / team0_count if team0_count > 0 else 0.0
                    avg_d = team0_deaths / team0_count if team0_count > 0 else 0.0
                    avg_h = team0_heals / team0_count if team0_count > 0 else 0.0

                    print(f"{self._role_prefix}Episode {self.episode_count}: "
                          f"reward={ep_reward:.2f}, steps={ep_length}, "
                          f"avgK={avg_k:.2f}, avgD={avg_d:.2f}, avgH={avg_h:.2f}, "
                          f"winning_team={winning_team}")

                # 보상 요소별 TensorBoard 기록
                for k in REWARD_KEYS:
                    val = self._ep_reward_sums[k]
                    self._reward_history[k].append(val)
                    # 개별 에피소드 값
                    self.logger.record(f"reward/{k}", val)

                # 최근 100 에피소드 평균도 기록
                if self.episode_count % 100 == 0:
                    for k in REWARD_KEYS:
                        recent = self._reward_history[k][-100:]
                        if recent:
                            self.logger.record(f"reward_avg/{k}", np.mean(recent))

                # 에피소드 누적 리셋
                self._ep_reward_sums = {k: 0.0 for k in REWARD_KEYS}

                # 100 에피소드마다 통계 출력
                if self.episode_count % 100 == 0 and self.verbose:
                    recent = self.episode_rewards[-100:]
                    win_rate = self.win_count / self.episode_count
                    print(f"\n{self._role_prefix}[Episode {self.episode_count}] "
                          f"Avg Reward: {np.mean(recent):.2f} | "
                          f"Win Rate: {win_rate:.2%} | "
                          f"Avg Length: {np.mean(self.episode_lengths[-100:]):.0f}")

                    # Phase 4 핵심 지표 출력
                    key_metrics = [
                        "kill", "death", "damage_dealt", "heal_ally",
                        "team_win", "team_eliminated", "teammate_death",
                        "invalid_action", "ranged_miss",
                    ]
                    parts = []
                    for k in key_metrics:
                        r100 = self._reward_history[k][-100:]
                        if r100:
                            avg = np.mean(r100)
                            if avg != 0:
                                parts.append(f"{k}={avg:+.2f}")
                    if parts:
                        print(f"  Rewards: {' | '.join(parts)}")

                # TensorBoard 기본 로깅
                self.logger.record("battle/episode_reward", ep_reward)
                self.logger.record("battle/episode_length", ep_length)
                if self.episode_count > 0:
                    self.logger.record("battle/win_rate",
                                       self.win_count / self.episode_count)
                # 역할 정보 로깅 (멀티 정책: TensorBoard 필터용)
                if self.role:
                    self.logger.record("battle/role", self.role)

        # 주기적 모델 저장
        if self.n_calls % self.save_freq == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"checkpoint_{self.n_calls}")
            self.model.save(path)
            if self.verbose:
                print(f"  Model saved: {path}")

        return True

    def _on_training_end(self):
        if self.verbose and self.episode_count > 0:
            print(f"\n=== Training Complete ===")
            print(f"Total Episodes: {self.episode_count}")
            print(f"Final Win Rate: {self.win_count / self.episode_count:.2%}")
            print(f"Avg Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")

            # 보상 요소별 최종 통계
            print(f"\n--- Reward Breakdown (last 100 episodes) ---")
            for k in REWARD_KEYS:
                recent = self._reward_history[k][-100:]
                if recent and np.mean(recent) != 0:
                    print(f"  {k:>15s}: {np.mean(recent):+.3f}")


class SelfPlayCallback(BaseCallback):
    """Self-Play 시 상대 정책을 주기적으로 갱신하는 콜백.

    SubprocVecEnv 호환: 모델을 임시 파일로 저장하고,
    각 subprocess 환경에서 파일 경로를 전달해 로드하도록 한다.
    """

    def __init__(self, update_interval: int = 10000,
                 opponent_path: str = "models/opponent_snapshot",
                 verbose: int = 1):
        super().__init__(verbose)
        self.update_interval = update_interval
        self.opponent_path = opponent_path

    def _on_step(self) -> bool:
        if self.n_calls % self.update_interval == 0:
            # 현재 모델을 스냅샷 파일로 저장
            os.makedirs(os.path.dirname(self.opponent_path), exist_ok=True)
            self.model.save(self.opponent_path)

            # SubprocVecEnv / DummyVecEnv 모두 env_method로 접근
            try:
                self.training_env.env_method("set_opponent_path", self.opponent_path)
                if self.verbose:
                    print(f"  [Self-Play] Opponent snapshot saved → {self.opponent_path} "
                          f"(step {self.n_calls})")
            except Exception as e:
                if self.verbose:
                    print(f"  [Self-Play] Warning: could not update opponent: {e}")
        return True


class MultiPolicySelfPlayCallback(BaseCallback):
    """멀티 정책 Self-Play: 현재 학습 역할의 스냅샷을 저장하고,
    모든 역할의 최신 스냅샷을 환경에 전달한다.
    """

    def __init__(self, learning_role: str, snapshot_dir: str = "models/multi_policy",
                 update_interval: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.learning_role = learning_role
        self.snapshot_dir = snapshot_dir
        self.update_interval = update_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.update_interval == 0:
            os.makedirs(os.path.join(self.snapshot_dir, self.learning_role), exist_ok=True)

            # 현재 역할의 스냅샷 저장
            role_path = os.path.join(self.snapshot_dir, self.learning_role, f"{self.learning_role}_snapshot")
            self.model.save(role_path)

            # 모든 역할의 최신 스냅샷 경로 수집
            paths = {}
            for role in ALL_ROLES:
                p = os.path.join(self.snapshot_dir, role, f"{role}_snapshot")
                if os.path.exists(p + ".zip"):
                    paths[role] = p

            # 환경들에 역할별 상대 정책 브로드캐스트
            try:
                self.training_env.env_method("set_opponent_paths", paths)
                if self.verbose:
                    loaded = ", ".join(paths.keys())
                    print(f"  [Multi-SP] {self.learning_role} snapshot saved "
                          f"(step {self.n_calls}), opponents: [{loaded}]")
            except Exception as e:
                if self.verbose:
                    print(f"  [Multi-SP] Warning: could not update opponents: {e}")
        return True


class NexusBattleCallback(BaseCallback):
    """넥서스 모드 학습 중 에피소드 통계를 기록하고 모델을 주기적으로 저장하는 콜백."""

    def __init__(self, save_dir: str = "models",
                 save_freq: int = 50000, verbose: int = 1,
                 role: str | None = None):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.role = role
        self._role_prefix = f"[{role.upper()}] " if role else ""
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.episode_count = 0

        self._ep_reward_sums = {k: 0.0 for k in NEXUS_REWARD_KEYS}
        self._reward_history = {k: [] for k in NEXUS_REWARD_KEYS}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            rd = info.get("reward_details", {})
            for k in NEXUS_REWARD_KEYS:
                self._ep_reward_sums[k] += rd.get(k, 0.0)

            if "episode" in info:
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # 승리 판정: 적 넥서스가 파괴됐으면 승리
                winning_team = info.get("winning_team", None)
                agents_info = info.get("agents", [])
                if agents_info:
                    my_team = agents_info[0].get("team_id", 0)
                    if winning_team is not None and winning_team == my_team:
                        self.win_count += 1

                # 보상 요소별 TensorBoard 기록
                for k in NEXUS_REWARD_KEYS:
                    val = self._ep_reward_sums[k]
                    self._reward_history[k].append(val)
                    self.logger.record(f"reward/{k}", val)

                if self.episode_count % 100 == 0:
                    for k in NEXUS_REWARD_KEYS:
                        recent = self._reward_history[k][-100:]
                        if recent:
                            self.logger.record(f"reward_avg/{k}", np.mean(recent))

                self._ep_reward_sums = {k: 0.0 for k in NEXUS_REWARD_KEYS}

                # 100 에피소드마다 통계 출력
                if self.episode_count % 100 == 0 and self.verbose:
                    recent = self.episode_rewards[-100:]
                    recent_len = self.episode_lengths[-100:]
                    win_rate = self.win_count / self.episode_count
                    print(f"\n{self._role_prefix}[Episode {self.episode_count}] "
                          f"Total steps: {self.n_calls:,} | "
                          f"Avg Reward: {np.mean(recent):.2f} | "
                          f"Win Rate: {win_rate:.2%} | "
                          f"Avg Length: {np.mean(recent_len):.0f} (min={min(recent_len):.0f} max={max(recent_len):.0f})")

                    # 넥서스 모드 핵심 지표 (전투/넥서스)
                    key_metrics = [
                        "kill", "death", "damage_dealt", "heal_ally",
                        "nexus_damage", "own_nexus_damaged",
                        "nexus_destroyed_win", "nexus_destroyed_loss",
                        "minion_kill", "defend_nexus",
                        "invalid_action",
                    ]
                    parts = []
                    for k in key_metrics:
                        r100 = self._reward_history[k][-100:]
                        if r100:
                            avg = np.mean(r100)
                            if avg != 0:
                                parts.append(f"{k}={avg:+.2f}")
                    if parts:
                        print(f"  Rewards: {' | '.join(parts)}")

                    # 접근/탐험 관련
                    approach_metrics = ["approach_nexus", "approach", "approach_teammate"]
                    approach_parts = []
                    for k in approach_metrics:
                        r100 = self._reward_history.get(k, [])[-100:]
                        if r100 and np.mean(r100) != 0:
                            approach_parts.append(f"{k}={np.mean(r100):+.2f}")
                    if approach_parts:
                        print(f"  Approach: {' | '.join(approach_parts)}")

                    # 패널티 요약
                    penalty_metrics = ["idle", "no_combat", "stay_near_own_nexus", "attack_miss", "ranged_miss", "wall_bump"]
                    penalty_parts = []
                    for k in penalty_metrics:
                        r100 = self._reward_history.get(k, [])[-100:]
                        if r100 and np.mean(r100) != 0:
                            penalty_parts.append(f"{k}={np.mean(r100):+.2f}")
                    if penalty_parts:
                        print(f"  Penalties: {' | '.join(penalty_parts)}")

                    # 넥서스 HP 정보
                    nexuses_info = info.get("nexuses", [])
                    if nexuses_info:
                        hp_parts = [f"Team{n['team_id']}={n['hp']}/{n['max_hp']}"
                                    for n in nexuses_info]
                        print(f"  Nexus HP: {' | '.join(hp_parts)}")

                    # 팀별 킬/사망 (마지막 완료 에피소드 기준)
                    if agents_info:
                        team_kills = {}
                        team_deaths = {}
                        for a in agents_info:
                            tid = a.get("team_id", 0)
                            team_kills[tid] = team_kills.get(tid, 0) + a.get("kills", 0)
                            team_deaths[tid] = team_deaths.get(tid, 0) + a.get("death_count", 0)
                        kill_parts = [f"Team{tid} K={team_kills[tid]}" for tid in sorted(team_kills)]
                        death_parts = [f"Team{tid} D={team_deaths[tid]}" for tid in sorted(team_deaths)]
                        print(f"  Kills: {' | '.join(kill_parts)}  Deaths: {' | '.join(death_parts)}")

                    # 미니언 생존 수 (마지막 에피소드 기준)
                    minions_alive = info.get("minions_alive", None)
                    if minions_alive is not None:
                        print(f"  Minions alive (end): {minions_alive}")

                self.logger.record("nexus/episode_reward", ep_reward)
                self.logger.record("nexus/episode_length", ep_length)
                if self.episode_count > 0:
                    self.logger.record("nexus/win_rate",
                                       self.win_count / self.episode_count)
                if self.role:
                    self.logger.record("nexus/role", self.role)

        # 주기적 모델 저장
        if self.n_calls % self.save_freq == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"checkpoint_{self.n_calls}")
            self.model.save(path)
            if self.verbose:
                print(f"  Model saved: {path}")

        return True

    def _on_training_end(self):
        if self.verbose and self.episode_count > 0:
            print(f"\n=== Nexus Training Complete ===")
            print(f"Total Episodes: {self.episode_count}")
            print(f"Final Win Rate: {self.win_count / self.episode_count:.2%}")
            print(f"Avg Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")

            print(f"\n--- Nexus Reward Breakdown (last 100 episodes) ---")
            for k in NEXUS_REWARD_KEYS:
                recent = self._reward_history[k][-100:]
                if recent and np.mean(recent) != 0:
                    print(f"  {k:>25s}: {np.mean(recent):+.3f}")
