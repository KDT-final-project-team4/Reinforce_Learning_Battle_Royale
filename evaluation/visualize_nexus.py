"""넥서스 모드 대전 시각화 실행 스크립트 - Pygame 버전"""

import argparse
import os
import sys
import time

import yaml
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agent import ALL_ROLES
from env.nexus_env import NexusBattleEnv
from env.base_env import ROLE_ACTION_MAP
from rendering.renderer import PygameRenderer


def load_model(model_path: str):
    """현재 프로젝트에서는 PPO만 사용하므로 PPO 전용 로더로 단순화."""
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    return PPO.load(model_path)


def load_role_models(model_dir: str) -> dict:
    """역할별 모델을 로드한다. model_dir/{role}/{role}_final 형태."""
    models = {}
    for role in ALL_ROLES:
        for suffix in ["_final", "_latest", "_snapshot"]:
            path = os.path.join(model_dir, role, f"{role}{suffix}")
            if os.path.exists(path + ".zip"):
                models[role] = load_model(path)
                print(f"  Loaded {role}: {path}")
                break
    return models


def visualize(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = NexusBattleEnv(config=config)
    # 평가/시각화는 양 팀 모두 deterministic으로 고정 (행동 성향 비교용)
    env.set_policy_deterministic(True)
    # Team1도 Team0과 같은 좌표계로 보이도록 정책 입력/출력 정규화
    env.set_policy_team_normalize(True)

    # 팀1(상대)용: --model-dir
    opponent_models = load_role_models(args.model_dir)
    if not opponent_models:
        print(f"Warning: No role models found in {args.model_dir}. Using random actions for opponents.")
    else:
        # Team 1 전체(5명) 정책 세트로 사용
        env.set_team_role_models(1, opponent_models)
        print(f"[Nexus Multi-Policy] Loaded {len(opponent_models)} opponent role models (Team 1)")

    # 팀0(에이전트 0 포함)용: --self-model-dir (없으면 opponent_models와 동일)
    if args.self_model_dir:
        self_models = load_role_models(args.self_model_dir)
        print(f"[Nexus Multi-Policy] Loaded self role models (Team 0) from {args.self_model_dir}")
    else:
        self_models = opponent_models

    # Team 0 나머지 팀원(에이전트 1~4)도 같은 세트로 구동
    if self_models:
        env.set_team_role_models(0, self_models)

    # Pygame 렌더러 생성
    # cell_size 변경 시 전체 UI 비율이 유지되도록 panel_width도 비례 조정
    panel_width = int(args.cell_size * 450 / 80)
    renderer = PygameRenderer(
        map_width=config["map"]["width"],
        map_height=config["map"]["height"],
        cell_size=args.cell_size,
        panel_width=panel_width,
        interp_frames=args.interp_frames,
    )

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    done = False
    running = True

    while not done and running:
        # 에이전트 0(팀0)의 행동: self_models 사용
        if self_models:
            agent0_role = env.agents[0].role
            if agent0_role in self_models:
                role_action, _ = self_models[agent0_role].predict(obs, deterministic=True)
                role_map = ROLE_ACTION_MAP[agent0_role]
                action = role_map[int(role_action)]
            else:
                action = 8  # ACTION_STAY fallback
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # 부활 타이머 수집
        respawn_timers = {}
        for a_info in info.get("agents", []):
            rt = a_info.get("respawn_timer", 0)
            if rt > 0:
                respawn_timers[a_info["id"]] = rt

        running = renderer.render(
            grid=env.grid,
            agents=env.agents,
            item_manager=env.item_manager,
            zone_manager=env.zone_manager,
            step=env.current_step,
            fps=args.fps,
            events=info.get("render_events", []),
            nexuses=env.nexuses,
            minions=env.minions,
            respawn_timers=respawn_timers,
        )

    # 결과 출력
    if running:
        print("\n=== Nexus Game Over ===")
        winning_team = info.get("winning_team", None)
        if winning_team is not None:
            print(f"Winner: Team {winning_team}!")
        else:
            print("Draw! (Timeout)")

        # 넥서스 HP 표시
        for n_info in info.get("nexuses", []):
            status = "DESTROYED" if not n_info.get("alive", True) else "ALIVE"
            print(f"  Nexus Team {n_info['team_id']}: {n_info['hp']}/{n_info['max_hp']} ({status})")

        print(f"Total Steps: {info['step']}")
        print(f"Total Reward: {total_reward:.2f}")

        # 에이전트별 통계
        for a_info in info.get("agents", []):
            role = a_info.get("role", "?").upper()
            deaths = a_info.get("death_count", 0)
            print(f"  A{a_info['id']} [{role}] T{a_info['team_id']}: "
                  f"K={a_info['kills']} D={deaths} HP={a_info['hp']}")

        time.sleep(2)

    renderer.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Nexus Battle (Pygame)")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/nexus_multi_policy",
        help="상대 팀(Team 1)에 사용할 역할별 모델 디렉터리 ({role}/{role}_*.zip)",
    )
    parser.add_argument(
        "--self-model-dir",
        type=str,
        default=None,
        help="내 팀(Team 0, 에이전트 0 포함)에 사용할 역할별 모델 디렉터리 (생략 시 --model-dir와 동일)",
    )
    parser.add_argument("--config", type=str, default="config/nexus_mode.yaml")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    # 기본 UI 크기 축소 (기존 비율 유지)
    parser.add_argument("--cell-size", type=int, default=60, help="Cell pixel size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--interp-frames", type=int, default=0,
                        help="Interpolation frames between steps (0=off, 6=recommended)")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
