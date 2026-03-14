"""넥서스 모드 대전 시각화 실행 스크립트 - Pygame 버전"""

import argparse
import os
import sys
import time

import yaml
from stable_baselines3 import DQN, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agent import ALL_ROLES
from env.nexus_env import NexusBattleEnv
from env.base_env import ROLE_ACTION_MAP
from rendering.renderer import PygameRenderer


def load_model(model_path: str):
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    try:
        return DQN.load(model_path)
    except Exception:
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

    # 멀티 정책 모드
    role_models = load_role_models(args.model_dir)
    if not role_models:
        print(f"Warning: No role models found in {args.model_dir}. Using random actions.")

    # 상대 에이전트에 역할별 정책 설정
    if role_models:
        env._opponent_policies = role_models
        print(f"[Nexus Multi-Policy] Loaded {len(role_models)} role models")

    # Pygame 렌더러 생성
    renderer = PygameRenderer(
        map_width=config["map"]["width"],
        map_height=config["map"]["height"],
        cell_size=args.cell_size,
        panel_width=450,
        interp_frames=args.interp_frames,
    )

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    done = False
    running = True

    while not done and running:
        if role_models:
            agent0_role = env.agents[0].role
            if agent0_role in role_models:
                role_action, _ = role_models[agent0_role].predict(obs, deterministic=True)
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
    parser.add_argument("--model-dir", type=str, default="models/nexus_multi_policy",
                        help="Directory with role models ({role}/{role}_final)")
    parser.add_argument("--config", type=str, default="config/nexus_mode.yaml")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--cell-size", type=int, default=80, help="Cell pixel size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--interp-frames", type=int, default=0,
                        help="Interpolation frames between steps (0=off, 6=recommended)")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
