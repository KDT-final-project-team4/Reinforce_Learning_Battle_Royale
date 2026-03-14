"""대전 시각화 실행 스크립트 - Pygame 버전"""

import argparse
import os
import sys
import time

import yaml
from stable_baselines3 import DQN, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agent import ALL_ROLES
from env.battle_env import BattleRoyaleEnv, ROLE_ACTION_MAP
from rendering.renderer import PygameRenderer


def load_model(model_path: str):
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    try:
        return DQN.load(model_path)
    except Exception:
        return PPO.load(model_path)


def load_role_models(model_dir: str) -> dict:
    """역할별 모델을 로드한다. model_dir/{role}_final 형태."""
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

    env = BattleRoyaleEnv(config=config)

    # 멀티 정책 모드
    if args.model_dir:
        role_models = load_role_models(args.model_dir)
        if not role_models:
            print(f"Error: No role models found in {args.model_dir}")
            return

        # 상대 에이전트에 역할별 정책 설정
        env._opponent_policies = role_models
        print(f"[Multi-Policy] Loaded {len(role_models)} role models")
        model = None  # agent 0은 아래에서 역할별로 선택
    else:
        # 단일 정책 모드 (기존)
        model = load_model(args.model)
        if args.opponent_model:
            opponent = load_model(args.opponent_model)
            env.opponent_policy = opponent
            print(f"[vs AI] {args.model} vs {args.opponent_model}")
        else:
            print(f"[vs Random] {args.model} vs Random")
        role_models = None

    # Pygame 렌더러 생성
    renderer = PygameRenderer(
        map_width=config["map"]["width"],
        map_height=config["map"]["height"],
        cell_size=args.cell_size,
        panel_width=400,
        interp_frames=args.interp_frames,
    )

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    done = False
    running = True

    while not done and running:
        if role_models:
            # 멀티 정책: agent 0의 역할에 맞는 모델 사용
            agent0_role = env.agents[0].role
            if agent0_role in role_models:
                role_action, _ = role_models[agent0_role].predict(obs, deterministic=True)
                # 역할 로컬 인덱스 → 글로벌 인덱스로 변환
                role_map = ROLE_ACTION_MAP[agent0_role]
                action = role_map[int(role_action)]
            else:
                action = 8  # ACTION_STAY fallback
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        running = renderer.render(
            grid=env.grid,
            agents=env.agents,
            item_manager=env.item_manager,
            zone_manager=env.zone_manager,
            step=env.current_step,
            fps=args.fps,
            events=info.get("render_events", []),
        )

    # 결과 출력
    if running:
        print("\n=== Game Over ===")
        winning_team = info.get("winning_team", None)
        if winning_team is not None:
            print(f"Winner: Team {winning_team}!")
            for a_info in info["agents"]:
                if a_info.get("team_id") == winning_team and a_info["alive"]:
                    print(f"  Survivor: Agent {a_info['id']} ({a_info.get('role', '?')})")
        else:
            print("Draw!")
        print(f"Total Steps: {info['step']}")
        print(f"Total Reward: {total_reward:.2f}")
        # 팀별 킬 통계
        team_kills: dict[int, int] = {}
        for a_info in info["agents"]:
            tid = a_info.get("team_id", 0)
            team_kills[tid] = team_kills.get(tid, 0) + a_info["kills"]
        for tid in sorted(team_kills):
            print(f"  Team {tid} kills: {team_kills[tid]}")

        # 결과 화면 2초 유지
        time.sleep(2)

    renderer.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize battle (Pygame)")
    parser.add_argument("--model", type=str, default=None, help="Path to single-policy model")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory with role models ({role}_final) for multi-policy")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--cell-size", type=int, default=80, help="Cell pixel size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--opponent-model", type=str, default=None, help="Path to opponent model (AI vs AI)")
    parser.add_argument("--interp-frames", type=int, default=0,
                        help="Interpolation frames between steps for smooth animation (0=off, 6=recommended)")
    args = parser.parse_args()

    if not args.model and not args.model_dir:
        parser.error("Either --model or --model-dir is required")

    visualize(args)


if __name__ == "__main__":
    main()
