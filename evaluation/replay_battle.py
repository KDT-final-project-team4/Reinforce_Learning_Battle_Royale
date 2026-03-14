"""학습된 AI의 실제 행동을 턴 단위로 리플레이하는 스크립트

각 턴마다 맵 상태, AI의 선택한 행동, 에이전트 상태를 출력한다.
"""

import os
import sys

import numpy as np
import yaml
from stable_baselines3 import DQN, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.battle_env import (
    BattleRoyaleEnv,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_ATTACK, ACTION_STAY,
    TILE_WALL, TILE_ZONE,
)
from env.items import ItemType

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_ATTACK: "ATTACK",
    ACTION_STAY: "STAY",
}


def load_model(model_path, env):
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    try:
        return DQN.load(model_path, env=env)
    except Exception:
        return PPO.load(model_path, env=env)


def render_map(env):
    """맵을 보기 좋게 렌더링한다."""
    grid = env.grid
    h, w = grid.shape
    lines = []

    # 상단 테두리
    lines.append("+" + "--" * w + "+")

    for r in range(h):
        row = "|"
        for c in range(w):
            # 에이전트 체크
            agent_here = None
            for a in env.agents:
                if a.alive and a.y == r and a.x == c:
                    agent_here = a
                    break

            if agent_here is not None:
                if agent_here.agent_id == 0:
                    row += "\033[94mA0\033[0m"  # 파란색 - 학습된 AI
                else:
                    row += "\033[91mA1\033[0m"  # 빨간색 - 상대
            elif grid[r, c] == TILE_WALL:
                row += "\033[90m##\033[0m"  # 회색 - 벽
            elif grid[r, c] == TILE_ZONE:
                row += "\033[35m~~\033[0m"  # 보라색 - 독가스
            else:
                # 아이템 체크
                if env.item_manager and env.item_manager.enabled:
                    item = env.item_manager.item_grid[r, c]
                    if item == ItemType.POTION:
                        row += "\033[92m P\033[0m"
                    elif item == ItemType.WEAPON:
                        row += "\033[93m W\033[0m"
                    elif item == ItemType.ARMOR:
                        row += "\033[96m A\033[0m"
                    else:
                        row += " ."
                else:
                    row += " ."
        row += "|"
        lines.append(row)

    # 하단 테두리
    lines.append("+" + "--" * w + "+")

    return "\n".join(lines)


def print_status(env, step, action):
    """현재 상태를 출력한다."""
    print(f"\n{'='*40}")
    print(f"  Step {step:3d}  |  AI Action: {ACTION_NAMES.get(action, '?'):>6s}")
    print(f"{'='*40}")
    print(render_map(env))
    print()

    for a in env.agents:
        if a.agent_id == 0:
            label = "\033[94m[AI]\033[0m "
        else:
            label = "\033[91m[Enemy]\033[0m "

        if a.alive:
            hp_bar_len = 20
            hp_filled = int(a.hp / 100 * hp_bar_len)
            hp_bar = "\033[92m" + "=" * hp_filled + "\033[90m" + "-" * (hp_bar_len - hp_filled) + "\033[0m"
            print(f"  {label} HP [{hp_bar}] {a.hp:3d}/100  ATK:{a.attack:2d}  DEF:{a.defense:2d}  Kills:{a.kills}")
        else:
            print(f"  {label} \033[90mDEAD\033[0m  (Kills: {a.kills})")


def replay(model_path, config_path="config/default.yaml", seed=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = BattleRoyaleEnv(config=config)
    model = load_model(model_path, env)

    if seed is None:
        seed = np.random.randint(0, 10000)

    obs, info = env.reset(seed=seed)
    print(f"\n{'#'*40}")
    print(f"  BattleRL Arena - AI Replay")
    print(f"  Seed: {seed}")
    print(f"{'#'*40}")

    print_status(env, 0, -1)
    input("  [Enter] to start...")

    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

        # 화면 클리어
        os.system("cls" if os.name == "nt" else "clear")

        print_status(env, step, action)

        if reward != 0:
            color = "\033[92m" if reward > 0 else "\033[91m"
            print(f"  {color}Reward: {reward:+.2f}\033[0m  (Total: {total_reward:.2f})")

        if done:
            break

        input("  [Enter] for next step...")

    # 결과
    print(f"\n{'#'*40}")
    print(f"  GAME OVER!")
    print(f"{'#'*40}")

    winner = None
    for a in env.agents:
        if a.alive:
            winner = a.agent_id

    if winner == 0:
        print(f"  \033[92mAI WINS!\033[0m")
    elif winner is not None:
        print(f"  \033[91mENEMY WINS!\033[0m")
    else:
        print(f"  \033[93mDRAW!\033[0m")

    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print()

    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/dqn_final")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    replay(args.model, args.config, args.seed)
