"""학습된 AI의 한 판을 자동으로 기록하여 출력하는 스크립트"""

import argparse
import os
import sys

import numpy as np
import yaml
from stable_baselines3 import DQN, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.battle_env import (
    BattleRoyaleEnv,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_ATTACK, ACTION_STAY,
    TILE_WALL,
)

ACTION_NAMES = {
    ACTION_UP: "UP   ", ACTION_DOWN: "DOWN ", ACTION_LEFT: "LEFT ",
    ACTION_RIGHT: "RIGHT", ACTION_ATTACK: "ATK  ", ACTION_STAY: "STAY ",
}


def load_model(model_path, env):
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    try:
        return DQN.load(model_path, env=env)
    except Exception:
        return PPO.load(model_path, env=env)


def render_compact(env):
    grid = env.grid
    h, w = grid.shape
    rows = []
    for r in range(h):
        row = ""
        for c in range(w):
            agent_here = None
            for a in env.agents:
                if a.alive and a.y == r and a.x == c:
                    agent_here = a
                    break
            if agent_here is not None:
                row += f"A{agent_here.agent_id}"
            elif grid[r, c] == TILE_WALL:
                row += "##"
            else:
                row += " ."
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/dqn_final")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = args.config
    model_path = args.model

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = BattleRoyaleEnv(config=config)
    model = load_model(model_path, env)

    seed = args.seed
    obs, info = env.reset(seed=seed)

    print(f"=== BattleRL Arena - Auto Replay (seed={seed}) ===")
    print(f"    A0 = Trained AI (blue),  A1 = Random Enemy (red)")
    print()

    # Initial state
    map_rows = render_compact(env)
    print(f"--- Initial State ---")
    print(f"  A0 pos=({env.agents[0].y},{env.agents[0].x})  A1 pos=({env.agents[1].y},{env.agents[1].x})")
    for row in map_rows:
        print(f"  {row}")
    print()

    done = False
    total_reward = 0.0
    step = 0
    history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        a0_prev = (env.agents[0].y, env.agents[0].x)
        a1_prev = (env.agents[1].y, env.agents[1].x)
        a1_hp_prev = env.agents[1].hp

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

        a0_now = (env.agents[0].y, env.agents[0].x)
        a1_now = (env.agents[1].y, env.agents[1].x)
        a1_hp_now = env.agents[1].hp if env.agents[1].alive else 0

        # Build event description
        events = []
        action_name = ACTION_NAMES.get(action, "???")

        if action == ACTION_ATTACK and a1_hp_now < a1_hp_prev:
            dmg = a1_hp_prev - a1_hp_now
            events.append(f"HIT! -{dmg}dmg")
        if not env.agents[1].alive and a1_hp_prev > 0:
            events.append("KILL!")
        if not env.agents[0].alive:
            events.append("AI DIED!")

        event_str = " | ".join(events) if events else ""

        record = {
            "step": step, "action": action_name,
            "a0": a0_now, "a1": a1_now,
            "a0_hp": env.agents[0].hp, "a1_hp": a1_hp_now,
            "reward": reward, "events": event_str,
            "a0_alive": env.agents[0].alive, "a1_alive": env.agents[1].alive,
        }
        history.append(record)

    # Print battle log
    print(f"{'Step':>4}  {'Action':>6}  {'AI pos':>8}  {'AI HP':>6}  {'Enemy pos':>10}  {'Enemy HP':>8}  {'Reward':>7}  Events")
    print("-" * 90)

    for h in history:
        a0_pos = f"({h['a0'][0]:2d},{h['a0'][1]:2d})"
        a1_pos = f"({h['a1'][0]:2d},{h['a1'][1]:2d})" if h['a1_alive'] else "  DEAD  "
        a0_hp = f"{h['a0_hp']:3d}" if h['a0_alive'] else "DEAD"
        a1_hp = f"{h['a1_hp']:3d}" if h['a1_alive'] else "DEAD"
        reward = f"{h['reward']:+.2f}"

        print(f"{h['step']:4d}  {h['action']}  {a0_pos:>8}  {a0_hp:>6}  {a1_pos:>10}  {a1_hp:>8}  {reward:>7}  {h['events']}")

    # Final map
    print()
    print(f"--- Final State (Step {step}) ---")
    map_rows = render_compact(env)
    for row in map_rows:
        print(f"  {row}")

    print()
    if env.agents[0].alive and not env.agents[1].alive:
        print(f"  >> AI WINS! <<")
    elif not env.agents[0].alive and env.agents[1].alive:
        print(f"  >> ENEMY WINS <<")
    elif not env.agents[0].alive and not env.agents[1].alive:
        print(f"  >> DRAW <<")
    else:
        print(f"  >> TIME OUT (Draw) <<")

    print(f"  Total Steps: {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  AI Kills: {env.agents[0].kills}")
    print()

    env.close()


if __name__ == "__main__":
    main()
