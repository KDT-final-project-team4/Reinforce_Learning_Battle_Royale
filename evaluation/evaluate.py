"""학습된 모델 평가 스크립트"""

import argparse
import os
import sys

import numpy as np
import yaml
from stable_baselines3 import DQN, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.battle_env import BattleRoyaleEnv


def load_model(model_path: str, env):
    """모델 파일을 로드한다. 확장자/알고리즘 자동 감지."""
    # .zip 확장자가 없으면 추가
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    # DQN 먼저 시도, 실패 시 PPO
    try:
        return DQN.load(model_path, env=env)
    except Exception:
        return PPO.load(model_path, env=env)


def evaluate(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    episodes = args.episodes
    wins = 0
    total_rewards = []
    total_kills = []
    total_steps = []

    # 환경 생성
    env = BattleRoyaleEnv(config=config)

    # 모델 로드
    model = load_model(args.model, env)

    # 상대 설정
    opponent_model = None
    if args.opponent and args.opponent != "random":
        opponent_model = load_model(args.opponent, env)
        env.opponent_policy = opponent_model

    print(f"=== Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Opponent: {args.opponent or 'random'}")
    print(f"Episodes: {episodes}")
    print()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)
        total_steps.append(info["step"])

        # 승리 판정
        agent_info = info["agents"][0]
        if agent_info["alive"]:
            wins += 1
        total_kills.append(agent_info.get("kills", 0))

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{episodes} - "
                  f"Win Rate: {wins / (ep + 1):.2%}")

    print(f"\n=== Results ({episodes} episodes) ===")
    print(f"Win Rate: {wins / episodes:.2%}")
    print(f"Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Avg Steps: {np.mean(total_steps):.1f}")
    print(f"Avg Kills: {np.mean(total_kills):.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--opponent", type=str, default="random",
                        help="Opponent: 'random' or path to model")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
