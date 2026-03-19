"""넥서스 모드 학습된 모델 평가 — 학습 콜백과 동일한 상세 로그 출력"""

import argparse
import os
import sys

import numpy as np
import yaml
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agent import ALL_ROLES
from env.nexus_env import NexusBattleEnv
from env.base_env import ROLE_ACTION_MAP

# 학습 콜백과 동일한 보상 키
NEXUS_REWARD_KEYS = [
    "kill", "death", "damage_dealt", "item_pickup",
    "survival", "idle", "wall_bump", "attack_miss",
    "combo", "no_combat", "approach", "flee", "oscillation",
    "heal_ally", "ranged_miss", "invalid_action",
    "teammate_death", "approach_teammate",
    "attack_cooldown", "heal_cooldown",
    "nexus_damage", "own_nexus_damaged",
    "nexus_destroyed_win", "nexus_destroyed_loss",
    "approach_nexus", "defend_nexus", "stay_near_own_nexus",
    "timeout_advantage", "timeout_disadvantage",
    "minion_kill",
]


def load_model(model_path: str):
    """현재 프로젝트에서는 PPO만 사용하므로 PPO 전용 로더로 단순화."""
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    return PPO.load(model_path)


def load_role_models(model_dir: str) -> dict:
    """역할별 모델을 로드한다."""
    models = {}
    for role in ALL_ROLES:
        for suffix in ["_final", "_latest", "_snapshot"]:
            path = os.path.join(model_dir, role, f"{role}{suffix}")
            if os.path.exists(path + ".zip"):
                models[role] = load_model(path)
                print(f"  Loaded {role}: {path}")
                break
    return models


def evaluate(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = NexusBattleEnv(config=config)
    # 평가 시에는 양 팀 모두 deterministic으로 고정 (탐험 노이즈 제거)
    env.set_policy_deterministic(True)
    # Team1도 Team0과 같은 좌표계로 보이도록 정책 입력/출력 정규화
    env.set_policy_team_normalize(True)

    # 팀1(상대)용: --model-dir (기본값: 최신 버전)
    opponent_models = load_role_models(args.model_dir)
    if opponent_models:
        env.set_team_role_models(1, opponent_models)

    # 팀0(학습 에이전트 팀, 에이전트 0 포함)용: --self-model-dir
    # 명시하지 않으면 opponent_models와 동일한 버전 사용
    if args.self_model_dir:
        self_models = load_role_models(args.self_model_dir)
    else:
        self_models = opponent_models

    # Team 0 나머지 팀원(에이전트 1~4)도 같은 세트로 구동
    if self_models:
        env.set_team_role_models(0, self_models)

    episodes = args.episodes
    win_count = 0
    episode_rewards = []
    episode_lengths = []
    reward_history = {k: [] for k in NEXUS_REWARD_KEYS}

    print(f"=== Nexus Evaluation ===")
    print(f"Opponent model dir (Team 1): {args.model_dir}")
    if args.self_model_dir:
        print(f"Self model dir (Team 0):     {args.self_model_dir}")
    else:
        print(f"Self model dir (Team 0):     (same as opponent)")
    print(f"Episodes: {episodes}")
    print(f"Seed: {args.seed}")
    print()

    for ep in range(episodes):
        obs, info = env.reset(seed=args.seed)
        done = False
        ep_reward = 0.0
        ep_reward_sums = {k: 0.0 for k in NEXUS_REWARD_KEYS}

        while not done:
            # 에이전트 0(팀0)의 행동: self_models 사용
            if self_models:
                agent0_role = env.agents[0].role
                if agent0_role in self_models:
                    role_action, _ = self_models[agent0_role].predict(obs, deterministic=True)
                    role_map = ROLE_ACTION_MAP[agent0_role]
                    action = role_map[int(role_action)]
                else:
                    action = 8  # ACTION_STAY
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            rd = info.get("reward_details", {})
            for k in NEXUS_REWARD_KEYS:
                ep_reward_sums[k] += rd.get(k, 0.0)

        episode_rewards.append(ep_reward)
        episode_lengths.append(info.get("step", 0))

        winning_team = info.get("winning_team", None)
        agents_info = info.get("agents", [])
        if agents_info:
            my_team = agents_info[0].get("team_id", 0)
            if winning_team is not None and winning_team == my_team:
                win_count += 1

        for k in NEXUS_REWARD_KEYS:
            reward_history[k].append(ep_reward_sums[k])

        # 100 에피소드마다 상세 로그 (학습 콜백과 동일 형식)
        if (ep + 1) % 100 == 0:
            recent = episode_rewards[-100:]
            recent_len = episode_lengths[-100:]
            win_rate = win_count / (ep + 1)
            total_steps = sum(episode_lengths)
            print(f"\n[EVAL Episode {ep + 1}] "
                  f"Total steps: {total_steps:,} | "
                  f"Avg Reward: {np.mean(recent):.2f} | "
                  f"Win Rate: {win_rate:.2%} | "
                  f"Avg Length: {np.mean(recent_len):.0f} (min={min(recent_len):.0f} max={max(recent_len):.0f})")

            key_metrics = [
                "kill", "death", "damage_dealt", "heal_ally",
                "nexus_damage", "own_nexus_damaged",
                "nexus_destroyed_win", "nexus_destroyed_loss",
                "minion_kill", "defend_nexus",
                "invalid_action",
            ]
            parts = []
            for k in key_metrics:
                r100 = reward_history[k][-100:]
                if r100 and np.mean(r100) != 0:
                    parts.append(f"{k}={np.mean(r100):+.2f}")
            if parts:
                print(f"  Rewards: {' | '.join(parts)}")

            approach_metrics = ["approach_nexus", "approach", "approach_teammate"]
            approach_parts = []
            for k in approach_metrics:
                r100 = reward_history.get(k, [])[-100:]
                if r100 and np.mean(r100) != 0:
                    approach_parts.append(f"{k}={np.mean(r100):+.2f}")
            if approach_parts:
                print(f"  Approach: {' | '.join(approach_parts)}")

            penalty_metrics = ["idle", "no_combat", "stay_near_own_nexus", "attack_miss", "ranged_miss", "wall_bump"]
            penalty_parts = []
            for k in penalty_metrics:
                r100 = reward_history.get(k, [])[-100:]
                if r100 and np.mean(r100) != 0:
                    penalty_parts.append(f"{k}={np.mean(r100):+.2f}")
            if penalty_parts:
                print(f"  Penalties: {' | '.join(penalty_parts)}")

            nexuses_info = info.get("nexuses", [])
            if nexuses_info:
                hp_parts = [f"Team{n['team_id']}={n['hp']}/{n['max_hp']}" for n in nexuses_info]
                print(f"  Nexus HP: {' | '.join(hp_parts)}")

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

            minions_alive = info.get("minions_alive", None)
            if minions_alive is not None:
                print(f"  Minions alive (end): {minions_alive}")

    # 최종 결과
    print(f"\n=== Nexus Evaluation Complete ({episodes} episodes) ===")
    print(f"Win Rate: {win_count / episodes:.2%}")
    print(f"Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Steps: {np.mean(episode_lengths):.1f}")
    print(f"\n--- Reward Breakdown (last 100 episodes) ---")
    for k in NEXUS_REWARD_KEYS:
        recent = reward_history[k][-100:]
        if recent and np.mean(recent) != 0:
            print(f"  {k:>25s}: {np.mean(recent):+.3f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nexus mode models")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/nexus_multi_policy",
        help="상대 팀(Team 1)에 사용할 역할별 모델 디렉터리",
    )
    parser.add_argument(
        "--self-model-dir",
        type=str,
        default=None,
        help="내 팀(Team 0, 에이전트 0 포함)에 사용할 역할별 모델 디렉터리 (생략 시 --model-dir와 동일)",
    )
    parser.add_argument("--config", type=str, default="config/nexus_mode.yaml")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
