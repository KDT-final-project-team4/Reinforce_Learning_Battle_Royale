"""멀티 정책 학습 스크립트 — 역할별 3개 PPO 모델을 라운드 로빈으로 학습"""

import argparse
import os
import sys

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agent import ROLE_TANK, ROLE_DEALER, ROLE_HEALER, ALL_ROLES
from env.battle_env import BattleRoyaleEnv
from training.callbacks import BattleRoyaleCallback, MultiPolicySelfPlayCallback


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_fn(config: dict, learning_role: str):
    """역할 고정 환경 생성 팩토리."""
    def _init():
        env = BattleRoyaleEnv(config=config, learning_role=learning_role)
        env = Monitor(env)
        return env
    return _init


def make_env(config: dict, n_envs: int, learning_role: str):
    """역할별 벡터화 환경 생성."""
    env_fns = [make_env_fn(config, learning_role) for _ in range(n_envs)]
    return DummyVecEnv(env_fns)


def create_model(role: str, env, config: dict, log_dir: str) -> PPO:
    """역할별 PPO 모델 생성."""
    training_cfg = config["training"]
    lr = training_cfg["learning_rate"]
    gamma = training_cfg.get("gamma", 0.99)
    n_steps = training_cfg.get("n_steps", 2048)
    ent_coef = training_cfg.get("ent_coef", 0.0)
    net_arch = training_cfg.get("net_arch", None)

    policy_kwargs = {}
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    role_log_dir = os.path.join(log_dir, role)
    os.makedirs(role_log_dir, exist_ok=True)

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=64,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=role_log_dir,
        policy_kwargs=policy_kwargs if policy_kwargs else {},
    )


def set_initial_opponents(env, snapshot_dir: str):
    """환경에 기존 스냅샷이 있으면 로드."""
    paths = {}
    for role in ALL_ROLES:
        p = os.path.join(snapshot_dir, role, f"{role}_snapshot")
        if os.path.exists(p + ".zip"):
            paths[role] = p
    if paths:
        env.env_method("set_opponent_paths", paths)


def train_multi(args):
    config = load_config(args.config)
    training_cfg = config["training"]

    n_envs = args.n_envs or training_cfg.get("n_envs", 8)
    num_rounds = training_cfg.get("num_rounds", 10)
    steps_per_round = training_cfg.get("steps_per_round", 300000)
    opponent_update_interval = training_cfg.get("opponent_update_interval", 25000)

    snapshot_dir = os.path.join("models", "multi_policy")
    log_dir = os.path.join("logs", "tensorboard", "multi_policy")
    os.makedirs(snapshot_dir, exist_ok=True)

    roles = [ROLE_TANK, ROLE_DEALER, ROLE_HEALER]

    # 정보 출력
    print("=== BattleRL Multi-Policy Training ===")
    print(f"Roles: {', '.join(roles)}")
    print(f"Rounds: {num_rounds}")
    print(f"Steps/round/role: {steps_per_round:,}")
    print(f"Total steps/role: {num_rounds * steps_per_round:,}")
    print(f"Parallel Envs: {n_envs}")
    print(f"Opponent Update: every {opponent_update_interval} steps")

    # 역할별 행동 공간 출력
    for role in roles:
        temp_env = BattleRoyaleEnv(config=config, learning_role=role)
        print(f"  {role}: action_space=Discrete({temp_env.action_space.n}), "
              f"obs_space={temp_env.observation_space.shape}")
        temp_env.close()
    print()

    # 모델 딕셔너리 (라운드 간 유지)
    models: dict[str, PPO] = {}

    for round_idx in range(num_rounds):
        # ── 라운드 시작마다 config 재로드 ──────────────────────────────
        # 학습 실행 중에도 phase4_team.yaml을 수정하면 다음 라운드부터 반영됨
        config = load_config(args.config)
        training_cfg = config["training"]
        n_envs = args.n_envs or training_cfg.get("n_envs", 8)
        steps_per_round = training_cfg.get("steps_per_round", steps_per_round)
        opponent_update_interval = training_cfg.get("opponent_update_interval", opponent_update_interval)
        # ────────────────────────────────────────────────────────────────

        print(f"\n{'='*60}")
        print(f"  Round {round_idx + 1}/{num_rounds}")
        print(f"  Config reloaded from: {args.config}")
        print(f"{'='*60}")

        for role in roles:
            print(f"\n--- Training: {role.upper()} (Round {round_idx + 1}) ---")

            # 환경은 매번 새 config로 생성 (보상 설정 반영)
            env = make_env(config, n_envs=n_envs, learning_role=role)

            # 모델 로드 또는 생성
            role_dir = os.path.join(snapshot_dir, role)
            os.makedirs(role_dir, exist_ok=True)
            latest_path = os.path.join(role_dir, f"{role}_latest")

            if role in models:
                # 가중치 유지, 환경만 새 config로 교체
                models[role].set_env(env)
            elif os.path.exists(latest_path + ".zip"):
                # 이전 학습에서 이어서
                role_log_dir = os.path.join(log_dir, role)
                models[role] = PPO.load(latest_path, env=env,
                                        tensorboard_log=role_log_dir)
                print(f"  Resumed from: {latest_path}")
            else:
                models[role] = create_model(role, env, config, log_dir)

            model = models[role]

            # 기존 스냅샷으로 상대 초기화
            set_initial_opponents(env, snapshot_dir)

            # 콜백
            callbacks = [
                BattleRoyaleCallback(
                    save_dir=os.path.join(role_dir, "checkpoints"),
                    save_freq=50000,
                    verbose=1,
                    role=role,
                ),
                MultiPolicySelfPlayCallback(
                    learning_role=role,
                    snapshot_dir=snapshot_dir,
                    update_interval=opponent_update_interval,
                    verbose=1,
                ),
            ]

            # 학습
            try:
                model.learn(
                    total_timesteps=steps_per_round,
                    callback=callbacks,
                    progress_bar=True,
                    reset_num_timesteps=False,
                )
            except Exception as e:
                print(f"\n[WARNING] {role.upper()} 학습 중 예외 발생: {e}")
                print("  현재까지 학습된 모델을 저장하고 다음 역할로 진행합니다.")

            # 스냅샷 저장 (예외 발생 시에도 반드시 저장)
            model.save(os.path.join(role_dir, f"{role}_snapshot"))
            model.save(latest_path)
            print(f"  [{role.upper()}] 스냅샷 저장 완료: {latest_path}")

            env.close()

        # 라운드 종료 시 체크포인트
        for role in roles:
            models[role].save(
                os.path.join(snapshot_dir, role, f"{role}_round{round_idx + 1}")
            )
        print(f"\n  Round {round_idx + 1} complete. Snapshots saved.")

    # 최종 모델 저장
    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"{'='*60}")
    for role in roles:
        final_path = os.path.join(snapshot_dir, role, f"{role}_final")
        models[role].save(final_path)
        print(f"  {role}: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="BattleRL Multi-Policy Training")
    parser.add_argument("--config", type=str, default="config/phase4_team.yaml",
                        help="Config file path")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--num-rounds", type=int, default=None,
                        help="Override num_rounds from config")
    parser.add_argument("--steps-per-round", type=int, default=None,
                        help="Override steps_per_round from config")
    args = parser.parse_args()

    # 커맨드라인 오버라이드
    config = load_config(args.config)
    if args.num_rounds:
        config["training"]["num_rounds"] = args.num_rounds
    if args.steps_per_round:
        config["training"]["steps_per_round"] = args.steps_per_round

    # 오버라이드된 config를 임시 저장 (train_multi가 다시 읽으므로)
    # 대신 args에서 직접 사용
    train_multi(args)


if __name__ == "__main__":
    main()
