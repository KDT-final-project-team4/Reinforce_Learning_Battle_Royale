"""DQN/PPO 학습 실행 스크립트"""

import argparse
import os
import sys
import yaml

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.battle_env import BattleRoyaleEnv
from training.callbacks import BattleRoyaleCallback, SelfPlayCallback


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_fn(config: dict):
    """SubprocVecEnv용 환경 생성 팩토리 함수."""
    def _init():
        env = BattleRoyaleEnv(config=config)
        env = Monitor(env)
        return env
    return _init


def make_env(config: dict, n_envs: int = 1, use_subproc: bool = False):
    """벡터화된 환경을 생성한다."""
    env_fns = [make_env_fn(config) for _ in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def train(args):
    # 설정 로드
    config = load_config(args.config)

    # 커맨드라인 인자로 오버라이드
    if args.map_size:
        config["map"]["width"] = args.map_size
        config["map"]["height"] = args.map_size
    if args.total_timesteps:
        config["training"]["total_timesteps"] = args.total_timesteps

    training_cfg = config["training"]

    # 병렬 환경 설정
    n_envs = args.n_envs or training_cfg.get("n_envs", 1)
    use_subproc = n_envs > 1
    env = make_env(config, n_envs=n_envs, use_subproc=use_subproc)

    # 알고리즘 선택
    algo_name = args.algo or training_cfg["algorithm"]
    lr = training_cfg["learning_rate"]
    total_timesteps = training_cfg["total_timesteps"]

    print(f"=== BattleRL Arena Training ===")
    print(f"Algorithm: {algo_name}")
    print(f"Map Size: {config['map']['width']}x{config['map']['height']}")
    print(f"Agents: {config['agent']['count']}")
    print(f"View Range: {config['agent']['view_range']}")

    # Phase 4 팀/역할 정보
    team_cfg = config.get("team", {})
    if team_cfg:
        print(f"Teams: {team_cfg.get('num_teams', '?')} teams × {team_cfg.get('agents_per_team', '?')} agents")
        print(f"Friendly Fire: {team_cfg.get('friendly_fire', False)}")
    roles_cfg = config.get("roles", {})
    if roles_cfg:
        role_names = list(roles_cfg.keys())
        print(f"Roles: {', '.join(role_names)}")
        for rname, rstats in roles_cfg.items():
            print(f"  {rname}: HP={rstats.get('hp','?')} ATK={rstats.get('attack','?')} "
                  f"DEF={rstats.get('defense','?')} Range={rstats.get('attack_range','?')}")

    print(f"Total Timesteps: {total_timesteps}")
    print(f"Learning Rate: {lr}")
    print(f"Parallel Envs: {n_envs} ({'SubprocVecEnv' if use_subproc else 'DummyVecEnv'})")

    # 관측/행동 공간 크기 출력
    temp_env = BattleRoyaleEnv(config=config)
    obs, _ = temp_env.reset()
    print(f"Observation Space: {temp_env.observation_space.shape} ({obs.shape[0]} dims)")
    print(f"Action Space: Discrete({temp_env.action_space.n})")
    temp_env.close()
    print()

    log_dir = os.path.join("logs", "tensorboard", f"{algo_name.lower()}_run")
    os.makedirs(log_dir, exist_ok=True)

    # 체크포인트에서 이어서 학습
    if args.resume:
        print(f"Resuming from: {args.resume}")
        AlgoClass = DQN if algo_name.upper() == "DQN" else PPO
        model = AlgoClass.load(args.resume, env=env, tensorboard_log=log_dir)
        # 남은 스텝 계산 (reset_num_timesteps=False로 이어서 카운트)
    elif algo_name.upper() == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=50000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log=log_dir,
        )
    elif algo_name.upper() == "PPO":
        # config에서 PPO 하이퍼파라미터 읽기 (Phase 3 지원)
        gamma = training_cfg.get("gamma", 0.99)
        n_steps = training_cfg.get("n_steps", 2048)
        ent_coef = training_cfg.get("ent_coef", 0.0)
        net_arch = training_cfg.get("net_arch", None)

        policy_kwargs = {}
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=64,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs if policy_kwargs else {},
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    # 콜백 설정
    callbacks = [
        BattleRoyaleCallback(
            save_dir="models",
            save_freq=50000,
            verbose=1,
        ),
    ]

    # Self-Play 콜백
    if training_cfg.get("self_play", False):
        callbacks.append(
            SelfPlayCallback(
                update_interval=training_cfg.get("opponent_update_interval", 10000),
                verbose=1,
            )
        )

    # 학습 시작
    # resume 시: reset_num_timesteps=True로 하되, 남은 스텝만 학습
    if args.resume:
        already_done = model.num_timesteps
        remaining = max(total_timesteps - already_done, 0)
        print(f"Already trained: {already_done} steps, remaining: {remaining} steps")
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
        )
    else:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

    # 최종 모델 저장
    save_path = os.path.join("models", f"{algo_name.lower()}_final")
    model.save(save_path)
    print(f"\nFinal model saved: {save_path}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="BattleRL Arena Training")
    parser.add_argument("--algo", type=str, default=None,
                        help="Algorithm: DQN or PPO")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total training timesteps")
    parser.add_argument("--map-size", type=int, default=None,
                        help="Map size (square)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Config file path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint (e.g. models/checkpoint_50000)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments (default: from config)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
