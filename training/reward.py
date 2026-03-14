"""보상 함수 정의 및 관리

보상 함수를 모듈화하여 실험별로 쉽게 교체할 수 있도록 한다.
현재는 battle_env.py 내의 _calculate_reward에서 직접 계산하고 있으며,
이 파일은 보상 설정을 실험별로 프리셋으로 관리하기 위한 유틸리티이다.
"""


# Phase 1 기본 보상 프리셋
REWARD_PHASE1_BASIC = {
    "kill": 10.0,
    "death": -10.0,
    "damage_dealt": 1.0,
    "item_pickup": 0.0,      # Phase 1에서는 아이템 없음
    "potion_use": 0.0,
    "zone_damage": 0.0,
    "survival_per_step": 0.01,
    "idle_penalty": -0.05,
}

# Phase 2 확장 보상 프리셋
REWARD_PHASE2_FULL = {
    "kill": 10.0,
    "death": -10.0,
    "damage_dealt": 1.0,
    "item_pickup": 2.0,
    "potion_use": 1.0,
    "zone_damage": -1.0,
    "survival_per_step": 0.01,
    "idle_penalty": -0.05,
}

# 공격 중심 보상 (실험용)
REWARD_AGGRESSIVE = {
    "kill": 15.0,
    "death": -5.0,
    "damage_dealt": 2.0,
    "item_pickup": 1.0,
    "potion_use": 0.5,
    "zone_damage": -1.0,
    "survival_per_step": 0.0,
    "idle_penalty": -0.1,
}


def get_reward_preset(name: str) -> dict:
    """이름으로 보상 프리셋을 반환한다."""
    presets = {
        "phase1_basic": REWARD_PHASE1_BASIC,
        "phase2_full": REWARD_PHASE2_FULL,
        "aggressive": REWARD_AGGRESSIVE,
    }
    if name not in presets:
        raise ValueError(f"Unknown reward preset: {name}. Available: {list(presets.keys())}")
    return presets[name].copy()
