import gym
gym.logger.set_level(40)  # nopep8
from gym.envs.registration import registry
from eagent.gym_evolving_locomotion_envs import EvolvingWalkerEnv  # noqa
from eagent.gym_evolving_manipulate_envs import EvolvingHandEnv  # noqa


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ----------------------------------
# Walking task
register(
    id="EvolvingWalkerEnv-v2",
    entry_point="eagent:EvolvingWalkerEnv",
    max_episode_steps=1000,  # Should be matched with num_steps_in_eval
    reward_threshold=99999.0,  # This might not be needed
)

# ----------------------------------
# EvolvingBlock
register(
    id="EvolvingHandBlockDense-v1",
    entry_point="eagent:EvolvingHandEnv",
    max_episode_steps=100,
    kwargs={
        "target_position": "fixed_random",
        "target_rotation": "xyz",
        "reward_type": "dense",
        "object_type": "block",
    },
)
register(
    id="EvolvingHandBlockSparse-v1",
    entry_point="eagent:EvolvingHandEnv",
    max_episode_steps=100,
    kwargs={
        "target_position": "fixed_random",
        "target_rotation": "xyz",
        "reward_type": "sparse",
        "object_type": "block",
    },
)
register(
    id="EvolvingHandBlockRotateZDense-v1",
    entry_point="eagent:EvolvingHandEnv",
    max_episode_steps=100,
    kwargs={
        "target_position": "ignore",
        "target_rotation": "z",
        "reward_type": "dense",
        "object_type": "block",
    },
)
register(
    id="EvolvingHandBlockRotateZSparse-v1",
    entry_point="eagent:EvolvingHandEnv",
    max_episode_steps=100,
    kwargs={
        "target_position": "ignore",
        "target_rotation": "z",
        "reward_type": "sparse",
        "object_type": "block",
    },
)

# ----------------------------------
# EvolvingEgg
register(
    id="EvolvingHandEggRotateZSparse-v1",
    entry_point="eagent:EvolvingHandEnv",
    max_episode_steps=100,
    kwargs={
        "target_position": "ignore",
        "target_rotation": "z",
        "reward_type": "sparse",
        "object_type": "egg",
    },
)
register(
    id="EvolvingHandEggRotateSparse-v1",
    entry_point="eagent:EvolvingHandEnv",
    max_episode_steps=100,
    kwargs={
        "target_position": "ignore",
        "target_rotation": "xyz",
        "reward_type": "sparse",
        "object_type": "egg",
    },
)
