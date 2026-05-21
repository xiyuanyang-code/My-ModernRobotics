from gymnasium.envs.registration import register
import numpy as np

register(
    id="TwoLinkArm-v0",
    entry_point="robotics_control.arm_env:TwoLinkArmEnv",
    kwargs={"goal_q": np.array([2.56, 3.0])},
)

register(
    id="TwoLinkArm-random-goal-v0",
    entry_point="robotics_control.arm_env:TwoLinkArmEnv",
)

register(
    id="TwoLinkArm-limited-torque-v0",
    entry_point="robotics_control.arm_env:LimitedTorqueTwoLinkArmEnv",
    kwargs={"goal_q": np.array([2.56, 3.0])},
)

register(
    id="TwoLinkArm-limited-torque-random-goal-v0",
    entry_point="robotics_control.arm_env:LimitedTorqueTwoLinkArmEnv",
)

register(
    id="TwoLinkArm-v1",
    entry_point="robotics_control.arm_env:TwoLinkArmEnv",
    kwargs={"goal_q": np.array([2.56, 3.0]), "R": np.eye(2)},
)

register(
    id="TwoLinkArm-random-goal-v1",
    entry_point="robotics_control.arm_env:TwoLinkArmEnv",
    kwargs={"R": np.eye(2)},
)

register(
    id="TwoLinkArm-limited-torque-v1",
    entry_point="robotics_control.arm_env:LimitedTorqueTwoLinkArmEnv",
    kwargs={"goal_q": np.array([2.56, 3.0]), "R": np.eye(2)},
)

register(
    id="TwoLinkArm-limited-torque-random-goal-v1",
    entry_point="robotics_control.arm_env:LimitedTorqueTwoLinkArmEnv",
    kwargs={"R": np.eye(2)},
)

register(
    id="CartPoleSwingUpStabilize-v0",
    entry_point="robotics_control.cartpole:SwingUpStabilizeCartPoleEnv",
)

register(
    id="CartPoleUprightStabilize-v0",
    entry_point="robotics_control.cartpole:UprightStabilizeCartPoleEnv",
)

register(
    id="CartPoleUprightContinuousStabilize-v0",
    entry_point="robotics_control.cartpole:ContinuousActionUprightStabilizeCartPoleEnv",
)
