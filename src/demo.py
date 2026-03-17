import roboticstoolbox as rtb
import swift
import spatialmath as sm
import numpy as np
from spatialmath import SE3


def demo_1():
    robot = rtb.models.Panda()
    robot.q = robot.qr
    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)

    Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ik_LM(Tep)
    q_pickup = sol[0]
    qt = rtb.jtraj(robot.qr, q_pickup, 1000)

    for qk in qt.q:
        robot.q = qk
        env.step(0.01)



def demo_2():
    env = swift.Swift()
    env.launch(realtime=True)

    panda = rtb.models.Panda()
    panda.q = panda.qr

    Tep = panda.fkine(panda.q) * sm.SE3.Trans(0.2, 0.2, 0.45)

    arrived = False
    env.add(panda)

    dt = 0.01
    while not arrived:
        v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
        panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
        env.step(dt)



def plot_robot_trajectory(
    start_point: tuple,
    end_point: tuple,
):
    """
    Plot robot trajectory from start point to end point in 3D space.

    Args:
        start_point: Tuple of (x, y, z) coordinates for the starting position.
        end_point: Tuple of (x, y, z) coordinates for the ending position.
        degrees_of_freedom: Number of active joints (default: 7 for full Panda).

    Returns:
        None
    """
    robot = rtb.models.Panda()

    # Initialize robot to ready position
    robot.q = robot.qr

    # Setup Swift environment
    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)

    # Create target transformation for end point
    Tep = SE3.Trans(*end_point) * SE3.OA([0, 1, 0], [0, 0, -1])

    # Calculate inverse kinematics for target position
    sol = robot.ik_LM(Tep)
    if not sol:
        raise ValueError("无法求解逆运动学，目标位置可能不可达")

    q_target = sol[0]

    # Calculate inverse kinematics for start position
    T_start = SE3.Trans(*start_point) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol_start = robot.ik_LM(T_start)
    if not sol_start:
        raise ValueError("无法求解起始位置逆运动学，起始位置可能不可达")

    q_start = sol_start[0]
    # Generate joint trajectory
    qt = rtb.jtraj(q_start, q_target, 500)

    # Animate robot motion along trajectory
    for qk in qt.q:
        robot.q = qk
        env.step(0.01)


if __name__ == "__main__":
    # demo_1()
    # demo_2()
    plot_robot_trajectory(
      start_point=(-1,-2, 8),
      end_point=(4, 2, 5),
  )
