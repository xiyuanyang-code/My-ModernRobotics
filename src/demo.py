import roboticstoolbox as rtb
from spatialmath import SE3
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np


def demo_1():
    robot = rtb.models.Panda()
    print(robot)
    Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ik_LM(Tep)  # solve IK
    print(sol)
    q_pickup = sol[0]
    print(robot.fkine(q_pickup))  # FK shows that desired end-effector pose was achieved

    # robot.plot(qt.q, backend='pyplot', movie="1.gif")
    qt = rtb.jtraj(robot.qr, q_pickup, 100)
    robot.plot(qt.q)


def demo_swift():
    # 1. 初始化机器人
    robot = rtb.models.Panda()
    robot.q = robot.qr  # 设置初始关节角度

    # 2. 初始化 Swift 仿真器
    env = swift.Swift()
    env.launch(realtime=True)  # 在浏览器中打开仿真界面

    # 3. 将机器人添加到仿真环境
    env.add(robot)

    # 4. 计算目标位姿和逆运动学
    Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ik_LM(Tep)
    q_pickup = sol[0]

    # 5. 生成轨迹 (50步)
    qt = rtb.jtraj(robot.qr, q_pickup, 50)

    # 6. 在 Swift 中循环播放轨迹
    for qk in qt.q:
        robot.q = qk  # 更新机器人关节状态
        env.step(0.02)  # 步进仿真器（0.02秒即50Hz）

    # 保持窗口开启
    env.hold()


def demo_2():
    env = swift.Swift()
    env.launch(realtime=True)

    panda = rtb.models.Panda()
    panda.q = panda.qr

    Tep = panda.fkine(panda.q) * sm.SE3.Trans(0.2, 0.2, 0.45)

    arrived = False
    env.add(panda)

    dt = 0.05

    while not arrived:

        v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
        panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
        env.step(dt)

    env.hold()


if __name__ == "__main__":
    # demo_swift()
    # demo_1()
    demo_2()
