# Intro to Robots

- AI Agent: 受限在语言空间中，输出可以通过 Tools 等执行更复杂的操作，与环境的交互反馈 & 执行操作。
- Robotics: 获得真实的物理环境交互和反馈

## Outline

- 机器人基础：
    - 机器人运动学 Robot Kinematics
        - 机器人姿态控制
        - 正向运动学（已知关节角求末端位置）
        - 逆向运动学（已知目标位置求关节角）
    - 机器人动力学 Robot Dynamics
        - 机器人力学和机器人运动的纽带
- 机器人算法：
    - Control
        - Low Level Control：电机驱动
        - Feedback Control: 反馈调整控制
        - Motion Control: Achieve a certain motions
        - Force Control: Achieve a certain force
        - Model Base Control (MPC)
        - Imitation Learning & Deep Reinforcement Learning
    - Perception（传感器）
        - 激光雷达 & 点云
        - 触觉传感器 Tactile
        - 本地感受
    - Planning
        - 路径规划算法
    
- Robotics Applications:
    - Manipulations 机械臂操作和控制
        - data. policy, systems
        - VLA (Vision-Language-Actions Models)
    - Locomotion 移动
        - ZMP: 零力矩点
        - Model Predictive Control 模型预测控制
        - RL + Sim2Real: 强化学习 + 环境仿真
    - Navigation 导航

