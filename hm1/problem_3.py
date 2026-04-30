import pybullet as p
import pybullet_data
import numpy as np
import time
import random

NAME = "Xiyuan Yang"
STUDENT_ID = "524531910015"


def load_robot():
    """
    Load ARX X5 robot in PyBullet.

    Returns:
        robot_id: PyBullet robot ID
    """
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1 / 240.0)
    # Set camera position for good view
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=50,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5],
    )
    create_text_overlay(NAME, STUDENT_ID)

    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(
        "/Users/xiyuanyang/Desktop/FK_Homwork/ARX-X5/X5A.urdf",
        [0, 0, 0],
        useFixedBase=True,
    )

    if isinstance(robot_id, list):
        robot = robot_id[0]
    else:
        robot = robot_id

    num_joints = p.getNumJoints(robot) if robot is not None else 0
    print(f"Number of joints in robot: {num_joints}")

    return robot


def create_text_overlay(name, student_id):
    """
    Create text overlay with name and student ID.

    Args:
        name: Student name
        student_id: Student ID
    """
    text = f"{name} | ID: {student_id}"
    p.addUserDebugText(text, [0, 0, 1.2], textColorRGB=[0, 0, 0], textSize=1.5)


def keyboard_control(robot_id):
    """
    6-DOF keyboard control with inverse kinematics.

    Controls:
        - x/y/z: position control
        - r/p/y: roll/pitch/yaw control
        - Space: reset pose

    Args:
        robot_id: PyBullet robot ID
    """

    # Initial end-effector pose (adjusted for workspace)
    pos = [0.3, 0.0, 0.4]
    orn = p.getQuaternionFromEuler([0, 0, 0])

    # Step size for incremental changes
    pos_step = 0.02
    orn_step = 0.05

    # Get joint indices
    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1  # Last joint as end-effector

    # Enable joint motors for smooth control
    for i in range(num_joints):
        p.setJointMotorControl2(
            robot_id, i, p.POSITION_CONTROL, force=500
        )

    print("\n=== Keyboard Control ===")
    print("Position:  i/k (x), j/l (y), u/o (z)")
    print("Orientation: 1/2 (roll), 3/4 (pitch), 5/6 (yaw)")
    print("0: Reset to initial pose")
    print("Close window to exit\n")

    while p.isConnected():
        # Get keyboard events
        keys = p.getKeyboardEvents()

        # Position control
        if ord(b"i") in keys and keys[ord(b"i")] & p.KEY_IS_DOWN:
            pos[0] += pos_step
        if ord(b"k") in keys and keys[ord(b"k")] & p.KEY_IS_DOWN:
            pos[0] -= pos_step
        if ord(b"j") in keys and keys[ord(b"j")] & p.KEY_IS_DOWN:
            pos[1] += pos_step
        if ord(b"l") in keys and keys[ord(b"l")] & p.KEY_IS_DOWN:
            pos[1] -= pos_step
        if ord(b"u") in keys and keys[ord(b"u")] & p.KEY_IS_DOWN:
            pos[2] += pos_step
        if ord(b"o") in keys and keys[ord(b"o")] & p.KEY_IS_DOWN:
            pos[2] -= pos_step

        # Orientation control
        if ord(b"1") in keys and keys[ord(b"1")] & p.KEY_IS_DOWN:
            euler = p.getEulerFromQuaternion(orn)
            orn = p.getQuaternionFromEuler([euler[0] + orn_step, euler[1], euler[2]])
        if ord(b"2") in keys and keys[ord(b"2")] & p.KEY_IS_DOWN:
            euler = p.getEulerFromQuaternion(orn)
            orn = p.getQuaternionFromEuler([euler[0] - orn_step, euler[1], euler[2]])
        if ord(b"3") in keys and keys[ord(b"3")] & p.KEY_IS_DOWN:
            euler = p.getEulerFromQuaternion(orn)
            orn = p.getQuaternionFromEuler([euler[0], euler[1] + orn_step, euler[2]])
        if ord(b"4") in keys and keys[ord(b"4")] & p.KEY_IS_DOWN:
            euler = p.getEulerFromQuaternion(orn)
            orn = p.getQuaternionFromEuler([euler[0], euler[1] - orn_step, euler[2]])
        if ord(b"5") in keys and keys[ord(b"5")] & p.KEY_IS_DOWN:
            euler = p.getEulerFromQuaternion(orn)
            orn = p.getQuaternionFromEuler([euler[0], euler[1], euler[2] + orn_step])
        if ord(b"6") in keys and keys[ord(b"6")] & p.KEY_IS_DOWN:
            euler = p.getEulerFromQuaternion(orn)
            orn = p.getQuaternionFromEuler([euler[0], euler[1], euler[2] - orn_step])

        # Reset
        if ord(b"0") in keys and keys[ord(b"0")] & p.KEY_WAS_TRIGGERED:
            pos = [0.3, 0.0, 0.4]
            orn = p.getQuaternionFromEuler([0, 0, 0])
            print("Reset to initial pose")

        # Compute IK
        joint_poses = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            pos,
            orn,
            lowerLimits=[-3.14] * num_joints,
            upperLimits=[3.14] * num_joints,
            jointRanges=[3.14] * num_joints,
            restPoses=[0] * num_joints,
        )

        # Apply joint positions using motor control (smooth)
        for i in range(min(num_joints, len(joint_poses))):
            p.setJointMotorControl2(
                robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
            )

        # Display current pose
        euler = p.getEulerFromQuaternion(orn)
        p.addUserDebugText(
            f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\nRPY: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]",
            [0, 0, 1.4],
            textColorRGB=[0, 0, 1],
            textSize=1,
            lifeTime=0.1,
        )

        # Step simulation
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def move_to_pose(robot_id, target_pos, target_orn, steps=240):
    """
    Smoothly move end-effector from current pose to target pose.

    Args:
        robot_id: PyBullet robot ID
        target_pos: (3,) target position [x, y, z]
        target_orn: (4,) target orientation as quaternion
        steps: number of interpolation steps (default 240 = 1 second at 240Hz)
    """

    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1

    # Get current end-effector pose
    state = p.getLinkState(robot_id, end_effector_index)
    cur_pos = np.array(state[0])
    cur_orn = np.array(state[1])

    target_pos = np.array(target_pos)
    target_orn = np.array(target_orn)

    print(f"Current position: [{cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f}]")
    print(f"Target  position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"Moving in {steps} steps ({steps/240:.1f}s)...")

    for step in range(steps + 1):
        t = step / steps

        # Linear interpolation for position
        pos = cur_pos + t * (target_pos - cur_pos)

        # Spherical linear interpolation (slerp) for orientation
        orn = p.getQuaternionSlerp(cur_orn, target_orn, t)

        # IK
        joint_poses = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            pos.tolist(),
            list(orn),
            lowerLimits=[-3.14] * num_joints,
            upperLimits=[3.14] * num_joints,
            jointRanges=[3.14] * num_joints,
            restPoses=[0] * num_joints,
        )

        # Apply
        for i in range(min(num_joints, len(joint_poses))):
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL,
                targetPosition=joint_poses[i], force=500,
            )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    print("Reached target pose.")


def mode_move_to_target(robot_id):
    """
    Generate random target poses and move the robot there repeatedly.
    """

    num_joints = p.getNumJoints(robot_id)
    end_effector_index = num_joints - 1

    # Workspace bounds for random target generation
    pos_bounds = {
        'x': (0.1, 0.5),
        'y': (-0.3, 0.3),
        'z': (0.2, 0.6),
    }

    print("\n=== Auto Random Target Mode ===")
    print("Robot will move to random targets continuously.")
    print("Press Ctrl+C to stop and return to menu.\n")

    try:
        while True:
            # Get current pose
            state = p.getLinkState(robot_id, end_effector_index)

            # Generate random target position within workspace
            target_pos = [
                random.uniform(*pos_bounds['x']),
                random.uniform(*pos_bounds['y']),
                random.uniform(*pos_bounds['z']),
            ]
            # Keep current orientation
            target_orn = list(state[1])

            print(f"Random target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

            move_to_pose(robot_id, target_pos, target_orn)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nReturning to menu...")


def main():
    """
    Interactive console: Mode 1 for keyboard control, Mode 2 for target-pose motion.
    """

    robot_id = load_robot()

    while True:
        print("\n" + "=" * 40)
        print("  ARX X5 Robot Control")
        print("=" * 40)
        print("  1 - Keyboard control (6-DOF)")
        print("  2 - Move to target pose")
        print("  q - Quit")
        print("=" * 40)

        choice = input("Select mode: ").strip().lower()

        if choice == '1':
            keyboard_control(robot_id)
        elif choice == '2':
            mode_move_to_target(robot_id)
        elif choice == 'q':
            break
        else:
            print("Invalid choice")

    p.disconnect()


if __name__ == "__main__":
    main()
