import numpy as np
import torch
import time
from hsl_ur5.robot.ur5 import UR5
from hsl_ur5.control.compliance_control import ComplianceControlConfig
from hsl_ur5.input.joystick import Joystick
import argparse
from hsl_ur5.utils.math_utils import (
    transform_from_spatial_vector, 
    spatial_vector_from_transform, 
    to_torch, 
    torch_to_np,
)

def apply_delta(delta_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    delta_transform = transform_from_spatial_vector(to_torch(delta_pose))
    current_transform = transform_from_spatial_vector(to_torch(current_pose))
    current_translation = torch.eye(4).unsqueeze(0)
    current_translation[:, :3, 3] = current_transform[:, :3, 3]
    current_rotation = torch.eye(4).unsqueeze(0)
    current_rotation[:, :3, :3] = current_transform[:, :3, :3]
    # calculate new ee pose (relative to base_link)
    to_t = current_translation @ delta_transform @ current_rotation
    return torch_to_np(spatial_vector_from_transform(to_t))

def map_range(x, in_min, in_max, out_min, out_max):
    # clip x to be within in_min and in_max
    x = np.clip(x, in_min, in_max)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def main(args): 
    # setup hardware
    joystick = Joystick()

    config = ComplianceControlConfig()
    config.damping_scaling = 30.0
    config.stiffness_params = [400, 400, 400, 10,10,10]
    config.max_spring_wrench = [24, 24, 24, 4,4,4]
    config.compliance_vector = [1,1,1,1,1,1]
    config.debug=False
    config.validate()
    robot = UR5(robot_ip=args.robot_ip, ft_sensor_ip=args.ft_sensor_ip, gripper_port=args.gripper_port, config=config)
    robot.movej([1.57, -1.57, 1.57, -1.57, -1.57, 0.0], 0.1, 0.1)

    min_rumble_force = 2
    max_rumble_force = 20

    # start data collection
    try:
        if not robot.is_compliance_on(): robot.start_compliance()
        while True:
            start_time = time.time()
            joystick.update()
            pose_delta = joystick.get_twist(max_translation=0.05, max_rotation=0.3, max_z_rotation=0.5)
            gripper_delta = joystick.get_gripper_delta(max_gripper_delta=0.1)
            if joystick.is_start_pressed():
                break

            current_pose = robot.get_state()["tcp_pose"]
            current_gripper = robot.get_gripper_q()

            command_pose = apply_delta(pose_delta, current_pose)
            command_gripper = current_gripper + gripper_delta

            robot.set_compliance_command(
                command_pose=command_pose,
                command_wrench=np.zeros(6)
            )
            robot.move_gripper(command_gripper, vel=0.1, force=0.2, block=False)

            force_mag = np.linalg.norm(robot.get_state()['wrench'][0:3])
            rumble_val = map_range(force_mag, min_rumble_force, max_rumble_force, 0, 1)
            joystick.rumble(val=rumble_val)
            time.sleep(max(0, 1/args.fps - (time.time() - start_time)))
        robot.stop_compliance()

    except KeyboardInterrupt:
        print("KeyboardInterrupt. Shutting down...")
        
    # cleanup
    robot.shutdown()
    joystick.close()


if __name__ == "__main__":
    # make args for "save" "resume" "num-episodes" and "repo-id"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--fps", type=int, default=30)
    argparser.add_argument("--robot-ip", type=str, default="172.22.22.2")
    argparser.add_argument("--ft-sensor-ip", type=str, default="172.22.22.3")
    argparser.add_argument("--gripper-port", type=str, default="/dev/ttyUSBGripper")
    args = argparser.parse_args()  
    main(args)