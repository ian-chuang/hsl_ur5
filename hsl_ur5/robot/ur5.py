from ati_axia80_ethernet_python import ForceTorqueSensorDriver
from robotiq_gripper_python import RobotiqGripper
import rtde_receive
import rtde_control
import time
import numpy as np
import multiprocessing
from hsl_ur5.control.compliance_control import ComplianceController, ComplianceControlConfig
from hsl_ur5.utils.math_utils import (
    spatial_vector_from_transform,
    transform_from_spatial_vector,
    to_torch,
    torch_to_np,
)
import torch
import logging

def compliance_control_loop(
    robot_ip,
    config: ComplianceControlConfig,
    raw_wrench_lock,
    raw_wrench,
    zero_wrench_flag,
    state_lock,
    state_wrench,
    state_tcp_pose,
    state_tcp_speed,
    state_q,
    state_qd,
    command_lock,
    command_pose,
    command_wrench,
    stop_event
):
    # Setting up the logger
    logger = logging.getLogger('compliance_control_loop')
    logger.setLevel(logging.DEBUG if config.debug else logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if config.debug else logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(message)s')

    # Add formatter to console handler
    ch.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(ch)

    logger.debug("[compliance_control_loop] Starting compliance control loop")

    logger.debug("[compliance_control_loop] FT sensor started")

    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_c.setTcp(torch_to_np(spatial_vector_from_transform(config.flange_to_tcp_frame)))

    logger.debug("[compliance_control_loop] RTDE interfaces connected")

    controller = ComplianceController(config, 1, torch.device("cpu"))
    controller.reset()

    logger.debug("[compliance_control_loop] Compliance controller initialized")

    try:
        logger.debug("[compliance_control_loop] Entering control loop")

        while not stop_event.is_set():
            t_start = rtde_c.initPeriod()

            base_to_tcp_frame = transform_from_spatial_vector(to_torch(rtde_r.getTargetTCPPose()))

            with raw_wrench_lock:
                wrench_at_sensor = np.array(raw_wrench[:]).copy()

            if np.isnan(wrench_at_sensor).any() or np.isinf(wrench_at_sensor).any():
                raise ValueError("Invalid wrench data from sensor")
            
            wrench_at_sensor = to_torch(wrench_at_sensor)

            if zero_wrench_flag.value:
                controller.zero_ft_sensor()
                zero_wrench_flag.value = False

            with command_lock:
                cmd_pose = np.array(command_pose[:]).copy()
                cmd_wrench = np.array(command_wrench[:]).copy()

            if np.isnan(cmd_pose).any() or np.isinf(cmd_pose).any():
                compliance_to_target_tcp_frame = torch.inverse(config.base_to_compliance_frame) @ base_to_tcp_frame
            else:
                compliance_to_target_tcp_frame = transform_from_spatial_vector(to_torch(cmd_pose))

            if np.isnan(cmd_wrench).any() or np.isinf(cmd_wrench).any():
                target_wrench_at_compliance = torch.zeros((1, 6))
            else:
                target_wrench_at_compliance = to_torch(cmd_wrench)

            controller.set_command(compliance_to_target_tcp_frame, target_wrench_at_compliance)

            vel_target_base_tcp, wrench_at_flange = controller.compute(base_to_tcp_frame, wrench_at_sensor)

            vel_target_base_tcp = torch_to_np(vel_target_base_tcp)
            wrench_at_flange = torch_to_np(wrench_at_flange)

            # Command the robot
            rtde_c.speedL(vel_target_base_tcp, 150, config.step_time)

            with state_lock:
                state_wrench[:] = wrench_at_flange.copy()
                state_tcp_pose[:] = np.array(rtde_r.getTargetTCPPose()).copy()
                state_tcp_speed[:] = np.array(rtde_r.getTargetTCPSpeed()).copy()
                state_q[:] = np.array(rtde_r.getTargetQ()).copy()
                state_qd[:] = np.array(rtde_r.getTargetQd()).copy()

            rtde_c.waitPeriod(t_start)
    
    except KeyboardInterrupt:
        logger.debug("[compliance_control_loop] Received keyboard interrupt. Exiting compliance control loop.")
    except Exception as e:
        logger.warning(f"WARNING, error in compliance control loop: {e}")
    
    logger.info("[compliance_control_loop] Exiting compliance control loop. Stopping RTDE interfaces...")
    rtde_c.speedStop()
    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()


class UR5:
    def __init__(self, robot_ip, ft_sensor_ip, gripper_port, config: ComplianceControlConfig):
        self.robot_ip = robot_ip
        self.ft_sensor_ip = ft_sensor_ip
        self.gripper_port = gripper_port
        self.config = config

        self.zero_wrench_flag = multiprocessing.Value('b', True)

        self.state_lock = multiprocessing.Lock()
        self.state_wrench = multiprocessing.Array('d', [float('nan')]*6)
        self.state_tcp_pose = multiprocessing.Array('d', [float('nan')]*6)
        self.state_tcp_speed = multiprocessing.Array('d', [float('nan')]*6)
        self.state_q = multiprocessing.Array('d', [float('nan')]*6)
        self.state_qd = multiprocessing.Array('d', [float('nan')]*6)

        self.command_lock = multiprocessing.Lock()
        self.command_pose = multiprocessing.Array('d', [float('nan')]*6)
        self.command_wrench = multiprocessing.Array('d', [float('nan')]*6)

        self.stop_event = multiprocessing.Event()

        self.thread = None

        self.ft_sensor = ForceTorqueSensorDriver(ft_sensor_ip)
        self.ft_sensor.start()

        self.gripper = RobotiqGripper(comport=gripper_port)
        self.gripper.start()
        self.gripper.move(pos=0, vel=255, force=255, block=True)

        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_c.setTcp(torch_to_np(spatial_vector_from_transform(config.flange_to_tcp_frame)))

    def __del__(self):
        self.shutdown()

    # make a decorator that checks not in compliance control mode
    def check_compliance_off(func):
        def wrapper(self, *args, **kwargs):
            if self.thread is not None:
                raise RuntimeError("Operation not allowed while in compliance control mode")
            return func(self, *args, **kwargs)
        return wrapper
    
    def check_compliance_on(func):
        def wrapper(self, *args, **kwargs):
            if self.thread is None:
                raise RuntimeError("Operation not allowed while not in compliance control mode")
            return func(self, *args, **kwargs)
        return wrapper
    
    def is_compliance_on(self):
        return self.thread is not None
    
    def shutdown(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
        
        if self.rtde_c.isConnected():
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()

        if self.rtde_r.isConnected():
            self.rtde_r.disconnect()

        self.ft_sensor.stop()
        self.gripper.shutdown()

    @check_compliance_off
    def start_compliance(self):

        if self.rtde_c.isConnected():
            self.rtde_c.disconnect()

        if self.rtde_r.isConnected():
            self.rtde_r.disconnect()

        self.command_wrench = multiprocessing.Array('d', [float('nan')]*6)

        self.thread = multiprocessing.Process(target=compliance_control_loop, args=(
            self.robot_ip,
            self.config,
            self.ft_sensor.data_lock,
            self.ft_sensor.force_torque_data,
            self.zero_wrench_flag,
            self.state_lock,
            self.state_wrench,
            self.state_tcp_pose,
            self.state_tcp_speed,
            self.state_q,
            self.state_qd,
            self.command_lock,
            self.command_pose,
            self.command_wrench,
            self.stop_event,
        ))
        self.thread.start()

        while np.isnan(self.state_wrench[:]).any():
            time.sleep(self.config.step_time)

    @check_compliance_on
    def stop_compliance(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event.clear()

        self.reset_compliance_command()

        if not self.rtde_c.isConnected():
            self.rtde_c.reconnect()

        if not self.rtde_r.isConnected():
            self.rtde_r.reconnect()

    @check_compliance_on
    def set_compliance_command(self, command_pose, command_wrench):
        with self.command_lock:
            self.command_pose[:] = command_pose
            self.command_wrench[:] = command_wrench
    
    def reset_compliance_command(self):
        with self.command_lock:
            self.command_pose[:] = [float('nan')]*6
            self.command_wrench[:] = [float('nan')]*6


    @check_compliance_on
    def zero_ft_sensor(self):
        self.zero_wrench_flag.value = True

    @check_compliance_on
    def get_state(self):
        with self.state_lock:
            state = {
                "wrench": np.array(self.state_wrench[:]),
                "tcp_pose": np.array(self.state_tcp_pose[:]),
                "tcp_speed": np.array(self.state_tcp_speed[:]),
                "q": np.array(self.state_q[:]),
                "qd": np.array(self.state_qd[:]),
            }
        return state
    
    @check_compliance_off
    def get_tcp_pose(self):
        return self.rtde_r.getActualTCPPose()
    
    @check_compliance_off
    def get_q(self):
        return self.rtde_r.getActualQ()
    
    @check_compliance_off
    def movej(self, q, v=0.1, a=0.1):
        self.rtde_c.moveJ(q, v, a)

    @check_compliance_off
    def movel(self, pose, v=0.1, a=0.1):
        self.rtde_c.moveL(pose, v, a)

    def move_gripper(self, pos, vel=1, force=1, block=False):
        pos = int(pos * 255.0)
        pos = np.clip(pos, 0, 255)
        vel = int(vel * 255.0)
        vel = np.clip(vel, 0, 255)
        force = int(force * 255.0)
        force = np.clip(force, 0, 255)
        self.gripper.move(pos=pos, vel=vel, force=force, block=block)

    def get_gripper_q(self):
        return self.gripper.get_pos() / 255.0
    
    def get_target_gripper_q(self):
        return self.gripper.get_req_pos() / 255.0
    
    def get_raw_wrench(self):
        return self.ft_sensor.get_wrench()
    
    

def main():

    config = ComplianceControlConfig()
    config.damping_scaling = 30.0
    config.stiffness_params = [100, 100, 100, 1, 1, 1]
    config.max_spring_wrench = [20, 20, 20, 2, 2, 2]
    config.debug=True
    config.validate()

    robot = UR5(robot_ip="172.22.22.2", ft_sensor_ip="172.22.22.3", gripper_port="/dev/ttyUSBGripper", config=config)

    robot.movej([1.57, -1.57, 1.57, -1.57, -1.57, 0.0], 0.1, 0.1)
    current_pose = robot.get_tcp_pose()
    
    robot.start_compliance()

    for i in range(50):

        for i in range(100):
            current_pose[2] += 0.001
            robot.set_compliance_command(
                command_pose=current_pose,
                command_wrench=np.zeros(6),
            )
            time.sleep(0.1)

        for i in range(100):
            current_pose[2] -= 0.001
            robot.set_compliance_command(
                command_pose=current_pose,
                command_wrench=np.zeros(6),
            )
            time.sleep(0.1)
        

        
if __name__ == "__main__":
    main()


