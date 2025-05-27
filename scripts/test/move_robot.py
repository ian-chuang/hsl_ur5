from hsl_ur5.robot.ur5 import UR5
from hsl_ur5.control.compliance_control import ComplianceControlConfig

def main():

    config = ComplianceControlConfig()
    config.damping_scaling = 30.0
    config.stiffness_params = [500, 500, 500, 15,15,15]
    config.max_spring_wrench = [30, 30, 30, 8,8,8]
    config.compliance_vector = [1,1,1,1,1,1]
    config.debug=False
    config.validate()

    robot = UR5(robot_ip="172.22.22.2", ft_sensor_ip="172.22.22.3", gripper_port="/dev/ttyUSBGripper", config=config)

    robot.movel([0.07, -0.35, 0.12, 0, 3.14, 0], 0.1, 0.1)
    print("Done!")


if __name__ == "__main__":
    main()