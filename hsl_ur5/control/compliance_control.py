from hsl_ur5.utils.math_utils import (
    transform_from_spatial_vector,
    adm_rotate_wrench_in_frame,
    wrench_trans,
    pose_error,
    wrench_dead_band_smooth,
    adm_rotate_velocity_in_frame,
    to_torch,
)
import torch

class ComplianceControlConfig:
    # general params
    step_time=1.0/125
    flange_to_tcp_frame = [0, 0, 0.3, 0, 0, 0]
    debug = False

    # ft sensor params
    gravity_compensation = True
    base2grav = [0, 0, 0, 0, 0, 0]
    flange2sensor = [0, 0, 0.02, 0, 0, 0]
    sensor2cog = [0, 0, 0, 0, 0, 0]
    cog_pos = [0, 0, 0.085]
    cog_force = 15

    # compliance control params
    mass_scaling=0.5
    damping_scaling=0.5
    mass_list=[22.5, 22.5, 22.5, 1, 1, 1]
    damping_list=[25, 25, 25, 2, 2, 2]
    base_to_compliance_frame=[0, 0, 0, 0, 0, 0]
    tool_flange_to_compliance_center=[0, 0, 0, 0, 0, 0]
    dead_band=[2, 0.15, 2, 0.15]
    compliance_vector=[1, 1, 1, 1, 1, 1]
    stiffness_params=[10, 10, 10, 0.1, 0.1, 0.1]
    max_spring_wrench=[25, 25, 25, 5, 5, 5]

    def validate(self, device=torch.device("cpu")):
        assert self.step_time > 0, "step_time must be positive"
        assert len(self.flange_to_tcp_frame) == 6, "flange2tcp must be a 6D vector"
        assert self.debug in [True, False], "debug must be a boolean"
        assert self.gravity_compensation in [True, False], "gravity_compensation must be a boolean"
        assert len(self.base2grav) == 6, "base2grav must be a 6D vector"
        assert len(self.flange2sensor) == 6, "flange2sensor must be a 6D vector"
        assert len(self.sensor2cog) == 6, "sensor2cog must be a 6D vector"
        assert len(self.cog_pos) == 3, "cog_pos must be a 3D vector"
        assert self.cog_force > 0, "cog_force must be positive"
        assert self.mass_scaling > 0, "mass_scaling must be positive"
        assert self.damping_scaling > 0, "damping_scaling must be positive"
        assert len(self.mass_list) == 6, "mass_list must be a 6D vector"
        assert len(self.damping_list) == 6, "damping_list must be a 6D vector"
        assert len(self.base_to_compliance_frame) == 6, "base_to_compliance_frame must be a 6D vector"
        assert len(self.tool_flange_to_compliance_center) == 6, "tool_flange_to_compliance_center must be a 6D vector"
        assert len(self.dead_band) == 4, "dead_band must be a 4D vector"
        assert len(self.compliance_vector) == 6, "compliance_vector must be a 6D vector"
        assert len(self.stiffness_params) == 6, "stiffness_params must be a 6D vector"
        assert len(self.max_spring_wrench) == 6, "max_spring_wrench must be a 6D vector"

        self.flange_to_tcp_frame = transform_from_spatial_vector(to_torch(self.flange_to_tcp_frame))

        self.base2grav = transform_from_spatial_vector(to_torch(self.base2grav))
        self.flange2sensor = transform_from_spatial_vector(to_torch(self.flange2sensor))
        self.sensor2cog = transform_from_spatial_vector(to_torch(self.sensor2cog))
        self.cog_pos = to_torch(self.cog_pos)

        self.mass_list = to_torch(self.mass_list)
        self.damping_list = to_torch(self.damping_list)
        self.base_to_compliance_frame = transform_from_spatial_vector(to_torch(self.base_to_compliance_frame))
        self.tool_flange_to_compliance_center = transform_from_spatial_vector(to_torch(self.tool_flange_to_compliance_center))
        self.compliance_vector = to_torch(self.compliance_vector)
        self.stiffness_params = to_torch(self.stiffness_params)
        self.max_spring_wrench = to_torch(self.max_spring_wrench)


class ComplianceController:
    def __init__(self, config: ComplianceControlConfig, num_envs: int, device: torch.device):
        self.config = config
        self.num_envs = num_envs
        self.device = device

        self.reset()

    def reset(self):
        self.x_e = torch.zeros(self.num_envs, 6, device=self.device)
        self.dx_e = torch.zeros(self.num_envs, 6, device=self.device)
        self.ddx_e = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_ddx_e = self.ddx_e.clone()
        self.last_dx_e = self.dx_e.clone()
        self.last_x_e = self.x_e.clone()

        self.zero_wrench_flag = True
        self.zero_wrench = torch.zeros(self.num_envs, 6, device=self.device)

    def set_command(self, compliance_to_target_tcp_frame, target_wrench_at_compliance):
        """
        cmd_pose (N, 4, 4)
        cmd_wrench (N, 6)
        """
        self.compliance_to_target_tcp_frame = compliance_to_target_tcp_frame
        self.target_wrench_at_compliance = target_wrench_at_compliance

    def zero_ft_sensor(self):
        self.zero_wrench_flag = True

    def apply_gravity_compensation(self, sensor_wrench, base_to_tcp_frame):
        grav2sensor = torch.inverse(self.config.base2grav) @ base_to_tcp_frame @ torch.inverse(self.config.flange_to_tcp_frame) @ self.config.flange2sensor
        grav2cog = grav2sensor  @ self.config.sensor2cog

        # create compensation wrench
        grav_comp_wrench = torch.zeros((sensor_wrench.shape[0], 6))
        r0 = grav2cog[:, 0:3, 0:3]
        grav_comp_wrench[:, 2] = -self.config.cog_force
        grav_comp_wrench[:, 3:6] = torch.cross(torch.matmul(r0, self.config.cog_pos.unsqueeze(2)).squeeze(2), grav_comp_wrench[:, 0:3], dim=1)

        # rotate wrench to sensor frame
        grav_comp_wrench = adm_rotate_wrench_in_frame(torch.inverse(grav2sensor), grav_comp_wrench)

        # add compensation wrench to sensor wrench
        sensor_wrench = sensor_wrench - grav_comp_wrench

        return sensor_wrench

    def compute(self, base_to_tcp_frame, wrench_at_sensor):
        """
        ee_pose (N, 4, 4)
        flange_wrench (N, 6)
        """

        # apply grav comp
        if self.config.gravity_compensation:
            wrench_at_sensor = self.apply_gravity_compensation(wrench_at_sensor, base_to_tcp_frame)

        # zero wrench if needed
        if self.zero_wrench_flag:
            self.zero_wrench_flag = False
            self.zero_wrench = wrench_at_sensor.clone()
        wrench_at_sensor = wrench_at_sensor - self.zero_wrench

        # transform sensor_wrench to flange frame
        wrench_at_flange = wrench_trans(torch.inverse(self.config.flange2sensor), wrench_at_sensor)

        # Apply dead band and smooth
        wrench_at_flange = wrench_dead_band_smooth(
            wrench_at_flange,
            self.config.dead_band[0], self.config.dead_band[1], self.config.dead_band[2], self.config.dead_band[3]
        )

        # Compute the wrench at the compliance center
        wrench_at_compliance_center = wrench_trans(self.config.tool_flange_to_compliance_center, wrench_at_flange)

        # compute transforms
        T_compliance_center_to_tcp = torch.matmul(torch.inverse(self.config.tool_flange_to_compliance_center), self.config.flange_to_tcp_frame)
        T_compliance_center_to_base = torch.matmul(T_compliance_center_to_tcp, torch.inverse(base_to_tcp_frame))
        T_compliance_frame_to_compliance_center = torch.inverse(torch.matmul(T_compliance_center_to_base, self.config.base_to_compliance_frame))

        # calc force and torque error in compliance frame
        wrench_at_compliance = adm_rotate_wrench_in_frame(T_compliance_frame_to_compliance_center, wrench_at_compliance_center)

        # cancel out specific force and torque directions
        wrench_at_compliance = wrench_at_compliance * self.config.compliance_vector

        compliance_to_tcp_frame = torch.matmul(torch.inverse(self.config.base_to_compliance_frame), base_to_tcp_frame)

        # calc spring wrench for compliance
        pose_err = pose_error(compliance_to_tcp_frame, self.compliance_to_target_tcp_frame)
        spring_wrench = pose_err * self.config.stiffness_params
        spring_wrench = torch.clip(spring_wrench, -self.config.max_spring_wrench, self.config.max_spring_wrench)

        self.ddx_e = 1/(self.config.mass_list*self.config.mass_scaling) * \
            (
                (wrench_at_compliance - self.target_wrench_at_compliance) - spring_wrench - 
                (self.dx_e * self.config.damping_list * self.config.damping_scaling)
            )
        self.dx_e = (self.config.step_time * 0.5) * (self.ddx_e + self.last_ddx_e) + self.last_dx_e
        self.x_e = (self.config.step_time * 0.5) * (self.dx_e + self.last_dx_e) + self.last_x_e

        self.last_ddx_e = self.ddx_e.clone()
        self.last_dx_e = self.dx_e.clone()
        self.last_x_e = self.x_e.clone()

        vel_target_compliance = self.dx_e
        vel_target_tcp = adm_rotate_velocity_in_frame(torch.inverse(compliance_to_tcp_frame), vel_target_compliance)
        vel_target_base = adm_rotate_velocity_in_frame(base_to_tcp_frame, vel_target_tcp)

        return vel_target_base, wrench_at_flange

