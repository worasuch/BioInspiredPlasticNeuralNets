# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
from abc import abstractmethod

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import tensor_clamp, torch_rand_float, unscale
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, get_euler_xyz
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.utils.ant_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *

# import omni.replicator.isaac as dr

class LocomotionTask(RLTask):
    def __init__(self, name, env, offset=None) -> None:
        print('-__init__LocomotionTask-')

        LocomotionTask.update_config(self)

        RLTask.__init__(self, name, env)
        return

    def update_config(self):
        print('-update_config-')
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.simulation_step = 0
        self.constant = 1.0
        self.rand_forces_min = self._cfg['randomization']['forces'][0]
        self.rand_forces_max = self._cfg['randomization']['forces'][1]

    @abstractmethod
    def set_up_scene(self, scene) -> None:
        pass

    @abstractmethod
    def get_robot(self):
        pass

    def get_slope_terrain(self, create_mesh=True):
        # self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        # self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.terrain = Terrain(self._cfg['terrain'], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size-7.5, 0.0])
        if create_mesh:
            print('-add_terrain_to_stage-')
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)

    def _create_trimesh(self, create_mesh=True):
        print('-_create_trimesh-')
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        # self.terrain = Terrain(self._cfg['terrain'], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        # position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
        position = torch.tensor([-self.terrain.border_size-8, -self.terrain.border_size-8, 0.0])
        if create_mesh:
            print('-add_terrain_to_stage-')
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def get_terrain(self, create_mesh=True):
        print('-get_terrain-')
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if self._cfg['terrain']['type'] == 'slope':
            self.get_slope_terrain(create_mesh=create_mesh)
        else:
            if not self.curriculum:
                self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
            self.terrain_levels = torch.randint(
                0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.randint(
                0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device
            )
            self._create_trimesh(create_mesh=create_mesh)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
    
    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        velocities = self._robots.get_velocities(clone=False)
        velocity = velocities[:, 0:3]
        ang_velocity = velocities[:, 3:6]
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        # force sensors attached to the feet
        sensor_force_torques = self._robots.get_measured_joint_forces(joint_indices=self._sensor_indices)
        # print('fc shape: ', sensor_force_torques.shape)
        sensor_fc = torch.norm(sensor_force_torques[:, :, 0:3], dim=-1) > 1.0
        # print('sensor_fc shape: ', sensor_fc[0])

        # IMU
        roll, pitch, yaw = get_euler_xyz(torso_rotation)
        
        # if self.simulation_step > 250:
        #     self.constant = 0.0
        self.simulation_step += 1
        # print('dof_names: ', self._robots.dof_names)
        (
            self.obs_buf_full[:],
            self.potentials[:],
            self.prev_potentials[:],
            self.up_vec[:],
            self.heading_vec[:],
        ) = get_observations(
            torso_position,
            torso_rotation,
            velocity,
            ang_velocity,
            dof_pos,
            dof_vel,
            self.targets,
            self.potentials,
            self.dt,
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.dof_vel_scale,
            sensor_force_torques,
            self._num_envs,
            self.contact_force_scale,
            self.actions,
            self.angular_velocity_scale,
            self.constant,
        )

        ### Extract only some observation for model inputs
        ### Example 2D tensors
        tensor_pos = self.obs_buf_full[:, 12:30].clone().detach() * 0.5
        tensor_fc = sensor_fc.clone().detach()
        imu = torch.cat((normalize_angle(roll).unsqueeze(-1),
                        normalize_angle(pitch).unsqueeze(-1),
                        normalize_angle(yaw).unsqueeze(-1),),dim=-1,)
        tensor_imu = imu.clone().detach()
        # print('tensor_pos shape: ', tensor_pos.shape)
        # print('tensor_fc shape: ', tensor_fc.shape)
        # print('tensor_imu shape: ', tensor_imu.shape)
        ### Concatenating along axis=1 (columns)
        self.obs_buf_custom = torch.cat((tensor_pos, tensor_fc, tensor_imu), dim=-1)
        self.obs_buf = self.obs_buf_custom.clone().detach()
        # print('Selected observation: ', self.obs_buf_custom.shape)
        # observations = {self._robots.name: {"obs_buf": self.obs_buf_custom}}

        # Original observation return
        observations = {self._robots.name: {"obs_buf": self.obs_buf}}

        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # print('reset_env_ids: ', reset_env_ids)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        # forces = self.actions * self.joint_gears * self.power_scale

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        # apply action lowpass
        self.actions = 0.1 * self.actions + 0.9 * self.prev_actions
        self.prev_actions = self.actions

        # applies joint torques
        # self._robots.set_joint_efforts(forces, indices=indices)

        # joint target position command
        # self.actions *= 180.0/math.pi
        self._robots.set_joint_position_targets(self.actions, indices=indices)

        if self._dr_randomizer.randomize:
            self.dr.physics_view.step_randomization(reset_env_ids)

        # apply an external force to all the rigid bodies to the indicated values.
        # Since there are 5 envs, the inertias are repeated 5 times
        # self.physics_ants.apply_forces(self.forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions and velocities
        dof_pos = torch_rand_float(-0.2, 0.2, (num_resets, self._robots.num_dof), device=self._device)
        dof_pos[:] = tensor_clamp(self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper)
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._robots.num_dof), device=self._device)

        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        # root_pos[:, 2] += 0.5 # move robot up in z-axis
        # move robot to a new level in y-axis
        
        # root_pos[:, 1] += torch.randint_like(root_pos[:, 1], 0, 1)*15
        # root_pos[:, 1] += torch.randint_like(root_pos[:, 1], 2)*15
        # print('root_pos_y: ', torch.randint_like(root_pos[:, 1], 0, 1)*15)

        # apply resets
        self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        to_target = self.targets[env_ids] - self.initial_root_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def post_reset(self):
        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(
            self.num_envs
        )
        self.prev_potentials = self.potentials.clone()
        self.obs_buf_full = torch.zeros((self._num_envs, self._num_observations), device=self._device, dtype=torch.float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        # Randomize Force act on the robot
        a = torch.Tensor([0.0, 0.0, 0.0])
        self.forces = torch.tile(a, (self._num_envs, 1)).cuda()
        self.forces[:,0] = torch.rand((self._num_envs)).uniform_(self.rand_forces_min, self.rand_forces_max)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = calculate_metrics(
            self.obs_buf_full,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.termination_height,
            self.death_cost,
            self._robots.num_dof,
            self.get_dof_at_limit_cost(),
            self.alive_reward_scale,
            self.motor_effort_ratio,
        )

    def is_done(self) -> None:
        self.reset_buf[:] = is_done(
            self.obs_buf_full, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def get_observations(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    targets,
    potentials,
    dt,
    inv_start_rot,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    sensor_force_torques,
    num_envs,
    contact_force_scale,
    actions,
    angular_velocity_scale,
    constant,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            vel_loc,
            angvel_loc * angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1)*constant,
            normalize_angle(roll).unsqueeze(-1)*constant,
            normalize_angle(angle_to_target).unsqueeze(-1)*constant,
            up_proj.unsqueeze(-1),
            heading_proj.unsqueeze(-1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,
            actions,
        ),
        dim=-1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec


@torch.jit.script
def is_done(obs_buf_full, termination_height, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tensor
    # reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return reset


@torch.jit.script
def calculate_metrics(
    obs_buf_full,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    death_cost,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio,
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, int, Tensor, float, Tensor) -> Tensor

    # Simple reward scheme IROS2024 ##############
    # roll, pitch, yaw = get_euler_xyz(self.torso_rotation)
    
    rew_lin_vel_x = obs_buf_full[:, 1] * 2.0
    # rew_lin_vel_y = torch.square(self.velocity[:, 1]) * -self.rew_lin_vel_y_scale
    rew_orient = torch.where(obs_buf_full[:, 10] > 0.93 , 0, -0.5)
    height_reward = torch.where(abs(obs_buf_full[:, 0] + 0.08) < 0.04 , 0.0, -0.5)
    # print('height : ', obs_buf_full[:, 0])
    # print('height_reward : ', height_reward)
    # print('yaw:  ', obs_buf_full[:, 7])
    rew_yaw = torch.where(abs(obs_buf_full[:, 7]) < 0.45 , 0.0, -0.5)
    # rew_yaw = torch.where(abs(obs_buf_full[:, 7]) < 0.45 , -abs(obs_buf_full[:, 7])/0.45, -1.0)

    total_reward = rew_lin_vel_x + rew_orient + rew_yaw + height_reward #+ gait_reward #+ rew_lin_vel_y #+ height_reward 
    ###############################################

    # heading_weight_tensor = torch.ones_like(obs_buf_full[:, 11]) * heading_weight
    # heading_reward = torch.where(obs_buf_full[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf_full[:, 11] / 0.8)

    # # aligning up axis of robot and environment
    # up_reward = torch.zeros_like(heading_reward)
    # up_reward = torch.where(obs_buf_full[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # # energy penalty for movement
    # actions_cost = torch.sum(actions**2, dim=-1)
    # electricity_cost = torch.sum(
    #     torch.abs(actions * obs_buf_full[:, 12 + num_dof : 12 + num_dof * 2]) * motor_effort_ratio.unsqueeze(0), dim=-1
    # )

    # # reward for duration of staying alive
    # alive_reward = torch.ones_like(potentials) * alive_reward_scale
    # progress_reward = potentials - prev_potentials

    # total_reward = (
    #     progress_reward
    #     # + alive_reward
    #     + up_reward
    #     + heading_reward
    #     # - actions_cost_scale * actions_cost
    #     # - energy_cost_scale * electricity_cost
    #     # - dof_at_limit_cost
    # )

    # adjust reward for fallen agents
    # total_reward = torch.where(
    #     obs_buf_full[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    # )
    return total_reward
