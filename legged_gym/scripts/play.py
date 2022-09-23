# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from multiprocessing.context import ForkContext
from re import I
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymtorch
import csv

from legged_gym.scripts.predictor import MLP

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True #def False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    _root_tensor = env.gym.acquire_actor_root_state_tensor(env.sim)
    root_tensor = gymtorch.wrap_tensor(_root_tensor)

    # root_positions = root_tensor[:, 0:3]
    root_orientations = root_tensor[:, 3:7]
    root_linvels = root_tensor[:, 7:10]
    root_angvels = root_tensor[:, 10:13]
    oldvel = torch.zeros(root_linvels.size(), device=env.device, dtype=torch.float)

    # f = open('write1_side_i2.csv', 'a', newline='')
    # wr = csv.writer(f)

    # f2 = open('write2_side_i2.csv', 'a', newline='')
    # wr2 = csv.writer(f2)

    # f3 = open('write3_side_i2.csv', 'a', newline='')
    # wr3 = csv.writer(f3)
    init = False
    ACTIONS = ['STOP', 'SLOW DOWN', 'NOISE', 'FASTER']

    for i in range(10*int(env.max_episode_length)):
        # with open(f'./data/imu.txt', 'a') as f:
        #     f.write(f'{root_tensor[0,3:].tolist()}, {env.force[0].tolist()}\n')
        # print(env.force[0].tolist())
        force = env.force * env.push_duration * env.dt
        root_linacc = (root_linvels - oldvel) / env.dt

        if not env.zero and not init:
            init = True
            print('impulse applied: ', force)
            if -600. <= force[0,0] <= -300.:
                print('GT: STOP')
            elif -300. < force[0,0] <= -50.:
                print('GT: SLOW DOWN')
            elif 300. <= force[0,0] <= 900.: # previous: lower bound 100, curr 300
                print('GT: FASTER')
            else:
                print('GT: NOISE')

        if env.zero:
            init = False

        if env.robot_action != 2:
            print(ACTIONS[env.robot_action])

        # print(ACTIONS[env.robot_action])
            
        imu = torch.hstack([root_orientations, root_angvels, root_linacc, env.dof_pos, env.dof_vel])

        # if i > 50 and not env.zero: # only record when pushed
        # if i > 30:
            # info = torch.hstack([imu, force])

            # dd = torch.hstack([imu, env.force])
            # wr.writerows(imu.tolist() + env.force.tolist())
            # wr.writerows(dd.tolist())

            # wr.writerow(root_tensor[0,3:].tolist() + env.force[0].tolist())
            # wr2.writerow(root_tensor[1,3:].tolist() + env.force[1].tolist())
            # wr3.writerow(root_tensor[2,3:].tolist() + env.force[2].tolist())

            # wr.writerow(imu.tolist() + env.force[0].tolist())
            # wr2.writerow(imu.tolist() + env.force[1].tolist())
            # wr3.writerow(imu.tolist() + env.force[2].tolist())
            
            # info = torch.hstack([imu, force[:,1].unsqueeze(1)])
            # wr.writerow(info[0].tolist())
            # wr2.writerow(info[1].tolist())
            # wr3.writerow(info[2].tolist())

        # print(env.dof_vel)
        # if not env.zero:
        #     # print(f'Predicted: {env.predicted_force}, Target: {env.force[0,0]}')
        #     print('Impulse: ', force)

        # print(f'Predicted: {env.pred}, Target: {force}')
        # print(env.dt)
        # print(env.sim_params.dt)

        # print(root_angacc[0])
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
            # f.close()
            return

        env.gym.refresh_actor_root_state_tensor(env.sim)
    # f.close()
    # f2.close()
    # f3.close()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
