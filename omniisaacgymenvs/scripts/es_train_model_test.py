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


import datetime
import os
import gym
import hydra
import torch
from omegaconf import DictConfig
import omniisaacgymenvs
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path, get_experience
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from omniisaacgymenvs.ES.rbf_neural_net import RBFNet
from omniisaacgymenvs.ES.rbf_hebbian_neural_net_new import RBFHebbianNet
from omniisaacgymenvs.ES.rbf_LSTM_neural_net import RBFLSTMNet
from omniisaacgymenvs.ES.rbf_FF_neural_net import RBFFFNet
from omniisaacgymenvs.ES.ES_classes import OpenES
import timeit
import pickle
import copy
import numpy as np
# -----------Old code-----------------------------------------------------------------------------
# class RLGTrainer:
#     def __init__(self, cfg, cfg_dict):
#         self.cfg = cfg
#         self.cfg_dict = cfg_dict

#     def launch_rlg_hydra(self, env):
#         # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
#         # We use the helper function here to specify the environment config.
#         self.cfg_dict["task"]["test"] = self.cfg.test

#         # register the rl-games adapter to use inside the runner
#         vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
#         env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: env})

#         self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

#     def run(self, module_path, experiment_dir):
#         self.rlg_config_dict["params"]["config"]["train_dir"] = os.path.join(module_path, "runs")

#         # create runner and set the settings
#         runner = Runner(RLGPUAlgoObserver())
#         runner.load(self.rlg_config_dict)
#         runner.reset()

#         # dump config dict
#         os.makedirs(experiment_dir, exist_ok=True)
#         with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
#             f.write(OmegaConf.to_yaml(self.cfg))

#         runner.run(
#             {"train": not self.cfg.test, "play": self.cfg.test, "checkpoint": self.cfg.checkpoint, "sigma": None}
#         )
# ----------------------------------------------------------------------------------------

def get_masked_sens_loss(ind_v):
    # Start with an empty tensor
    arr = torch.tensor([]).reshape(0, 2)  # Assuming you want to concatenate 2D tensors with 2 columns
    for i in ind_v:
        if i == 0:
            tensor = torch.Tensor([[1,0]])
        elif i == 1:
            tensor = torch.Tensor([[0,1]])
        # else:
        #     tensor = torch.Tensor([[1,1]])
        arr = torch.cat((arr, tensor))
    return arr

def get_masked_sens_loss_v2(ind_v):
    # Start with an empty tensor
    # mask IMU sensor and joint contact sensor
    arr = torch.tensor([]).reshape(0, 15)  # Assuming you want to concatenate 2D tensors with 2 columns
    for i in ind_v:
        if i == 0:
            tensor = torch.Tensor([[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        elif i == 1:
            tensor = torch.Tensor([[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])
        # else:
        #     tensor = torch.Tensor([[1,1]])
        arr = torch.cat((arr, tensor))
    return arr

@hydra.main(version_base=None, config_name="es_config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    # Initialize ES parameters
    POPSIZE             = cfg.num_envs
    RANK_FITNESS        = cfg.ES_params.rank_fitness
    ANTITHETIC          = cfg.ES_params.antithetic
    LEARNING_RATE       = cfg.ES_params.learning_rate
    LEARNING_RATE_DECAY = cfg.ES_params.learning_rate_decay
    SIGMA_INIT          = cfg.ES_params.sigma_init
    SIGMA_DECAY         = cfg.ES_params.sigma_decay
    LEARNING_RATE_LIMIT = cfg.ES_params.learning_rate_limit
    SIGMA_LIMIT         = cfg.ES_params.sigma_limit

    # Model
    ARCHITECTURE_NAME = cfg.model
    ARCHITECTURE_TYPE = cfg.model_type
    FF_ARCHITECTURE = cfg.FF_ARCHITECTURE
    RBF_ARCHITECTURE = cfg.RBF_ARCHITECTURE
    HEBB_ARCHITECTURE = cfg.HEBB_ARCHITECTURE
    LSTM_ARCHITECTURE = cfg.LSTM_ARCHITECTURE
    HEBB_init_wnoise = cfg.HEBB_init_wnoise
    HEBB_norm = cfg.HEBB_norm
    USE_TRAIN_RBF = cfg.USE_TRAIN_RBF
    USE_TRAIN_HEBB = cfg.USE_TRAIN_HEBB
    
    # Training parameters
    # EPOCHS = configs['Train_params']['EPOCH']
    EPOCHS = cfg.EPOCHS
    EPISODE_LENGTH_TRAIN = cfg.EPISODE_LENGTH_TRAIN
    EPISODE_LENGTH_TEST = cfg.EPISODE_LENGTH_TEST
    SAVE_EVERY = cfg.SAVE_EVERY
    USE_TRAIN_PARAM = cfg.USE_TRAIN_PARAM

    # General info
    TASK = cfg.task_name
    TEST = cfg.test
    if TEST:
        USE_TRAIN_PARAM = True
    train_ff_path = cfg.train_ff_path
    train_rbf_path = cfg.train_rbf_path
    train_hebb_path = cfg.train_hebb_path
    train_lstm_path = cfg.train_lstm_path
    collect_w_matrix = cfg.collect_w_matrix

    # Initialize model &
    if ARCHITECTURE_NAME == 'rbf_ff':
        models = RBFFFNet(popsize=POPSIZE, 
                            num_basis=RBF_ARCHITECTURE[0], 
                            num_output=RBF_ARCHITECTURE[1], 
                            ARCHITECTURE=FF_ARCHITECTURE,
                            robot=TASK,
                            )
        dir_path = 'runs_ES/'+TASK+'/rbf_ff/'
        # Use train rbf params by default
        trained_data = pickle.load(open('runs_ES/'+TASK+'/rbf/'+train_rbf_path, 'rb'))
        open_es_data = trained_data[0]
        rbf_params = open_es_data.best_param() # best_mu
        models.set_a_rbf_params(rbf_params)
    elif ARCHITECTURE_NAME == 'rbf':
        models = RBFNet(popsize=POPSIZE,
                        num_basis=RBF_ARCHITECTURE[0],
                        num_output=RBF_ARCHITECTURE[1],
                        robot=TASK,
                        motor_encode='semi-indirect',
                        )
        dir_path = 'runs_ES/'+TASK+'/rbf/'
    elif ARCHITECTURE_NAME == 'rbf_hebb':
        models = RBFHebbianNet(popsize=POPSIZE, 
                               num_basis=RBF_ARCHITECTURE[0], 
                               num_output=RBF_ARCHITECTURE[1], 
                               ARCHITECTURE=HEBB_ARCHITECTURE, 
                               mode=ARCHITECTURE_TYPE,
                               hebb_init_wnoise=HEBB_init_wnoise,
                               hebb_norm_mode=HEBB_norm,
                               robot=TASK)
        dir_path = 'runs_ES/'+TASK+'/rbf_hebb/'
        # Use train rbf params by default
        trained_data = pickle.load(open('runs_ES/'+TASK+'/rbf/'+train_rbf_path, 'rb'))
        open_es_data = trained_data[0]
        rbf_params = open_es_data.best_param() # best_mu
        models.set_a_rbf_params(rbf_params)
    elif ARCHITECTURE_NAME == 'rbf_lstm':
        models = RBFLSTMNet(popsize=POPSIZE, 
                               num_basis=RBF_ARCHITECTURE[0], 
                               num_output=RBF_ARCHITECTURE[1], 
                               ARCHITECTURE=LSTM_ARCHITECTURE,
                                robot=TASK,
                                )
        dir_path = 'runs_ES/'+TASK+'/rbf_lstm/'
        # Use train rbf params by default
        trained_data = pickle.load(open('runs_ES/'+TASK+'/rbf/'+train_rbf_path, 'rb'))
        open_es_data = trained_data[0]
        rbf_params = open_es_data.best_param() # best_mu
        models.set_a_rbf_params(rbf_params)

    n_params_a_model = models.get_n_params_a_model()

    # Initialize OpenES Evolutionary Strategy Optimizer
    solver = OpenES(n_params_a_model,
                    popsize=POPSIZE,
                    rank_fitness=RANK_FITNESS,
                    antithetic=ANTITHETIC,
                    learning_rate=LEARNING_RATE,
                    learning_rate_decay=LEARNING_RATE_DECAY,
                    sigma_init=SIGMA_INIT,
                    sigma_decay=SIGMA_DECAY,
                    learning_rate_limit=LEARNING_RATE_LIMIT,
                    sigma_limit=SIGMA_LIMIT)
    solver.set_mu(models.get_models_params().reshape(cfg.num_envs, n_params_a_model))

    # Use train rbf params
    # 1. solver 2. copy.deepcopy(models)  3. pop_mean_curve 4. best_sol_curve,
    if USE_TRAIN_PARAM:
        if cfg.model == 'ff':
            trained_data = pickle.load(open(dir_path+train_ff_path, 'rb'))
            train_params = trained_data[0].best_param()
            solver.set_mu(train_params)
        elif cfg.model == 'rbf':
            trained_data = pickle.load(open(dir_path+train_rbf_path, 'rb'))
            train_params = trained_data[0].best_param()
            solver.set_mu(train_params)
        elif cfg.model == 'rbf_hebb':
            trained_data = pickle.load(open(dir_path+train_hebb_path, 'rb'))
            train_params = trained_data[0].best_param()
            solver.set_mu(train_params)
            print('train_params number: ', len(train_params))
        elif cfg.model == 'rbf_ff':
            trained_data = pickle.load(open(dir_path+train_ff_path, 'rb'))
            train_params = trained_data[0].best_param()
            solver.set_mu(train_params)
        elif cfg.model == 'rbf_lstm':
            trained_data = pickle.load(open(dir_path+train_lstm_path, 'rb'))
            train_params = trained_data[0].best_param()
            solver.set_mu(train_params)
            print('train_params number: ', len(train_params))

    print('--- Used train RBF params ---')
    print('file_name: ', train_hebb_path)

    # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # headless = cfg.headless

    # # local rank (GPU id) in a current multi-gpu mode
    # local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # # global rank (GPU id) in multi-gpu multi-node mode
    # global_rank = int(os.getenv("RANK", "0"))
    # if cfg.multi_gpu:
    #     cfg.device_id = local_rank
    #     cfg.rl_device = f'cuda:{local_rank}'
    # enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # # select kit app file
    # experience = get_experience(headless, cfg.enable_livestream, enable_viewport, cfg.enable_recording, cfg.kit_app)

    # env = VecEnvRLGames(
    #     headless=headless,
    #     sim_device=cfg.device_id,
    #     enable_livestream=cfg.enable_livestream,
    #     enable_viewport=enable_viewport or cfg.enable_recording,
    #     experience=experience
    # )

    # # parse experiment directory
    # module_path = os.path.abspath(os.path.join(os.path.dirname(omniisaacgymenvs.__file__)))
    # experiment_dir = os.path.join(module_path, "runs", cfg.train.params.config.name)

    # # use gym RecordVideo wrapper for viewport recording
    # if cfg.enable_recording:
    #     if cfg.recording_dir == '':
    #         videos_dir = os.path.join(experiment_dir, "videos")
    #     else:
    #         videos_dir = cfg.recording_dir
    #     video_interval = lambda step: step % cfg.recording_interval == 0
    #     video_length = cfg.recording_length
    #     env.is_vector_env = True
    #     if env.metadata is None:
    #         env.metadata = {"render_modes": ["rgb_array"], "render_fps": cfg.recording_fps}
    #     else:
    #         env.metadata["render_modes"] = ["rgb_array"]
    #         env.metadata["render_fps"] = cfg.recording_fps
    #     env = gym.wrappers.RecordVideo(
    #         env, video_folder=videos_dir, step_trigger=video_interval, video_length=video_length
    #     )

    # # ensure checkpoints can be specified as relative paths
    # if cfg.checkpoint:
    #     cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
    #     if cfg.checkpoint is None:
    #         quit()

    # cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # # sets seed. if seed is -1 will pick a random one
    # from omni.isaac.core.utils.torch.maths import set_seed
    # cfg.seed = cfg.seed + global_rank if cfg.seed != -1 else cfg.seed
    # cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    # cfg_dict["seed"] = cfg.seed

    # task = initialize_task(cfg_dict, env)

    # if cfg.wandb_activate and global_rank == 0:
    #     # Make sure to install WandB if you actually use this.
    #     print()
    #     import wandb

    #     # run_name = f"{cfg.wandb_name}_{ARCHITECTURE_NAME}_{time_str}"
    #     run_name = f"{cfg.wandb_name}_{ARCHITECTURE_NAME}_{cfg.wandb_group}"

    #     wandb.init(
    #         project=cfg.wandb_project,
    #         group=cfg.wandb_group,
    #         # entity=cfg.wandb_entity,
    #         config=cfg_dict,
    #         # sync_tensorboard=False,
    #         name=run_name,
    #         # resume="allow",
    #     )

    # torch.cuda.set_device(local_rank)

    # print data on terminal
    # print('TASK', TASK)
    # print('model: ', ARCHITECTURE_NAME)
    # print('model size: ', models.architecture)
    # print('trainable parameters a model: ', models.get_n_params_a_model())
    # print('trainable parameters a model: ', len(models.get_models_params()))
    # print("Observation space is", env.observation_space)
    # print("Action space is", env.action_space)

    # ------Old code--------------------------------------
    # rlg_trainer = RLGTrainer(cfg, cfg_dict)
    # rlg_trainer.launch_rlg_hydra(env)
    # rlg_trainer.run(module_path, experiment_dir)
    # --------------------------------------------

    # ES code
    # Log data initialize
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)
    eval_curve = np.zeros(EPOCHS)
    
    # initial time to measure time of training loop 
    # initial_time = timeit.default_timer()
    # print("initial_time", initial_time)

    # Testing Loop ----------------------------------
    if TEST:
        # sample params from ES and set model params
        # solutions = solver.ask()
        # models.set_models_params(solutions)        
        
        models.set_a_model_params(train_params)
        # obs = env.reset()
        # obs['obs'] = obs['obs'][:, 7:8].repeat(1,2)
        # print("obs['obs'].shape: ", obs['obs'].shape)
        
        # IMU + force sensors attached to the feet
        # imu = obs['obs'][:, 7:10]
        # joint_forces = obs['obs'][:, 28:52].reshape(-1, 4, 6)[:, :, 0:3]
        # # joint_forces = torch.norm(joint_forces[:, :, 0:3], p=2, dim=2)
        # joint_forces = joint_forces.reshape(cfg.num_envs, 12)
        # obs['obs'] = torch.cat((imu,
        #                     joint_forces),
        #                     dim = 1)

        # Epoch rewards
        total_rewards = torch.zeros(cfg.num_envs)
        total_rewards = total_rewards.cuda()
        rew = torch.zeros(cfg.num_envs).cuda()

        # # collect_w_matrix
        # if collect_w_matrix:
        #     w1 = []
        #     w2 = []
        #     params = []
        #     action_arr = []
        #     rewards_arr = []
        # prev_actions = torch.zeros(cfg.num_envs, env.action_space.shape[0]).cuda()

        # # Randomize sensory loss
        # rand = torch.randint(0, 2, (cfg.num_envs,))
        # v = get_masked_sens_loss(rand).cuda()
        

        # rollout 
        for sim_step in range(EPISODE_LENGTH_TEST):
            actions = models.forward(obs['obs'])
            # actions = 0.3*actions + 0.7*prev_actions
            # prev_actions = actions
            # obs, reward, done, info = env.step(
            #     actions
            # )
            # Select observation

            # Duplicate yaw angle
            # obs['obs'] = obs['obs'][:, 7:8].repeat(1,2)

            # IMU + force sensors attached to the feet
            # imu = obs['obs'][:, 7:10]
            # joint_forces = obs['obs'][:, 28:52].reshape(-1, 4, 6)[:, :, 0:3]
            # # joint_forces = torch.norm(joint_forces[:, :, 0:3], p=2, dim=2)
            # joint_forces = joint_forces.reshape(cfg.num_envs, 12)
            # obs['obs'] = torch.cat((imu,
            #                     joint_forces),
            #                     dim = 1)
            # ###################################
            # randomize sensory loss of each individual
            # obs['obs'] = obs['obs'] * v
            ####################################

            # if sim_step >= 0 and sim_step < 500:
            #     # print('-{}-', sim_step)
            #     # Multiply the first column by 0.5
            #     obs['obs'][:, 0:3] *= 0.0
            #     rew += reward/EPISODE_LENGTH_TEST*100
            #     if sim_step == 499:
            #         print(rew)
            #         rew *= 0.0  
            # if sim_step >= 500 and sim_step < 1000:
            #     # print('--{}--', sim_step)
            #     # Multiply the first column by 0.5
            #     obs['obs'][:, 0] *= 0.0
            #     rew += reward/EPISODE_LENGTH_TEST*100
            #     if sim_step == 999:
            #         print(rew)
            #         rew *= 0.0  
            # if sim_step >= 1000 and sim_step < 1500:
            #     # print('---{}---', sim_step)
            #     obs['obs'][:, 1] *= 0.0
            #     rew += reward/EPISODE_LENGTH_TEST*100
            #     if sim_step == 1499:
            #         print(rew)
            #         rew *= 0.0  
            # if sim_step >= 1500 and sim_step < 2000:
            #     # obs['obs'][:, 0] *= 0.0
            #     rew += reward/EPISODE_LENGTH_TEST*100
            #     if sim_step == 1999:
            #         print(rew)
            #         rew *= 0.0
            # # if sim_step >= 2000 and sim_step < 2500:
            # #     obs['obs'][:, :] *= 0.0
            # # if sim_step >= 2500 and sim_step < 3000:
            # #     obs['obs'][:, :] *= 0.0
            
            # total_rewards += reward/EPISODE_LENGTH_TEST*100
        #     if collect_w_matrix:
        #         weight = models.get_hebb_weights()
        #         param = models.get_models_params()
        #         w1.append(weight[0].cpu().numpy())
        #         w2.append(weight[1].cpu().numpy())
        #         params.append(param.cpu().numpy())
        #         action_arr.append(actions.cpu().numpy())
        #         rewards_arr.append(reward.cpu().numpy())
        
        # # save weight matrix
        # if collect_w_matrix:
        #     np.save('analysis/weights/w1_noFC_randF.npy'   , w1)
        #     np.save('analysis/weights/w2_noFC_randF.npy'   , w2)
        #     np.save('analysis/weights/param_noFC_randF.npy', params)
        #     np.save('analysis/weights/action_noFC_randF.npy', action_arr)
        #     np.save('analysis/weights/rewards_noFC_randF.npy', rewards_arr)
        #     np.save('analysis/weights/total_rewards.npy', rewards_arr)

        # update reward arrays to ES
        total_rewards_cpu = total_rewards.cpu().numpy()
        fitlist = list(total_rewards_cpu)
        fit_arr = np.array(fitlist)
        # np.save('analysis/weights/total_rewards_Limu_'+cfg.model+'_max.npy', total_rewards_cpu)

        print('mean', fit_arr.mean(), 
            "best", fit_arr.max(), )

    else:
        # Training Loop epoch ###################################
        for epoch in range(EPOCHS):
            # sample params from ES and set model params
            solutions = solver.ask()
            models.set_models_params(solutions)
            # obs = env.reset()
            # imu = obs['obs'][:, 7:10]
            # # print('imu: ', imu.shape)
            # joint_forces = obs['obs'][:, 28:52].reshape(-1, 4, 6)[:, :, 0:3]
            # # print('joint_forces: ', joint_forces.shape)
            # # joint_forces = torch.norm(joint_forces[:, :, 0:3], p=2, dim=2)
            # joint_forces = joint_forces.reshape(cfg.num_envs, 12)
            # obs['obs'] = torch.cat((imu,
            #                     joint_forces),
            #                     dim = 1)            
            # print('obs[obs]: ', obs['obs'].shape)
            # print('observation: ', obs['obs'].shape)

            # Epoch rewards
            total_rewards = torch.zeros(cfg.num_envs)
            total_rewards = total_rewards.cuda()

            # Randomize sensory loss
            rand = torch.randint(0, 2, (cfg.num_envs,))
            v = get_masked_sens_loss_v2(rand).cuda()

            # rollout 
            for sim_step in range(EPISODE_LENGTH_TRAIN):
                # Random actions array for testing
                # actions = torch.zeros(cfg.num_envs, env.action_space.shape[0])
                # actions = models.forward(obs['obs'])

                # print("observation", obs['obs'].shape)
                # print("action", actions.shape)
                # obs, reward, done, info = env.step(
                #     actions
                # )
                # Select observation

                # Duplicate yaw angle
                # obs['obs'] = obs['obs'][:, 7:8].repeat(1,2)

                # IMU + force sensors attached to the feet
                # imu = obs['obs'][:, 7:10]
                # joint_forces = obs['obs'][:, 28:52].reshape(-1, 4, 6)[:, :, 0:3]
                # # joint_forces = torch.norm(joint_forces[:, :, 0:3], p=2, dim=2)
                # joint_forces = joint_forces.reshape(cfg.num_envs, 12)
                # obs['obs'] = torch.cat((imu,
                #                     joint_forces),
                #                     dim = 1)
                # print('imu.shape: ', imu.shape)
                # print('imu: ', imu)
                # print('joint_forces.shape: ', joint_forces.shape)
                # print('joint_forces: ', joint_forces)
                # print('obs[obs]: ', obs['obs'])                
                # ###################################
                # randomize sensory loss of each individual
                # option: 1
                # if sim_step > 100 and sim_step < 600:
                #     obs['obs'] = obs['obs'] * v
                # option: 2 long_take
                # if sim_step > 100 and sim_step < 600:
                #     obs['obs'][:, 0] *= 0.05
                # if sim_step > 600 and sim_step < 1100:
                #     obs['obs'][:, 1] *= 0.05

                ####################################
                # print('observation: ', obs['obs'][:,0])
                reward = np.random.rand()
                total_rewards += reward/EPISODE_LENGTH_TRAIN*100


            # update reward arrays to ES
            total_rewards_cpu = total_rewards.cpu().numpy()
            fitlist = list(total_rewards_cpu)
            solver.tell(fitlist)

            fit_arr = np.array(fitlist)

            print('epoch', epoch, 'mean', fit_arr.mean(), 
                "best", fit_arr.max(), )


            pop_mean_curve[epoch] = fit_arr.mean()
            best_sol_curve[epoch] = fit_arr.max()

            # WanDB Log data -------------------------------
            if cfg.wandb_activate:
                wandb.log({"epoch": epoch,
                            "mean" : np.mean(fitlist),
                            "best" : np.max(fitlist),
                            "worst": np.min(fitlist),
                            "std"  : np.std(fitlist)
                            })
            # -----------------------------------------------

            # Save model params and OpenES params
            if (epoch + 1) % SAVE_EVERY == 0:
                print('saving..')
                pickle.dump((
                    solver,
                    copy.deepcopy(models),
                    pop_mean_curve,
                    best_sol_curve,
                    ), open(dir_path+TASK+'_'+cfg.model+'_' + str(n_params_a_model) +'_' + str(epoch) + '_' + str(pop_mean_curve[epoch])[:8] + '.pickle', 'wb'))



    env.close()

    if cfg.wandb_activate and global_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parse_hydra_configs()
