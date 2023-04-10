import torch
from torch import nn, optim
from torch.nn import functional as F

from algos.AdditionalModels.base_model import Model_base
from utils.models.observation_model import MultimodalObservationModel, ObservationModel_dummy

import wandb

class ObservationModelLearning(Model_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        self.init_models(device)
        self.init_optimizer()
    
    def eval(self):
        self.rssm.eval()
        self.observation_model.eval()
    
    def train(self):
        self.rssm.train()
        self.observation_model.train()
    
    def init_models(self, device):
        if len(self.cfg.additional.observation_names_rec) > 0:
            self.observation_model = MultimodalObservationModel(observation_names_rec=self.cfg.additional.observation_names_rec,
                                                                observation_shapes=self.cfg.env.observation_shapes,
                                                                embedding_size=dict(self.cfg.rssm.embedding_size),
                                                                belief_size=self.cfg.rssm.belief_size,
                                                                state_size=self.cfg.rssm.state_size,
                                                                hidden_size=self.cfg.rssm.hidden_size,
                                                                activation_function=dict(self.cfg.additional.activation_function),
                                                                normalization=self.cfg.additional.normalization,
                                                                device=device)
        else:
            self.observation_model = ObservationModel_dummy()
        self.param_list = self.observation_model.get_model_params()

    def init_optimizer(self):
        self.model_optimizer = optim.Adam(self.param_list, lr=0 if self.cfg.additional.learning_rate_schedule != 0 else self.cfg.additional.model_learning_rate, eps=self.cfg.additional.adam_epsilon)

    def get_state_dict(self):
        state_dict = {'observation_model': self.observation_model.get_state_dict(),
                      'rssm': self.rssm.get_state_dict(),
                     }
        return state_dict

    def load_state_dict(self, state_dict):
        self.rssm.load_state_dict(state_dict["rssm"])
        self.observation_model._load_state_dict(state_dict["observation_model"])

    def calc_loss(self, 
                  observations_target, 
                  actions, 
                  rewards, 
                  nonterminals, 
                  states
                  ):
        beliefs             = states["beliefs"]
        prior_states        = states["prior_states"]
        posterior_states, posterior_means, posterior_std_devs = self.rssm._get_posterior_states(states)

        observations_loss = self._calc_observations_loss(observations_target, beliefs, posterior_states)
        observations_loss_sum = torch.tensor(0., device=self.cfg.main.device)
        for key in observations_loss.keys():
            observations_loss_sum += observations_loss[key]
        
        # Log loss info
        loss_info = dict()
        loss_info["observations_loss_sum"] = observations_loss_sum.item()
        for name in observations_loss.keys():
            loss_info["observation_{}_loss".format(name)] = observations_loss[name].item()
        
        return observations_loss_sum, loss_info


    def optimize_loss(self,
                      observations_target,
                      actions, 
                      rewards,
                      nonterminals, 
                      states,
                      itr_optim):
        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            observations_loss_sum, loss_info = self.calc_loss(observations_target, actions, rewards, nonterminals, states)
        
        # Update model parameters
        self.model_optimizer.zero_grad()

        self.scaler.scale(observations_loss_sum).backward()
        nn.utils.clip_grad_norm_(self.param_list, self.cfg.additional.grad_clip_norm, norm_type=2)
        self.scaler.step(self.model_optimizer)
        self.scaler.update()
        
        # Log loss info
        if self.cfg.main.wandb:
            for name in loss_info.keys():
                wandb.log(data={"{}/train".format(name):loss_info[name]}, step=itr_optim)
            frame = itr_optim * self.cfg.train.batch_size * self.cfg.train.chunk_size
            wandb.log(data={"frame":frame}, step=itr_optim)

    def optimize(self,
                 D, 
                 ):
        self.itr_optim += 1
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals, _ = D.sample(
                self.cfg.train.batch_size, self.cfg.train.chunk_size)  # Transitions start at time t = 0

        observations_target = self._clip_obs(observations, idx_start=1)

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            states = self.rssm.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
        
        if not self.cfg.additional.rssm.fix:
            self.rssm.optimize_loss(observations_target,
                                    actions, 
                                    rewards, 
                                    nonterminals,
                                    states,
                                    self.itr_optim)
        
        self.optimize_loss(observations_target,
                           actions, 
                           rewards,
                           nonterminals,
                           states,
                           self.itr_optim)
    
    def validation(self,
                    D,
                    ):
        self.eval()
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals, _ = D.sample(
                self.cfg.train.batch_size, self.cfg.train.chunk_size)  # Transitions start at time t = 0

        observations_target = self._clip_obs(observations, idx_start=1)

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            states = self.rssm.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
            observations_loss_sum, loss_info = self.calc_loss(observations_target, actions, rewards, nonterminals, states)

            if not self.cfg.additional.rssm.fix:
                rssm_loss, rssm_loss_info = self.rssm._get_model_loss(observations_target, actions, rewards, nonterminals, states)

        # Log loss info
        if self.cfg.main.wandb:
            for name in loss_info.keys():        
                wandb.log(data={"{}/validation".format(name):loss_info[name]}, step=self.itr_optim)
            if not self.cfg.additional.rssm.fix:
                for name in rssm_loss_info.keys():
                    wandb.log(data={"{}/validation".format(name):rssm_loss_info[name]}, step=self.itr_optim)

        self.train()
    
    def _calc_observations_loss(self,
                               observations_target,
                               beliefs,
                               posterior_states,
                               ):
        observations_loss = dict()
        if len(self.cfg.additional.observation_names_rec) > 0:
            if self.cfg.additional.worldmodel_observation_loss == "log_prob":
                log_probs = self.observation_model.get_log_prob(beliefs, posterior_states, observations_target)
                for name in log_probs.keys():
                    observations_loss[name] = -log_probs[name].mean(dim=(0,1)).sum()
            elif self.cfg.additional.worldmodel_observation_loss == "mse":
                mse = self.observation_model.get_mse(h_t=beliefs, s_t=posterior_states, o_t=observations_target)
                for name in mse.keys():
                    observations_loss[name] = mse[name].mean(dim=(0,1)).sum()
            elif self.cfg.additional.worldmodel_observation_loss == "mae":
                mae = self.observation_model.get_mae(h_t=beliefs, s_t=posterior_states, o_t=observations_target)
                for name in mae.keys():
                    observations_loss[name] = mae[name].mean(dim=(0,1)).sum()
            else:
                raise NotImplementedError
            
        return observations_loss
