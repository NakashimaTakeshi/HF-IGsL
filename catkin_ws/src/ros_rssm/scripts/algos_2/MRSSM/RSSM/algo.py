import torch
from torch.nn import functional as F

import wandb

from utils.models.encoder import build_Encoder, bottle_tupele
from utils.models.observation_model import build_ObservationModel, ObservationModel_dummy
from utils.models.pose_predict_model import PosePredictModel
from utils.models.reward_model import RewardModel
from utils.models.transition_model import TransitionModel
from utils.models.contrastive import LinearPredictor

from algos.MRSSM.base.base import RSSM_base

class RSSM(RSSM_base):
    def __init__(self, cfg, device):
        if not (len(cfg.rssm.observation_names_enc) == 1):
            raise NotImplementedError("Num of observations is {}. Please use Multimodal-RSSM.".format(len(cfg.rssm.observation_names_enc)))
        self.observation_name = cfg.rssm.observation_names_enc[0]
        
        super().__init__(cfg, device)
        print("RSSM")

    def _init_models(self, device):
        self.transition_model = TransitionModel(self.cfg.rssm.belief_size, self.cfg.rssm.state_size, self.cfg.env.action_size,
                                        self.cfg.rssm.hidden_size, dict(self.cfg.rssm.embedding_size), self.cfg.rssm.activation_function.dense).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.rssm.belief_size, s_size=self.cfg.rssm.state_size, hidden_size=self.cfg.rssm.hidden_size,
                                activation=self.cfg.rssm.activation_function.dense).to(device=device)
        
        if len(self.cfg.rssm.observation_names_rec) > 0:
            self.observation_model = build_ObservationModel(name=self.cfg.rssm.observation_names_rec[0],
                                                            observation_shapes=self.cfg.env.observation_shapes,
                                                            embedding_size=dict(self.cfg.rssm.embedding_size),
                                                            belief_size=self.cfg.rssm.belief_size,
                                                            state_size=self.cfg.rssm.state_size,
                                                            hidden_size=self.cfg.rssm.hidden_size,
                                                            activation_function=dict(self.cfg.rssm.activation_function),
                                                            mode = self.cfg.rssm.HF_PGM["use"],
                                                            normalization=self.cfg.rssm.normalization,
                                                            ).to(device=device)
        else:
            self.observation_model = ObservationModel_dummy()
        if self.cfg.rssm.dreaming.use:
            self.linear_predictor = LinearPredictor(state_size=self.cfg.rssm.state_size,
                                                    action_size=self.cfg.env.action_size,
                                                    embedding_size=self.cfg.rssm.embedding_size["fusion"],
                                                    reward_size=1,
                                                    ).to(device=device)
        self.encoder = build_Encoder(name=self.cfg.rssm.observation_names_enc[0],
                                    observation_shapes=self.cfg.env.observation_shapes,
                                    embedding_size=dict(self.cfg.rssm.embedding_size), 
                                    activation_function=dict(self.cfg.rssm.activation_function),
                                    normalization=self.cfg.rssm.normalization,
                                    ).to(device=device)
        #HF-PGM Predict x_t models from ht (chenged)                            
        if self.cfg.rssm.HF_PGM["pose_predict"]:
            self.pose_poredict_model = PosePredictModel(belief_size = self.cfg.rssm.belief_size, action_size = self.cfg.env.action_size,
                                        activation_function=dict(self.cfg.rssm.activation_function)["dense"]).to(device=device)

        
        if self.cfg.main.wandb:
            wandb.watch(self.observation_model)
            wandb.watch(self.transition_model)
            wandb.watch(self.reward_model)
            wandb.watch(self.encoder)
            if self.cfg.rssm.dreaming.use:
                wandb.watch(self.linear_predictor)
            if self.cfg.rssm.HF_PGM["pose_predict"]:
                wandb.watch(self.pose_poredict_model)


    def _init_model_modules(self):
        if self.cfg.rssm.HF_PGM["pose_predict"]:
            self.model_modules = self.encoder.modules \
                            + self.transition_model.modules \
                            + self.observation_model.modules \
                            + self.reward_model.modules \
                            + self.pose_poredict_model.modules
        else:
            self.model_modules = self.encoder.modules \
                            + self.transition_model.modules \
                            + self.observation_model.modules \
                            + self.reward_model.modules
                           
    def _init_param_list(self):
        self.param_list = list(self.parameters())

    def get_state_dict(self):
        return self.state_dict()

    def estimate_state(self,
                        observations,
                        actions, 
                        rewards, 
                        nonterminals,
                        batch_size=None,
                        det=False):
        if batch_size == None:
            batch_size = actions.shape[1]

        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(batch_size, self.cfg.rssm.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.rssm.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        
        obs_emb = bottle_tupele(self.encoder, observations)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, expert_means, expert_std_devs = self.transition_model(
            init_state, actions, init_belief, obs_emb, nonterminals, det=det)

        states = dict(beliefs=beliefs,
                     prior_states=prior_states,
                     prior_means=prior_means,
                     prior_std_devs=prior_std_devs,
                     posterior_states=posterior_states,
                     posterior_means=posterior_means,
                     posterior_std_devs=posterior_std_devs,
                     expert_means=expert_means, 
                     expert_std_devs=expert_std_devs,
                     )
        return states
    
    def _calc_observations_loss(self,
                               observations_target,
                               beliefs,
                               posterior_states,
                               ):
        observations_loss = dict()
        observations_loss["observations_loss_sum"] = torch.tensor(0., device=self.cfg.main.device)
        if len(self.cfg.rssm.observation_names_rec) > 0:
            if self.cfg.rssm.worldmodel_LogProbLoss:
                log_probs = self.observation_model.get_log_prob(beliefs, posterior_states, observations_target[self.observation_name])
                observation_loss = (-log_probs).mean(dim=(0,1)).sum()
            else:
                mse = self.observation_model.get_mse(beliefs, posterior_states, observations_target[self.observation_name])
                observation_loss = (mse).mean(dim=(0,1)).sum()
            observations_loss[self.observation_name] = observation_loss
            observations_loss["observations_loss_sum"] += observation_loss
        return observations_loss

    def _calc_predict_pose_loss(self, action_target, beliefs):
        predict_pose_loss = dict()
        predict_pose_loss["predict_pose_loss"] = torch.tensor(0., device=self.cfg.main.device)
        log_probs = self.pose_poredict_model.get_log_prob(beliefs, action_target)
        predict_pose_loss["predict_pose_loss"] = (-log_probs).mean(dim=(0,1)).sum()
        return predict_pose_loss

    def estimate_state_online(self, observations, actions, past_state, past_belief):
        """
        関数horiemon
        estimate_stateのオンライン版
        
        """
        batch_size = 1
        det = False
        nonterminals = torch.ones(1, 1, 1, device=self.cfg.main.device)

        obs_emb = bottle_tupele(self.encoder, observations)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, expert_means, expert_std_devs = self.transition_model(
            past_state, actions, past_belief, obs_emb, nonterminals, det=det)

        states = dict(beliefs=beliefs,
                     prior_states=prior_states,
                     prior_means=prior_means,
                     prior_std_devs=prior_std_devs,
                     posterior_states=posterior_states,
                     posterior_means=posterior_means,
                     posterior_std_devs=posterior_std_devs,
                     expert_means=expert_means, 
                     expert_std_devs=expert_std_devs,
                     ) 
        return states
