import torch

from utils.models.encoder import bottle_tupele_multimodal
from utils.models.encoder import MultimodalEncoderNN
from utils.models.transition_model import MultimodalTransitionModel_emb
from utils.models.reward_model import RewardModel
from utils.models.observation_model import MultimodalObservationModel, ObservationModel_dummy
from utils.models.contrastive import LinearPredictor, Feature2ObsEmbed
from algos.MRSSM.base.base import MRSSM_st_base, MRSSM_obs_emb_base

# class MRSSM_st_NN(MRSSM_st_base):
#     def __init__(self, cfg, device):
#         super().__init__(cfg, device)
#         print("Multimodal RSSM (fusion states by NN)")

class MRSSM_obs_emb_NN(MRSSM_obs_emb_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (fusion obs_emb by NN)")
    
    def _init_models(self, device):
        self.encoder = MultimodalEncoderNN(observation_names_enc=self.cfg.rssm.observation_names_enc,
                                            observation_shapes=self.cfg.env.observation_shapes,
                                            embedding_size=dict(self.cfg.rssm.embedding_size),
                                            activation_function=dict(self.cfg.rssm.activation_function),
                                            normalization=self.cfg.rssm.normalization,
                                            device=device
                                            )
        
        self.transition_model = MultimodalTransitionModel_emb(belief_size=self.cfg.rssm.belief_size, 
                                                        state_size=self.cfg.rssm.state_size,
                                                        action_size=self.cfg.env.action_size, 
                                                        hidden_size=self.cfg.rssm.hidden_size, 
                                                        observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                        embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                        device=device,
                                                        ).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.rssm.belief_size, s_size=self.cfg.rssm.state_size, hidden_size=self.cfg.rssm.hidden_size,
                                activation=self.cfg.rssm.activation_function.dense).to(device=device)
        
        if len(self.cfg.rssm.observation_names_rec) > 0:
            self.observation_model = MultimodalObservationModel(observation_names_rec=self.cfg.rssm.observation_names_rec,
                                                                observation_shapes=self.cfg.env.observation_shapes,
                                                                embedding_size=dict(self.cfg.rssm.embedding_size),
                                                                belief_size=self.cfg.rssm.belief_size,
                                                                state_size=self.cfg.rssm.state_size,
                                                                hidden_size=self.cfg.rssm.hidden_size,
                                                                activation_function=dict(self.cfg.rssm.activation_function),
                                                                normalization=self.cfg.rssm.normalization,
                                                                device=device)
        else:
            self.observation_model = ObservationModel_dummy()
        
        if self.cfg.rssm.dreaming.use:
            self.linear_predictor = LinearPredictor(state_size=self.cfg.rssm.state_size,
                                                    action_size=self.cfg.env.action_size,
                                                    embedding_size=self.cfg.rssm.embedding_size["fusion"],
                                                    reward_size=1,
                                                    ).to(device=device)

        if self.cfg.rssm.MuMMI.use:
            self.feat2obs_emb = Feature2ObsEmbed(belief_size=self.cfg.rssm.belief_size,
                                                state_size=self.cfg.rssm.state_size,
                                                embedding_size=self.cfg.rssm.embedding_size["fusion"]
                                                ).to(device=device)

    def get_obs_emb_posterior(self, observations, use_mean=False):
        obs_emb = bottle_tupele_multimodal(self.encoder, observations)
        return obs_emb["mixed"]
    
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
        
        obs_emb_posterior = self.get_obs_emb_posterior(observations)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, expert_means, expert_std_devs = self.transition_model(
            init_state, actions, init_belief, obs_emb_posterior, nonterminals, det=det)

        
        states = dict(beliefs=beliefs,
                     prior_states=prior_states,
                     prior_means=prior_means,
                     prior_std_devs=prior_std_devs,
                     posterior_states=posterior_states,
                     posterior_means=posterior_means,
                     posterior_std_devs=posterior_std_devs,
                     )
        return states
