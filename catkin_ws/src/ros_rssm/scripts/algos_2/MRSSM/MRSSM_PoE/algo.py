import torch
from torch.distributions import Normal

from utils.models.encoder import bottle_tupele_multimodal
from utils.dist import get_poe_params
from algos.MRSSM.base.base import MRSSM_st_base, MRSSM_obs_emb_base

class MRSSM_st_PoE(MRSSM_st_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (fusion states by PoE)")

class MRSSM_obs_emb_PoE(MRSSM_obs_emb_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (fusion obs_emb by PoE)")

    def get_obs_emb_posterior(self, observations, use_mean=False):
        _obs_emb = bottle_tupele_multimodal(self.encoder, observations)
        expert_means = dict()
        expert_std_devs = dict()
        for key in _obs_emb.keys():
            expert_means[key] = _obs_emb[key]["loc"]
            expert_std_devs[key] = _obs_emb[key]["scale"]
        means, std_devs = get_poe_params(expert_means, expert_std_devs, 
                                                  fusion_method=self.cfg.rssm.multimodal_params.fusion_method, 
                                                  num_components=self.cfg.rssm.multimodal_params.num_components,
                                                  )
        if use_mean:
            obs_emb = means
        else:
            obs_emb = Normal(means, std_devs).rsample()
        return obs_emb
    
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
    
    def _calc_MuMMI_loss_poe(self,
                                observations,
                                beliefs,
                                posterior_states,
                                use_mean=False,
                                ):
        feat_embed = self.feat2obs_emb(h_t=beliefs, s_t=posterior_states)

        obs_embs = bottle_tupele_multimodal(self.encoder, observations)
        expert_means = dict()
        expert_std_devs = dict()
        for key in obs_embs.keys():
            expert_means[key] = obs_embs[key]["loc"]
            expert_std_devs[key] = obs_embs[key]["scale"]
        
        contrasts = dict()
        contrasts["mean"] = torch.tensor(0., device=self.cfg.main.device)
        for key in expert_means.keys():
            if use_mean:
                obs_emb = expert_means[key]
            else:
                obs_emb = Normal(expert_means[key], expert_std_devs[key]).rsample()
            contrasts[key] = self._calc_contrastive(feat_embed, obs_emb).mean()
            contrasts["mean"] += self.cfg.rssm.coefficients[key] * contrasts[key]
        contrasts["mean"] /= len(expert_means.keys())
        return contrasts
    
    def _calc_MuMMI_loss(self,
                        observations,
                        beliefs,
                        posterior_states,
                        ):
        return self._calc_MuMMI_loss_poe(observations, beliefs, posterior_states)