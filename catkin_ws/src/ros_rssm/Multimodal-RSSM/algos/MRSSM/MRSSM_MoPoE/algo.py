import torch
from torch.distributions import Normal
from torch.nn import functional as F

from utils.dist import calc_kl_divergence, calc_subset_states, get_mopoe_params, _calc_subset_states
from utils.models.encoder import bottle_tupele_multimodal
from algos.MRSSM.base.base import MRSSM_st_base, MRSSM_obs_emb_base

class MRSSM_st_MoPoE(MRSSM_st_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (fusion states by MoPoE)")
    
    def _latent_overshooting(self,
                             actions,
                             rewards,
                             nonterminals,
                             states,
                             ):
        beliefs             = states["beliefs"]
        prior_states        = states["prior_states"]
        
        subset_means, subset_std_devs = calc_subset_states(states["expert_means"], states["expert_std_devs"])
        n_subset = len(subset_means)
        kl_loss_sum = torch.tensor(0., device=self.cfg.main.device)
        reward_loss = torch.tensor(0., device=self.cfg.main.device)

        for i in range(n_subset):
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            for t in range(1, self.cfg.train.chunk_size - 1):
                d = min(t + self.cfg.rssm.overshooting_distance,
                        self.cfg.train.chunk_size - 1)  # Overshooting distance
                # Use t_ and d_ to deal with different time indexing for latent states
                t_, d_ = t - 1, d - 1
                # Calculate sequence padding so overshooting terms can be calculated in one batch
                seq_pad = (0, 0, 0, 0, 0, t - d + self.cfg.rssm.overshooting_distance)
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(subset_means[i][t_ + 1:d_ + 1].detach(), seq_pad), 
                F.pad(subset_std_devs[i][t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, self.cfg.train.batch_size, self.cfg.rssm.state_size, device=self.cfg.main.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            _beliefs, _prior_states, _prior_means, _prior_std_devs = self.transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(
                overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            div = calc_kl_divergence(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1),
                                     _prior_means, _prior_std_devs,
                                     kl_balancing_alpha=self.cfg.rssm.kl_balancing_alpha) * seq_mask
            kl_loss_sum += self.cfg.rssm.overshooting_kl_beta * torch.max(div, self.free_nats).mean(dim=(0, 1))  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
        kl_loss_sum = kl_loss_sum / n_subset
        # Calculate overshooting reward prediction loss with sequence mask
        if self.cfg.rssm.overshooting_reward_scale != 0:
            reward_loss += (1 / self.cfg.rssm.overshooting_distance) * self.cfg.rssm.overshooting_reward_scale * F.mse_loss(self.reward_model(_beliefs, _prior_states)['loc'] * seq_mask[:, :, 0], torch.cat(
                overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (self.cfg.train.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        return kl_loss_sum, reward_loss

    def _calc_mopoe_kl(self,
                        expert_means,
                        expert_std_devs,
                        prior_means,
                        prior_std_devs,
                        ):
        
        subset_means, subset_std_devs = calc_subset_states(expert_means, expert_std_devs)
        kl_losses = []
        for i in range(len(subset_means)):
            div = calc_kl_divergence(subset_means[i], subset_std_devs[i], 
                                     prior_means, prior_std_devs,
                                     kl_balancing_alpha=self.cfg.rssm.kl_balancing_alpha)
            kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))
            kl_losses.append(kl_loss)
        
        return torch.stack(kl_losses).mean(dim=0)

    def _calc_kl(self,
                states,
                ):
        prior_means         = states["prior_means"]
        prior_std_devs      = states["prior_std_devs"]
        expert_means = states["expert_means"]
        expert_std_devs = states["expert_std_devs"]

        kl_loss = self._calc_mopoe_kl(expert_means, expert_std_devs, prior_means, prior_std_devs)

        return kl_loss


class MRSSM_obs_emb_MoPoE(MRSSM_obs_emb_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (fusion obs_emb by MoPoE)")

    def get_obs_emb_posterior(self, observations, use_mean=False, subset_index = None):
        _obs_emb = bottle_tupele_multimodal(self.encoder, observations)
        expert_means = dict()
        expert_std_devs = dict()
        for key in _obs_emb.keys():
            expert_means[key] = _obs_emb[key]["loc"]
            expert_std_devs[key] = _obs_emb[key]["scale"]
        means, std_devs = get_mopoe_params(expert_means, expert_std_devs, 
                                           fusion_method=self.cfg.rssm.multimodal_params.fusion_method, 
                                           num_components=self.cfg.rssm.multimodal_params.num_components,
                                           subset_index = subset_index
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
                        det=False,
                        subset_index = None):
        T,B = actions.shape[:2]
        if batch_size == None:
            batch_size = actions.shape[1]

        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(batch_size, self.cfg.rssm.belief_size, device=self.cfg.main.device), torch.zeros(batch_size, self.cfg.rssm.state_size, device=self.cfg.main.device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        
        obs_emb_posterior = self.get_obs_emb_posterior(observations, subset_index = subset_index)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, expert_means, expert_std_devs = self.transition_model(
            init_state, actions, init_belief, obs_emb_posterior, nonterminals, det=det)

        # subset states
        obs_emb = bottle_tupele_multimodal(self.encoder, observations)
        expert_means = dict()
        expert_std_devs = dict()
        for key in obs_emb.keys():
            expert_means[key] = obs_emb[key]["loc"]
            expert_std_devs[key] = obs_emb[key]["scale"]
        obs_emb_subset_means, obs_emb_subset_std_devs = _calc_subset_states(expert_means, expert_std_devs)

        _init_belief = torch.vstack([init_belief.reshape(1,batch_size,-1), beliefs[:-1]]).reshape(-1,self.cfg.rssm.belief_size)
        _init_state = torch.vstack([init_state.reshape(1,batch_size,-1), posterior_states[:-1]]).reshape(-1,self.cfg.rssm.state_size)
        _actions = actions.reshape(1,-1, actions.size(-1))
        _nonterminals = nonterminals.reshape(1,-1, nonterminals.size(-1))
        posterior_means_subset = dict()
        posterior_std_devs_subset = dict()
        
        obs_emb_size = self.cfg.rssm.embedding_size.fusion
        for subset_key in obs_emb_subset_means.keys():
            _beliefs, _prior_states, _prior_means, _prior_std_devs, _posterior_states, _posterior_means, _posterior_std_devs, _expert_means, _expert_std_devs = self.transition_model(
                _init_state, _actions, _init_belief, obs_emb_subset_means[subset_key].reshape(1,-1, obs_emb_size), _nonterminals, det=det)
            posterior_means_subset[subset_key] = _posterior_means.reshape(T,B,self.cfg.rssm.state_size)
            posterior_std_devs_subset[subset_key] = _posterior_std_devs.reshape(T,B,self.cfg.rssm.state_size)
        
        states = dict(beliefs=beliefs,
                     prior_states=prior_states,
                     prior_means=prior_means,
                     prior_std_devs=prior_std_devs,
                     posterior_states=posterior_states,
                     posterior_means=posterior_means,
                     posterior_std_devs=posterior_std_devs,
                     obs_emb_subset_means=obs_emb_subset_means,
                     obs_emb_subset_std_devs=obs_emb_subset_std_devs,
                     posterior_means_subset=posterior_means_subset,
                     posterior_std_devs_subset=posterior_std_devs_subset,
                     )
        return states
    
    def estimate_state_online(self, observations, actions, past_state, past_belief, subset_index = None):
        """
        estimate_state(MoPoE)のオンライン版
        """
        det = False
        nonterminals = torch.ones(1, 1, 1, device=self.cfg.main.device)


        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        obs_emb_posterior = self.get_obs_emb_posterior(observations, subset_index = subset_index)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, expert_means, expert_std_devs = self.transition_model(
            past_state, actions, past_belief, obs_emb_posterior, nonterminals, det=det)

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

    def _calc_kl(self,
                states
                ):
        prior_means                 = states["prior_means"]
        prior_std_devs              = states["prior_std_devs"]
        posterior_means_subset      = states["posterior_means_subset"]
        posterior_std_devs_subset   = states["posterior_std_devs_subset"]
        kl_losses = []
        for key in posterior_means_subset.keys():
            div = calc_kl_divergence(posterior_means_subset[key], posterior_std_devs_subset[key], 
                                     prior_means, prior_std_devs,
                                     kl_balancing_alpha=self.cfg.rssm.kl_balancing_alpha)
            kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))
            kl_losses.append(kl_loss)
        
        return torch.stack(kl_losses).mean(dim=0)

    def get_coefficient(self, subset_key):
        coefficients = 1.
        for key in self.cfg.rssm.coefficients.keys():
            if key in subset_key:
                coefficients *= self.cfg.rssm.coefficients[key]
        return coefficients

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

    def _calc_MuMMI_loss_mopoe(self,
                               observations,
                               beliefs,
                               posterior_states,
                               use_mean=False,
                               ):
        feat_embed = self.feat2obs_emb(h_t=beliefs, s_t=posterior_states)

        obs_embs = bottle_tupele_multimodal(self.encoder, observations)
        expert_means = dict()
        expert_std_devs = dict()
        for obs_emb_key in obs_embs.keys():
            expert_means[obs_emb_key] = obs_embs[obs_emb_key]["loc"]
            expert_std_devs[obs_emb_key] = obs_embs[obs_emb_key]["scale"]
        subset_means, subset_std_devs = _calc_subset_states(expert_means, expert_std_devs)

        contrasts = dict()
        contrasts["mean"] = torch.tensor(0., device=self.cfg.main.device)
        for subset_key in subset_means.keys():
            if use_mean:
                obs_emb = subset_means[subset_key]
            else:
                obs_emb = Normal(subset_means[subset_key], subset_std_devs[subset_key]).rsample()
            contrasts[subset_key] = self._calc_contrastive(feat_embed, obs_emb).mean()
            coefficient = self.get_coefficient(subset_key)
            contrasts["mean"] += coefficient * contrasts[subset_key]
        contrasts["mean"] /= len(subset_means.keys())
        return contrasts
    
    def _calc_MuMMI_loss(self,
                        observations,
                        beliefs,
                        posterior_states,
                        ):
        # return self._calc_MuMMI_loss_mopoe(observations, beliefs, posterior_states)
        return self._calc_MuMMI_loss_poe(observations, beliefs, posterior_states)