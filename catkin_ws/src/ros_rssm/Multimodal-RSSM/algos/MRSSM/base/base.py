import os

import numpy as np

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

import wandb

from utils.models.encoder import bottle_tupele_multimodal
from utils.models.encoder import MultimodalEncoder, MultimodalObservationEncoder
from utils.models.transition_model import MultimodalTransitionModel, MultimodalTransitionModel_emb
from utils.models.reward_model import RewardModel
from utils.models.observation_model import MultimodalObservationModel, ObservationModel_dummy
from utils.models.contrastive import LinearPredictor, Feature2ObsEmbed
from utils.dist import calc_kl_divergence

class RSSM_base(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        
        self.cfg = cfg
        self.device = device
        self._init_models(device)
        self._init_model_modules()
        self._init_param_list()
        self._init_optimizer()
        
        self.global_prior = Normal(torch.zeros(cfg.train.batch_size, cfg.rssm.state_size, device=device), 
                                   torch.ones(cfg.train.batch_size, cfg.rssm.state_size, device=device))  # Global prior N(0, I)
        # Allowed deviation in KL divergence
        self.free_nats = torch.full((1, ), cfg.rssm.free_nats, device=device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
        self.itr_optim = 0

    def _init_models(self, device):
        raise NotImplementedError
    
    def _init_model_modules(self):
        raise NotImplementedError

    def _init_param_list(self):
        raise NotImplementedError
                    
    def _init_optimizer(self):
        self.model_optimizer = optim.Adam(self.param_list, 
                                          lr=0 if self.cfg.rssm.learning_rate_schedule != 0 else self.cfg.rssm.model_learning_rate, 
                                          eps=self.cfg.rssm.adam_epsilon)

    def get_state_dict(self):
        raise NotImplementedError

    def _load_model_dicts(self, model_path):
        print("load model_dicts from {}".format(model_path))
        return torch.load(model_path, map_location=torch.device(self.device))
    
    def load_model(self, model_path, reset_optimizer=True):
        model_dicts = self._load_model_dicts(model_path)
        self.load_state_dict(model_dicts)
        if reset_optimizer:
            self._init_optimizer() # init optimizer for reset

    def load_encoder_state_dict(self, obs_name, state_dict):
        self.encoder.encoders[obs_name].load_state_dict(state_dict["encoder"]["encoder"][obs_name])

    def load_decoder_state_dict(self, obs_name, state_dict):
        self.observation_model.observation_models[obs_name].decoder.load_state_dict(state_dict["observation_model"][obs_name]["decoder"])

    # def load_pretrain_model(self, model_path, obs_names, load_encoder=True, load_decoder=True):
    #     model_dicts = self._load_model_dicts(model_path)
    #     for obs_name in obs_names:
    #         if load_encoder:
    #             self.load_encoder_state_dict(obs_name, model_dicts)
    #         if load_decoder:
    #             self.load_decoder_state_dict(obs_name, model_dicts)

    def load_pretrain_model(self, model_path, obs_names, load_encoder=True, load_decoder=True):
        model_dicts = self._load_model_dicts(model_path)
        for obs_name in obs_names:
            if load_encoder:
                self.load_encoder_state_dict(obs_name, model_dicts)
            if load_decoder:
                self.load_decoder_state_dict(obs_name, model_dicts)

    def save_model(self, results_dir, itr):
        state_dict = self.get_state_dict()
        torch.save(state_dict, os.path.join(results_dir, 'models_%d.pth' % itr))

    def _clip_obs(self, observations, idx_start=0, idx_end=None):
        output = dict()
        for k in observations.keys():
            output[k] = observations[k][idx_start:idx_end]
        return output

    def get_obs_emb(self, observations):
        raise NotImplementedError

    def get_obs_emb_posterior(self, observations):
        raise NotImplementedError

    def estimate_state(self,
                        observations,
                        actions, 
                        rewards, 
                        nonterminals,
                        batch_size=None,
                        det=False):
        raise NotImplementedError
    
    def _calc_kl(self,
                states,
                ):
        prior_means         = states["prior_means"]
        prior_std_devs      = states["prior_std_devs"]
        posterior_means     = states["posterior_means"]
        posterior_std_devs  = states["posterior_std_devs"]

        div = calc_kl_divergence(posterior_means, posterior_std_devs,
                                 prior_means, prior_std_devs,
                                 kl_balancing_alpha=self.cfg.rssm.kl_balancing_alpha,
                                 )
        # Note that normalization by overshooting distance and weighting by overshooting distance cancel out
        kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))
        return kl_loss

    def _calc_reward_loss(self,
                          rewards,
                          beliefs,
                          posterior_states,
                          ):
        if self.cfg.rssm.worldmodel_observation_loss == "log_prob":
            reward_loss = -self.reward_model.get_log_prob(beliefs, posterior_states, rewards[:-1])
            reward_loss = reward_loss.mean(dim=(0, 1))
        elif self.cfg.rssm.worldmodel_observation_loss == "mse":
            reward_mean = self.reward_model(
                h_t=beliefs, s_t=posterior_states)['loc']
            reward_loss = F.mse_loss(reward_mean, rewards[:-1], reduction='none').mean(dim=(0, 1))
        elif self.cfg.rssm.worldmodel_observation_loss == "mae":
            reward_mean = self.reward_model(
                h_t=beliefs, s_t=posterior_states)['loc']
            reward_loss = F.l1_loss(reward_mean, rewards[:-1], reduction='none').mean(dim=(0, 1))

        return reward_loss

    def _latent_overshooting(self,
                             actions,
                             rewards,
                             nonterminals,
                             states,
                             ):
        beliefs             = states["beliefs"]
        prior_states        = states["prior_states"]
        posterior_states, posterior_means, posterior_std_devs = self._get_posterior_states(states)
        

        kl_loss_sum = torch.tensor(0., device=self.cfg.main.device)
        reward_loss = torch.tensor(0., device=self.cfg.main.device)

        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, self.cfg.train.chunk_size - 1):
            d = min(t + self.cfg.rssm.overshooting_distance,
                    self.cfg.train.chunk_size - 1)  # Overshooting distance
            # Use t_ and d_ to deal with different time indexing for latent states
            t_, d_ = t - 1, d - 1
            # Calculate sequence padding so overshooting terms can be calculated in one batch
            seq_pad = (0, 0, 0, 0, 0, t - d + self.cfg.rssm.overshooting_distance)
            # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
            overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(
                posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, self.cfg.train.batch_size, self.cfg.rssm.state_size, device=self.cfg.main.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
        overshooting_vars = tuple(zip(*overshooting_vars))
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = self.transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(
            overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
        seq_mask = torch.cat(overshooting_vars[7], dim=1)
        # Calculate overshooting KL loss with sequence mask
        div = calc_kl_divergence(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1), 
                                 prior_means, prior_std_devs,
                                 kl_balancing_alpha=self.cfg.rssm.kl_balancing_alpha,
                                 ) * seq_mask
        kl_loss_sum += self.cfg.rssm.overshooting_kl_beta * torch.max(div, self.free_nats).mean(dim=(0, 1))  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
        # Calculate overshooting reward prediction loss with sequence mask
        if self.cfg.rssm.overshooting_reward_scale != 0:
            reward_loss += (1 / self.cfg.rssm.overshooting_distance) * self.cfg.rssm.overshooting_reward_scale * F.mse_loss(self.reward_model(beliefs, prior_states)['loc'] * seq_mask[:, :, 0], torch.cat(
                overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (self.cfg.train.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        return kl_loss_sum, reward_loss

    def _calc_dreaming_loss(self,
                            observations_contrastive,
                            actions,
                            rewards,
                            nonterminals,
                            posterior_states,
                            ):
        # parameters
        cpc_overshooting_distance = self.cfg.rssm.dreaming.cpc_overshooting_distance
        
        obs_emb = self.get_obs_emb_posterior(observations_contrastive)

        obs = torch.cat([obs_emb, rewards[1:].unsqueeze(-1)], dim=-1)
        _pred = self.linear_predictor.w_z_x(obs)
        pred = []
        for t in range(cpc_overshooting_distance):
            pred_tmp = torch.zeros_like(_pred)
            pred_tmp[:-t-1] = _pred[t+1:]
            pred.append(pred_tmp)
        pred = torch.stack(pred, axis=-2)

        features = self.linear_predictor(posterior_states, actions, nonterminals[1:], T=cpc_overshooting_distance)

        
        T, B, O, Z = features.shape  # TxBxOxZ
        if self.cfg.rssm.dreaming.cpc_contrast == 'window':
            features = features.reshape(T, B * O, Z)  # TxBxOxZ => Tx(BxO)xZ => TxB'xZ
            pred = pred.reshape(T, B * O, Z)
            features = features.unsqueeze(2)  # TxB'x1xZ
            pred = pred.unsqueeze(1)  # Tx1xB'xZ

            labels = torch.diag(torch.ones(B * O, device=self.cfg.main.device)).unsqueeze(0)  # 1xB'xB'
            labels = torch.tile(labels, (T, 1, 1))  # TxB'xB'
                
        elif self.cfg.rssm.dreaming.cpc_contrast == 'time':
            features = features.unsqueeze(3)  # TxBxOx1xZ
            pred = pred.unsqueeze(2)  # TxBx1xOxZ

            labels = torch.diag(torch.ones(O, device=self.cfg.main.device)).unsqueeze(0).unsqueeze(0)  # 1x1xOxO
            labels = torch.tile(labels, (T, B, 1, 1))  # TxBxOxO
            
        elif self.cfg.rssm.dreaming.cpc_contrast == 'batch':
            features = features.unsqueeze(2)  # TxBx1xOxZ
            pred = pred.unsqueeze(1)  # Tx1xBxOxZ

            labels = torch.diag(torch.ones(B, device=self.cfg.main.device)).unsqueeze(0).unsqueeze(-1)  # 1xBxBx1
            labels = torch.tile(labels, (T, 1, 1, O))  # TxBxBxO
            
        else:
            raise NotImplementedError(self.cfg.rssm.dreaming.cpc_contrast)
        
        logits = torch.sum(pred * features, axis=-1)  # TxBxBxO
        logits /= self.cfg.rssm.dreaming.mycpc_temp_logits
        
        logits = logits.reshape(T*B*O, -1)
        labels = labels.reshape(T*B*O, -1)
        cross_entropy = F.cross_entropy(logits, labels, reduction='none')  # TxBxO
        cross_entropy = cross_entropy.reshape(T, B, O)
        
        mask = torch.zeros(T, B, O, device=self.cfg.main.device)
        for t in range(O):
            _mask = torch.zeros(T, B, 1, device=self.cfg.main.device)
            if t == 0:
                _mask = nonterminals[t+1:]
            else:
                _mask[:-t] = nonterminals[t+1:]
            mask[:,:,t:t+1] = _mask

        cross_entropy *= mask
        return cross_entropy.mean()
    
    def _contrastive_predictive_coding_loss(self):
        raise NotImplementedError

    def _get_posterior_states(self,
                              states,
                              ):
        posterior_states    = states["posterior_states"]
        posterior_means     = states["posterior_means"]
        posterior_std_devs  = states["posterior_std_devs"]
        return posterior_states, posterior_means, posterior_std_devs

    def _calc_loss(self, 
                  observations_target,
                  actions, 
                  rewards, 
                  nonterminals,
                  states,
                  observations_contrastive=None,
                  ):

        beliefs             = states["beliefs"]
        prior_states        = states["prior_states"]
        posterior_states, posterior_means, posterior_std_devs = self._get_posterior_states(states)

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        observations_loss = self._calc_observations_loss(observations_target, beliefs, posterior_states)
        reward_loss = self._calc_reward_loss(rewards, beliefs, posterior_states)
        
        
        # transition loss
        kl_loss = dict()
        kl_loss["kl_loss_sum"] = torch.tensor(0., device=self.cfg.main.device)
        kl_loss["kl_loss"] = self._calc_kl(states)
        kl_loss["kl_loss_sum"] += kl_loss["kl_loss"]

        if self.cfg.rssm.obs_emb_kl_beta != 0:
            obs_emb_kl = torch.tensor(0., device=self.cfg.main.device)
            for key in states["obs_emb_subset_means"].keys():
                div = kl_divergence(Normal(states["obs_emb_subset_means"][key], states["obs_emb_subset_std_devs"][key]), 
                                    Normal(0,1)).sum(dim=2)
                obs_emb_kl += torch.max(div, self.free_nats).mean(dim=(0, 1))
            kl_loss["kl_loss_sum"] += self.cfg.rssm.obs_emb_kl_beta * obs_emb_kl / len(states["obs_emb_subset_means"].keys())
            kl_loss["obs_emb_kl"] = obs_emb_kl / len(states["obs_emb_subset_means"].keys())
        
        if self.cfg.rssm.global_kl_beta != 0:
            global_kl = kl_divergence(Normal(posterior_means, posterior_std_devs), 
                                      self.global_prior).sum(dim=2).mean(dim=(0, 1))
            kl_loss["kl_loss_sum"] += self.cfg.rssm.global_kl_beta * global_kl
            kl_loss["global_kl"] = global_kl
        
        # Calculate latent overshooting objective for t > 0
        if self.cfg.rssm.overshooting_kl_beta != 0:
            latent_overshooting_kl_loss, latent_overshooting_reward_loss = self._latent_overshooting(actions, rewards, nonterminals, states)
            kl_loss["kl_loss_sum"] += latent_overshooting_kl_loss
            kl_loss["latent_overshooting_kl_loss"] += latent_overshooting_kl_loss
            reward_loss += latent_overshooting_reward_loss
        # Apply linearly ramping learning rate schedule
        if self.cfg.rssm.learning_rate_schedule != 0:
            for group in self.model_optimizer.param_groups:
                group['lr'] = min(group['lr'] + self.cfg.rssm.model_learning_rate /
                                    self.cfg.rssm.learning_rate_schedule, self.cfg.rssm.model_learning_rate)
        
        if self.cfg.rssm.HF_PGM["pose_predict"]:
            predict_pose_loss = self._calc_predict_pose_loss(actions[1:], beliefs)
        
        if not self.cfg.rssm.predict_reward:
            reward_loss = torch.zeros_like(reward_loss)

        contrastive_loss = dict()
        contrastive_loss["contrastive_loss_sum"] = torch.tensor(0., device=self.cfg.main.device)
        if self.cfg.rssm.dreaming.use:
            contrastive_loss["dreaming_cpc_loss"] = self._calc_dreaming_loss(observations_contrastive,
                                                                            actions,
                                                                            rewards,
                                                                            nonterminals,
                                                                            posterior_states,
                                                                            )
            contrastive_loss["contrastive_loss_sum"] += contrastive_loss["dreaming_cpc_loss"]
        if self.cfg.rssm.MuMMI.use:
            MuMMI_loss = self._calc_MuMMI_loss(observations_target,
                                                beliefs,
                                                posterior_states)
            for key in MuMMI_loss.keys():
                contrastive_loss["MuMMI_loss_"+key] = MuMMI_loss[key]
            contrastive_loss["contrastive_loss_sum"] += MuMMI_loss["mean"]

        if self.cfg.rssm.HF_PGM["pose_predict"]:
            return observations_loss, reward_loss, kl_loss, contrastive_loss, predict_pose_loss
        else:
            return observations_loss, reward_loss, kl_loss, contrastive_loss

    def _get_model_loss(self,
                        observations_target, 
                        actions, 
                        rewards, 
                        nonterminals,
                        states,
                        observations_contrastive=None,
                        ):

        if self.cfg.rssm.HF_PGM["pose_predict"]:
            with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
                observations_loss, reward_loss, kl_loss, contrastive_loss, predict_pose_loss = self._calc_loss(observations_target, actions, rewards, nonterminals, states, observations_contrastive)

            model_loss = observations_loss["observations_loss_sum"] \
                        + reward_loss \
                        + contrastive_loss["contrastive_loss_sum"] \
                        + self.cfg.rssm.kl_beta*kl_loss["kl_loss_sum"]\
                        + self.cfg.rssm.HF_PGM["pose_predict_index"] * predict_pose_loss["predict_pose_loss"]
        else:
            with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
                observations_loss, reward_loss, kl_loss, contrastive_loss = self._calc_loss(observations_target, actions, rewards, nonterminals, states, observations_contrastive)
            
            model_loss = observations_loss["observations_loss_sum"] \
                        + reward_loss \
                        + contrastive_loss["contrastive_loss_sum"] \
                        + self.cfg.rssm.kl_beta*kl_loss["kl_loss_sum"]
                        
        
        # Log loss info
        loss_info = dict()
        if len(observations_loss.keys()) > 1:
            for name in observations_loss.keys():
                loss_info[name] = observations_loss[name].item()
        loss_info["reward_loss"] = reward_loss.item()
        if len(contrastive_loss.keys()) > 1:
            for key in contrastive_loss.keys():
                loss_info[key] = contrastive_loss[key].item()
        for key in kl_loss.keys():
            loss_info[key] = kl_loss[key].item()
        
        if self.cfg.rssm.HF_PGM["pose_predict"]:
            for key in predict_pose_loss.keys():
                loss_info[key] = predict_pose_loss[key].item()
        
        return model_loss, loss_info

    def _sample_data(self,
                     D,
                     ):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals, observations_contrastive = D.sample(
            self.cfg.train.batch_size, self.cfg.train.chunk_size, self.cfg.rssm.dreaming.use)  # Transitions start at time t = 0
        
        observations_target = self._clip_obs(observations, idx_start=1)
        if self.cfg.rssm.dreaming.use:
            observations_contrastive = self._clip_obs(observations_contrastive, idx_start=1)
        return observations_target, actions, rewards, nonterminals, observations_contrastive

    def optimize_loss(self,
                      observations_target, 
                      actions, 
                      rewards, 
                      nonterminals,
                      states,
                      itr_optim,
                      observations_contrastive=None,
                      ):
        model_loss, loss_info = self._get_model_loss(observations_target, actions, rewards, nonterminals, states, observations_contrastive)

        # Update model parameters
        self.model_optimizer.zero_grad()

        self.scaler.scale(model_loss).backward()
        nn.utils.clip_grad_norm_(self.param_list, self.cfg.rssm.grad_clip_norm, norm_type=2)
        self.scaler.step(self.model_optimizer)
        self.scaler.update()

        if self.cfg.main.wandb:
            for name in loss_info.keys():
                #print("train losses:", loss_info)
                wandb.log(data={"{}/train".format(name):loss_info[name]}, step=itr_optim)
            frame = itr_optim * self.cfg.train.batch_size * self.cfg.train.chunk_size
            wandb.log(data={"frame":frame}, step=itr_optim)


    #train
    def optimize(self,
                 D, 
                 ):
        self.itr_optim += 1

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            observations_target, actions, rewards, nonterminals, observations_contrastive = self._sample_data(D)
            states = self.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
        self.optimize_loss(observations_target, actions, rewards, nonterminals, states, self.itr_optim, observations_contrastive)

    def validation(self,
                 D, 
                 ):
        # self.eval()

        with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
            observations_target, actions, rewards, nonterminals, observations_contrastive = self._sample_data(D)
            states = self.estimate_state(observations_target, actions[:-1], rewards, nonterminals[:-1])
        _, loss_info = self._get_model_loss(observations_target, actions, rewards, nonterminals, states, observations_contrastive)
        
        if self.cfg.main.wandb:
            for name in loss_info.keys():        
                wandb.log(data={"{}/validation".format(name):loss_info[name]}, step=self.itr_optim)

        # self.train()

class MRSSM_base(RSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

    def eval(self):
        raise NotImplementedError
        
    def train(self):
        raise NotImplementedError

    def load_state_dict(self, model_dicts):
        raise NotImplementedError
    
    def get_state_dict(self):
        raise NotImplementedError
    
    def estimate_state(self,
                        observations,
                        actions, 
                        rewards, 
                        nonterminals,
                        batch_size=None,
                        det=False):
        raise NotImplementedError

    def _calc_observations_loss(self,
                               observations_target,
                               beliefs,
                               posterior_states,
                               ):
        observations_loss = dict()
        observations_loss["observations_loss_sum"] = torch.tensor(0., device=self.cfg.main.device)
        if len(self.cfg.rssm.observation_names_rec) > 0:
            if self.cfg.rssm.worldmodel_observation_loss == "log_prob":
                log_probs = self.observation_model.get_log_prob(beliefs, posterior_states, observations_target)
                for name in log_probs.keys():
                    observations_loss["observation_{}_loss".format(name)] = -log_probs[name].mean(dim=(0,1)).sum()
                    observations_loss["observations_loss_sum"] += observations_loss["observation_{}_loss".format(name)]
            elif self.cfg.rssm.worldmodel_observation_loss == "mse":
                mse = self.observation_model.get_mse(h_t=beliefs, s_t=posterior_states, o_t=observations_target)
                for name in mse.keys():
                    observations_loss["observation_{}_loss".format(name)] = mse[name].mean(dim=(0,1)).sum()
                    observations_loss["observations_loss_sum"] += observations_loss["observation_{}_loss".format(name)]
            elif self.cfg.rssm.worldmodel_observation_loss == "mae":
                mae = self.observation_model.get_mae(h_t=beliefs, s_t=posterior_states, o_t=observations_target)
                for name in mae.keys():
                    observations_loss["observation_{}_loss".format(name)] = mae[name].mean(dim=(0,1)).sum()
                    observations_loss["observations_loss_sum"] += observations_loss["observation_{}_loss".format(name)]
            else:
                raise NotImplementedError
        
        return observations_loss

    def get_obs_emb(self, observations):
        return bottle_tupele_multimodal(self.encoder, observations)

class MRSSM_st_base(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        if self.cfg.rssm.dreaming.use:
            raise NotImplementedError("dreaming loss is not implemented when fusion timing is st.")
        if self.cfg.rssm.MuMMI.use:
            raise NotImplementedError("MuMMI loss is not implemented when fusion timing is st.")
        
    def eval(self):
        self.encoder.eval()
        self.transition_model._eval()
        self.observation_model.eval()
        self.reward_model.eval()
        
    def train(self):
        self.encoder.train()
        self.transition_model._train()
        self.observation_model.train()
        self.reward_model.train()
    
    def _init_models(self, device):
        self.encoder = MultimodalEncoder(observation_names_enc=self.cfg.rssm.observation_names_enc,
                                        observation_shapes=self.cfg.env.observation_shapes,
                                        embedding_size=dict(self.cfg.rssm.embedding_size), 
                                        activation_function=dict(self.cfg.rssm.activation_function),
                                        normalization=self.cfg.rssm.normalization,
                                        device=device
                                        )

        self.transition_model = MultimodalTransitionModel(belief_size=self.cfg.rssm.belief_size, 
                                                            state_size=self.cfg.rssm.state_size,
                                                            action_size=self.cfg.env.action_size, 
                                                            hidden_size=self.cfg.rssm.hidden_size, 
                                                            observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                            embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                            device=device,
                                                            fusion_method=self.cfg.rssm.multimodal_params.fusion_method,
                                                            num_components=self.cfg.rssm.multimodal_params.num_components,
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
        
    def _init_model_modules(self):
        self.model_modules = self.encoder.modules \
                           + self.transition_model.modules \
                           + self.observation_model.modules \
                           + self.reward_model.modules
        
    def load_state_dict(self, model_dicts):
        self.encoder._load_state_dict(model_dicts['encoder'])
        self.transition_model._load_state_dict(model_dicts['transition_model'])
        self.observation_model._load_state_dict(model_dicts['observation_model'])        
        self.reward_model.load_state_dict(model_dicts['reward_model'])
        self.model_optimizer.load_state_dict(model_dicts['model_optimizer'])
        
    def _init_param_list(self):
        encoder_params = self.encoder.get_model_params()
        transition_model_params = self.transition_model.get_model_params()
        observation_model_params = self.observation_model.get_model_params()
        self.param_list = transition_model_params \
                        + observation_model_params \
                        + list(self.reward_model.parameters()) \
                        + encoder_params
    
    def get_state_dict(self):
        state_dict = {
                      'encoder': self.encoder.get_state_dict(),
                      'transition_model': self.transition_model.get_state_dict(),
                      'observation_model': self.observation_model.get_state_dict(),
                      'reward_model': self.reward_model.state_dict(),
                      'model_optimizer': self.model_optimizer.state_dict(),
                    }
        return state_dict
    
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
        
        obs_emb = bottle_tupele_multimodal(self.encoder, observations)

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
                     obs_emb=obs_emb,
                     )
        return states

class MRSSM_obs_emb_base(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
    
    def eval(self):
        self.encoder.eval()
        self.transition_model._eval()
        self.observation_model.eval()
        self.reward_model.eval()
        if self.cfg.rssm.dreaming.use:
            self.linear_predictor.eval()
    
    def train(self):
        self.encoder.train()
        self.transition_model._train()
        self.observation_model.train()
        self.reward_model.train()
        if self.cfg.rssm.dreaming.use:
            self.linear_predictor.train()
    
    def _init_models(self, device):
        self.encoder = MultimodalObservationEncoder(observation_names_enc=self.cfg.rssm.observation_names_enc,
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
                                                                device=device,
                                                                HFPGM_mode=self.cfg.rssm.HF_PGM["use"])
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

    def _init_model_modules(self):
        self.model_modules = self.encoder.modules \
                           + self.transition_model.modules \
                           + self.observation_model.modules \
                           + self.reward_model.modules
        if self.cfg.rssm.dreaming.use:
            self.model_modules + self.linear_predictor.modules
        if self.cfg.rssm.MuMMI.use:
           self. model_modules + self.feat2obs_emb.modules
    
    def load_state_dict(self, model_dicts):
        self.encoder._load_state_dict(model_dicts['encoder'])
        self.transition_model._load_state_dict(model_dicts['transition_model'])
        self.observation_model._load_state_dict(model_dicts['observation_model'])        
        self.reward_model.load_state_dict(model_dicts['reward_model'])
        self.model_optimizer.load_state_dict(model_dicts['model_optimizer'])
        if self.cfg.rssm.dreaming.use:
            self.linear_predictor.load_state_dict(model_dicts['linear_predictor'])
        if self.cfg.rssm.MuMMI.use:
            self.feat2obs_emb.load_state_dict(model_dicts['feat2obs_emb'])
    
    def _init_param_list(self):
        encoder_params = self.encoder.get_model_params()
        transition_model_params = self.transition_model.get_model_params()
        observation_model_params = self.observation_model.get_model_params()
        
        self.param_list = transition_model_params \
                + observation_model_params \
                + list(self.reward_model.parameters()) \
                + encoder_params
        if self.cfg.rssm.dreaming.use:
            self.param_list += list(self.linear_predictor.parameters())
        if self.cfg.rssm.MuMMI.use:
            self.param_list += list(self.feat2obs_emb.parameters())
            
    def get_state_dict(self):
        state_dict = {
                      'encoder': self.encoder.get_state_dict(),
                      'transition_model': self.transition_model.get_state_dict(),
                      'observation_model': self.observation_model.get_state_dict(),
                      'reward_model': self.reward_model.state_dict(),
                      'model_optimizer': self.model_optimizer.state_dict(),
                    }
        if self.cfg.rssm.dreaming.use:
            state_dict["linear_predictor"] = self.linear_predictor.state_dict()
        if self.cfg.rssm.MuMMI.use:
            state_dict["feat2obs_emb"] = self.feat2obs_emb.state_dict()
        
        return state_dict

    def get_obs_emb_posterior(self, observations):
        raise NotImplementedError

    def _calc_contrastive(self, feat_embed, obs_emb, kernel="mse"):
        x = obs_emb.reshape(-1, obs_emb.shape[-1]).transpose(1,0)
        z = feat_embed.reshape(-1, feat_embed.shape[-1])

        if kernel == "mse":
            z_prod = torch.sum(torch.square(z), dim=-1).reshape(-1,1)
            x_prod = torch.sum(torch.square(x), dim=-2).reshape(1,-1)
            weight_mat = 2 * torch.matmul(z, x) - z_prod - x_prod
        elif kernel == "biliner":
            weight_mat = torch.matmul(z, x)
        positive = torch.diag(weight_mat, 0)
        norm = torch.logsumexp(weight_mat, axis=1)

        info_nce = (positive - norm)
        return -info_nce
