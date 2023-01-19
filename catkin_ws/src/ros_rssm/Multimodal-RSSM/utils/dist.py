import itertools
import numpy as np

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

def calc_kl_divergence(posterior_means, 
                       posterior_std_devs, 
                       prior_means, 
                       prior_std_devs, 
                       kl_balancing_alpha=None,
                       ):
  if kl_balancing_alpha is None:
    div = kl_divergence(Normal(posterior_means, posterior_std_devs), 
                        Normal(prior_means, prior_std_devs)).sum(dim=2)
  else:
    kl1 = kl_divergence(Normal(posterior_means.detach(), posterior_std_devs.detach()), 
                        Normal(prior_means, prior_std_devs)).sum(dim=2)
    kl2 = kl_divergence(Normal(posterior_means, posterior_std_devs), 
                        Normal(prior_means.detach(), prior_std_devs.detach())).sum(dim=2)
    div = kl_balancing_alpha * kl1 + (1-kl_balancing_alpha) * kl2
  return div


def poe(mu, scale):
    # precision of i-th Gaussian expert at point x
    T = 1. / scale
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_scale = 1. / torch.sum(T, dim=0)
    return pd_mu, pd_scale

def get_poe_params(expert_means,
                  expert_std_devs,
                  fusion_method="PoE",
                  num_components=None,
                  ):
    experts_loc = []
    experts_scale = []
    for name in expert_means.keys():
        experts_loc.append(expert_means[name])
        experts_scale.append(expert_std_devs[name])

    experts_loc = torch.stack(experts_loc)
    experts_scale = torch.stack(experts_scale)

    posterior_means, posterior_std_devs = poe(experts_loc, experts_scale)
    return posterior_means, posterior_std_devs

def get_poe_state(expert_means,
                  expert_std_devs,
                  fusion_method="PoE",
                  num_components=None,
                  ):
    posterior_means, posterior_std_devs = get_poe_params(expert_means,
                                                         expert_std_devs,
                                                         fusion_method,
                                                         num_components,
                                                         )
    posterior_states = Normal(posterior_means, posterior_std_devs).rsample()
    return posterior_states, posterior_means, posterior_std_devs

def _calc_subset_states(expert_means,
                       expert_std_devs,
                       ):
    expert_keys = list(expert_means.keys())
    if "prior_expert" in expert_keys:
        expert_keys.remove("prior_expert")
        prior_expert_means = expert_means["prior_expert"]
        prior_expert_std_devs = expert_std_devs["prior_expert"]
    else:
        prior_expert_means = torch.zeros_like(expert_means[expert_keys[0]])
        prior_expert_std_devs = torch.ones_like(expert_std_devs[expert_keys[0]])
    
    subset_means = dict()
    subset_std_devs = dict()

    for n in range(len(expert_keys)+1):
        combination = list(itertools.combinations(expert_keys, n))
        for experts in combination:
            name = "(prior"
            means = [prior_expert_means]
            std_devs = [prior_expert_std_devs]
            for expert in experts:
                name += ",{}".format(expert)
                means.append(expert_means[expert])
                std_devs.append(expert_std_devs[expert])
            name += ")"
            
            expert_loc = torch.stack(means)
            expert_scale = torch.stack(std_devs)
            subset_mean, subset_std_dev = poe(expert_loc, expert_scale)
            subset_means[name] = subset_mean
            subset_std_devs[name] = subset_std_dev
    return subset_means, subset_std_devs

def calc_subset_states(expert_means,
                       expert_std_devs,
                       ):
    subset_means, subset_std_devs = _calc_subset_states(expert_means, expert_std_devs)
    means = []
    std_devs = []
    print(subset_means.keys())
    for key in subset_means.keys():
        means.append(subset_means[key])
        std_devs.append(subset_std_devs[key])
    return means, std_devs

# mixture_component_selection is implemented in MoPoE-VAE (https://github.com/thomassutter/MoPoE/blob/023d3191e35e3d6e94cc9ce109125d553212ef14/utils/utils.py#L61)
def mixture_component_selection(subset_means,
                                subset_std_devs,
                                ):
    num_components = len(subset_means)
    num_samples = subset_means[0].shape[-1]
    w_modalities = (1/float(num_components))*torch.ones(num_components).to(subset_means[0].device)
    idx_start = []
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0
        else:
            i_start = int(idx_end[k-1])
        if k == w_modalities.shape[0]-1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)
    idx_end[-1] = num_samples
    posterior_means = torch.cat([subset_means[k][:, :, idx_start[k]:idx_end[k]] for k in range(w_modalities.shape[0])], dim=-1)
    posterior_std_devs = torch.cat([subset_std_devs[k][:, :, idx_start[k]:idx_end[k]] for k in range(w_modalities.shape[0])], dim=-1)
    return posterior_means, posterior_std_devs

def arithmetic_mean(subset_means,
                    subset_std_devs,
                    ):
    posterior_means = torch.stack(subset_means, axis=-1).mean(-1)
    posterior_std_devs = torch.stack(subset_std_devs, axis=-1).mean(-1)
    return posterior_means, posterior_std_devs

def gmm(subset_means,
        subset_std_devs,
        ):
    subset_means = torch.stack(subset_means, axis=-1)
    subset_std_devs = torch.stack(subset_std_devs, axis=-1)
    T,B,O, num_subset = subset_means.shape
    posterior_means = []
    posterior_std_devs = []
    idx_subset = np.random.randint(0,num_subset,(T,B))
    for t in range(T):
        means_T = []
        std_devs_T = []
        for b in range(B):
            means_T.append(subset_means[t,b,:,idx_subset[t,b]])
            std_devs_T.append(subset_std_devs[t,b,:,idx_subset[t,b]])
        posterior_means.append(torch.stack(means_T, axis=0))
        posterior_std_devs.append(torch.stack(std_devs_T, axis=0))
    
    posterior_means = torch.stack(posterior_means, axis=0)
    posterior_std_devs = torch.stack(posterior_std_devs, axis=0)
    posterior_states = Normal(posterior_means, posterior_std_devs).rsample()
    return posterior_states, posterior_means, posterior_std_devs

def get_slice_idx(num_samples, num_components, w_modalities):
    idx_start = []
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0
        else:
            i_start = int(idx_end[k-1])
        if k == w_modalities.shape[0]-1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)
    idx_end[-1] = num_samples
    return idx_start, idx_end

def gmm_component(subset_means,
                  subset_std_devs,
                  equal_division=True,
                  num_components=None,
                  subset_index = None,
                  ):
    subset_means = torch.stack(subset_means, axis=-1)
    subset_std_devs = torch.stack(subset_std_devs, axis=-1)
    T,B, num_samples, num_subset = subset_means.shape
    if num_components is None:
        num_components = num_subset
    elif num_components > num_samples:
        raise NotImplementedError("num_components is larger than num_samples.")
    
    w_modalities = (1/float(num_components))*torch.ones(num_components).to(subset_means.device)
    idx_start, idx_end = get_slice_idx(num_samples, num_components, w_modalities)
    rng = np.random.default_rng()
    if equal_division:
        # 各サブセットからサンプリングされる次元の数が同じになるようにランダムソートを使用
        idx_base = np.arange(num_subset)
        idx_subset = []
        for t in range(T*B):
            rng.shuffle(idx_base, axis=0)
            _idx_subset = np.hstack([idx_base]*(num_components//num_subset)+[idx_base[:num_components%num_subset]])
            rng.shuffle(_idx_subset, axis=0)
            idx_subset.append(_idx_subset)
        idx_subset = np.array(idx_subset).reshape(T,B,-1).astype(np.int32)
    else:
        # 各サブセットからサンプリングされる次元の数が同じことを保証しない
        idx_subset = np.random.randint(0,num_subset,(T,B,num_components))
    
    mask = torch.zeros_like(subset_means)
    for t in (range(T)):
        for b in range(B):
            for k in range(num_components):
                mask[t, b, idx_start[k]:idx_end[k],idx_subset[t,b,k]] = 1.
    posterior_means = (subset_means * mask.detach()).sum(dim=-1)
    posterior_std_devs = (subset_std_devs * mask.detach()).sum(dim=-1)    
    return posterior_means, posterior_std_devs

def crossmodal_inference(subset_means,
                        subset_std_devs,
                        subset_index=None,
                        ):
    subset_means = torch.stack(subset_means, axis=-1)
    subset_std_devs = torch.stack(subset_std_devs, axis=-1)
    # T, B, num_samples, num_subset = subset_means.shape
    
    posterior_means = subset_means[:,:,:,subset_index]
    posterior_std_devs = subset_std_devs[:,:,:,subset_index]
    return posterior_means, posterior_std_devs

def get_mopoe_params(expert_means,
                    expert_std_devs,
                    fusion_method="MoPoE",
                    num_components=None,
                    subset_index = None
                    ):
    subset_means, subset_std_devs = calc_subset_states(expert_means, expert_std_devs)
    if subset_index != None:
        return crossmodal_inference(subset_means, subset_std_devs, subset_index = subset_index)
    else:
        if fusion_method == "MoPoE":
            return gmm_component(subset_means, subset_std_devs, equal_division=False, num_components=1, subset_index = subset_index)
        elif fusion_method == "MoPoE_gmm_component":
            return gmm_component(subset_means, subset_std_devs, num_components=num_components, subset_index = subset_index)
        elif fusion_method == "MoPoE_select":
            return mixture_component_selection(subset_means, subset_std_devs)
        elif fusion_method == "MoPoE_mean":
            return arithmetic_mean(subset_means, subset_std_devs)
        else:
            raise NotImplementedError("{} is not implemented".format(fusion_method))

def get_mopoe_state(expert_means,
                    expert_std_devs,
                    fusion_method="MoPoE",
                    num_components=None,
                    ):
    posterior_means, posterior_std_devs = get_mopoe_params(expert_means,
                                                           expert_std_devs,
                                                           fusion_method,
                                                           num_components,
                                                           )
    posterior_states = Normal(posterior_means, posterior_std_devs).rsample()
    return posterior_states, posterior_means, posterior_std_devs

