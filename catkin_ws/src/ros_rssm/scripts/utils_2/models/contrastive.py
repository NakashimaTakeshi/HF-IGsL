from typing import Optional, List

import torch
from torch import nn

class LinearPredictor(nn.Module):
    def __init__(self, state_size: int, action_size: int, embedding_size: int, reward_size: int=1) -> None:
        super().__init__()
        self.reward_size = reward_size
        self.w_z_x = nn.Linear(embedding_size + reward_size, state_size, bias=False)
        self.w_za = nn.Linear(state_size + action_size, state_size, bias=False)
        self.modules = [self.w_z_x, self.w_za]

    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, nonterminals: Optional[torch.Tensor] = None, T: int = 3) -> List[torch.Tensor]:
        '''
        generate a sequence of data

        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                    torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        prior_states = [torch.empty(0)] * (T+1)
        prior_states[0] = prev_state

        # Loop over time sequence
        for t in range(T):
             # Select appropriate previous state
            if t == 0:
                _state = prior_states[t]
            else:
                _state = prior_states[t][:-t]
            _state = _state * nonterminals[t:]  # Mask if previous transition was terminal

            prior_states[t+1] = torch.zeros_like(prev_state)
            if t == 0:
                prior_states[t+1] = self.w_za(torch.cat([_state, actions[t+1:]], dim=-1))
            else:
                prior_states[t+1][:-t] = self.w_za(torch.cat([_state, actions[t+1:]], dim=-1))
            
        # Return new hidden states
        _prior_states = torch.stack(prior_states[1:], dim=2)
        
        return _prior_states
    
    def predict(self, obs_emb, rewards=None):
        if rewards is None:
            shape = obs_emb.shape[:-1]
            rewards = torch.zeros((*shape, self.reward_size), device=obs_emb.shape, dtype=torch.float32)
        return self.w_z_x(torch.cat([obs_emb, rewards], dim=-1))
    
    def get_feature(self, posterior_states, actions):
        return self.w_za(torch.cat([posterior_states, actions], dim=-1))

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

class Feature2ObsEmbed(nn.Module):
    def __init__(self, belief_size: int, state_size: int, embedding_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(belief_size + state_size, embedding_size, bias=False)
        self.modules = [self.fc]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor):
        x = torch.cat([h_t, s_t], dim=-1)
        return self.fc(x)
        
