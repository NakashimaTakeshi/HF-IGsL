from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions

from torch.distributions import Normal

class PosePredictModel(nn.Module):
    def __init__(self, 
                 belief_size: int, 
                 action_size: int,
                 activation_function: str = 'relu',
                 min_std_dev: float = 0.1
                 ):
        super().__init__()
        self.min_std_dev = min_std_dev
        self.act_fn = getattr(F, activation_function)

        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size, belief_size)
        self.fc2 = nn.Linear(belief_size, belief_size)
        self.fc3 = nn.Linear(belief_size, action_size*2)
        self.modules = [self.fc1, self.fc2, self.fc3]
    

    def forward(self, h_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        # No nonlinearity here
        hidden = self.act_fn(self.fc1(h_t))
        hidden = self.act_fn(self.fc2(hidden))

        features_shape = hidden.size()[1:]
        hidden = hidden.reshape(T, B, *features_shape)

        loc, scale = torch.chunk(self.fc3(hidden), 2, dim=-1)
        scale = F.softplus(scale) + self.min_std_dev
        # hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        return {"loc": loc, "scale": scale}
    
    def get_state_dict(self):
        state_dict = dict(fc1=self.fc1.state_dict(),
                          fc2=self.fc2.state_dict(),
                          fc3=self.fc3.state_dict(),
                          )
        return state_dict
        # return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.fc1.load_state_dict(state_dict["fc1"])
        self.fc2.load_state_dict(state_dict["fc2"])
        self.fc3.load_state_dict(state_dict["fc3"])
        # self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

    def get_log_prob(self, h_t, a_t):
        loc_and_scale = self.forward(h_t)
        dist = Normal(loc_and_scale['loc'], loc_and_scale['scale'])
        log_prob = dist.log_prob(a_t)
        return log_prob