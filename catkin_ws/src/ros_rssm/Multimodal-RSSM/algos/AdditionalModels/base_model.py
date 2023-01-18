import os

import torch

from utils.models.encoder import bottle_tupele, bottle_tupele_multimodal
from algos.MRSSM.MRSSM.algo import build_RSSM

class Model_base:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.rssm = build_RSSM(cfg, device)
        if cfg.rssm.multimodal:
            self.bottle_tupele = bottle_tupele_multimodal
        else:
            self.bottle_tupele = bottle_tupele

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
        self.itr_optim = 0
    
    def eval(self):
        raise NotImplementedError
        
    def train(self):
        raise NotImplementedError
        
    def init_models(self, device):
        raise NotImplementedError
        
    def init_optimizer(self):
        raise NotImplementedError
    
    def load_model_dicts(self, model_path):
        print("load model_dicts from {}".format(model_path))
        return torch.load(model_path, map_location=torch.device(self.device))

    def load_rssm(self, model_path):
        model_dicts = self.load_model_dicts(model_path)
        self.rssm.load_state_dict(model_dicts)
        try:
            self.rssm.load_state_dict(model_dicts)
        except:
            self.rssm.load_state_dict(model_dicts["rssm"])
        self.rssm._init_optimizer()
    
    def load_state_dict(self, state_dict):
        raise NotImplementedError
    
    def load_model(self, model_path):
        model_dicts = self.load_model_dicts(model_path)
        self.load_state_dict(model_dicts)
        self.init_optimizer()

    def get_state_dict(self):
        raise NotImplementedError

    def save_model(self, results_dir, itr):
        state_dict = self.get_state_dict()
        torch.save(state_dict, os.path.join(results_dir, 'models_%d.pth' % itr))

    def _clip_obs(self, observations, idx_start=0, idx_end=None):
        return self.rssm._clip_obs(observations, idx_start, idx_end)

    def observation2np(self, observation):
        return self.rssm.observation2np(observation)

    def estimate_state(self, observations, actions, rewards, nonterminals):
        return self.rssm.estimate_state(observations, actions, rewards, nonterminals)

    def calc_loss(self,
                  states,
                  actions,
                  rewards,
                  nonterminals):
        raise NotImplementedError

    def optimize_loss(self, 
                      states,
                      actions, 
                      reawrds,
                      nonterminals,
                      itr_optim):
        raise NotImplementedError
            
    def optimize(self, D):
        raise NotImplementedError
    
    def validation(self, D):
        raise NotImplementedError
