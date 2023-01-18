from algos.MRSSM.RSSM.algo import RSSM
from algos.MRSSM.MRSSM_NN.algo import MRSSM_obs_emb_NN
from algos.MRSSM.MRSSM_PoE.algo import MRSSM_st_PoE, MRSSM_obs_emb_PoE
from algos.MRSSM.MRSSM_MoPoE.algo import MRSSM_st_MoPoE, MRSSM_obs_emb_MoPoE

def build_RSSM(cfg, device):
    if cfg.rssm.multimodal:
        if cfg.rssm.multimodal_params.fusion_timing == "stochastic_state":
            if cfg.rssm.multimodal_params.fusion_method == "NN":
                raise NotImplementedError
            elif cfg.rssm.multimodal_params.fusion_method == "PoE":
                rssm = MRSSM_st_PoE(cfg, device)
            elif "MoPoE" in cfg.rssm.multimodal_params.fusion_method:
                rssm = MRSSM_st_MoPoE(cfg, device)
            else:
                raise NotImplementedError
        elif cfg.rssm.multimodal_params.fusion_timing == "obs_emb":
            if cfg.rssm.multimodal_params.fusion_method == "NN":
                rssm = MRSSM_obs_emb_NN(cfg, device)
            elif cfg.rssm.multimodal_params.fusion_method == "PoE":
                rssm = MRSSM_obs_emb_PoE(cfg, device)
            elif "MoPoE" in cfg.rssm.multimodal_params.fusion_method:
                rssm = MRSSM_obs_emb_MoPoE(cfg, device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        rssm = RSSM(cfg, device)
    return rssm
