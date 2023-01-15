import os
from tqdm import tqdm

from utils.logger import setup_experiment, stop_logger
from utils.replay_buffer.memory import ExperienceReplay_Multimodal, load_dataset

from .algo import ObservationModelLearning

def get_dataset_loader(cfg, cwd, device, dataset_path):
    observation_names = list(set(cfg.rssm.observation_names_enc+cfg.rssm.observation_names_rec))

    D = ExperienceReplay_Multimodal(size=cfg.train.experience_size,
                                    observation_names=observation_names,
                                    observation_shapes=cfg.env.observation_shapes,
                                    n_crop=cfg.train.augmentation.n_crop,
                                    dh_base=cfg.train.augmentation.dh_base,
                                    dw_base=cfg.train.augmentation.dw_base,
                                    noise_scales=cfg.train.augmentation.noise_scales,
                                    pca_scales=cfg.train.augmentation.pca_scales,
                                    action_name=cfg.env.action_name,
                                    action_size=cfg.env.action_size,
                                    bit_depth=cfg.env.bit_depth,
                                    device=device)
    load_dataset(cfg, cwd, D, dataset_path)
    return D

def train(cfg, cwd, results_dir, model_class, device):
    print("Initialise training environment and experience replay memory")
    
    # load train data
    D = get_dataset_loader(cfg, cwd, device, cfg.train.train_data_path)

    # load validation data
    D_val = get_dataset_loader(cfg, cwd, device, cfg.train.validation_data_path)

    # Initialise model parameters randomly
    print("Initialise model parameters randomly")
    model = model_class(cfg, device)
    
    if cfg.train.model_path != None:
        model_path = os.path.join(cwd, cfg.train.model_path)
        if os.path.exists(model_path):
            model.load_model(model_path)
        else:
            raise NotImplementedError("{} is not exist".format(model_path))
    
    if cfg.additional.rssm.load:
        if cfg.additional.rssm.model_path==None:
            raise NotImplementedError
        model_path = os.path.join(cwd, cfg.additional.rssm.model_path)
        if os.path.exists(model_path):
            print("load RSSM from {}".format(model_path))
            model.load_rssm(model_path)
        else:
            raise NotImplementedError("{} is not exist".format(model_path))
    
    for itr in tqdm(range(1, cfg.train.train_iteration+1), desc="train"):
        # optimize model
        model.optimize(D)
        
        # Validation
        if (not cfg.train.validation_interval is None) and (not cfg.train.validation_data_path is None) and (itr % cfg.train.validation_interval == 0):
            model.validation(D_val)

        # Checkpoint models
        if itr % cfg.train.checkpoint_interval == 0:
            model.save_model(results_dir, itr)


def run_base(cfg, model_class):
    # init logger and set seeds
    cwd, results_dir, device = setup_experiment(cfg)

    # ---------- train RSSM ----------
    train(cfg, cwd, results_dir, model_class, device)

    # ---------- ML Flow log ----------
    stop_logger(cfg)

def run(cfg):
    run_base(cfg, ObservationModelLearning)