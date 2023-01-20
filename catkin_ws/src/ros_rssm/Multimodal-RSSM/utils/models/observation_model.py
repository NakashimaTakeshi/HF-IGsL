from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions

from torch.distributions import Normal
from utils.models.ResidualBlocks import ResidualBlock2dTransposeConv


class ObservationModel_base(nn.Module):
    def __init__(self,
                 belief_size: int,
                 state_size: int,
                 decoder,
                 mode,
                 activation_function: str = 'relu'
                 ):
        super().__init__()
        self.hf_pgm_use = mode
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = decoder.embedding_size

        if self.hf_pgm_use:
            self.fc = nn.Linear(state_size, self.embedding_size)
        else:
            self.fc = nn.Linear(belief_size + state_size, self.embedding_size)

        self.decoder = decoder
        self.modules = [self.fc, self.decoder]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        # No nonlinearity here
        if self.hf_pgm_use:
            hidden = self.act_fn(self.fc(s_t))
        else:
            hidden = self.act_fn(self.fc(torch.cat([h_t, s_t], dim=1)))
        # hidden = hidden.reshape(-1, self.embedding_size, 1, 1)

        observation = self.decoder(hidden)

        features_shape = observation.size()[1:]
        observation = observation.reshape(T, B, *features_shape)
        return {'loc': observation, 'scale': 1.0}

    def get_state_dict(self):
        state_dict = dict(fc=self.fc.state_dict(),
                          decoder=self.decoder.state_dict(),
                          )
        return state_dict
        # return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.fc.load_state_dict(state_dict["fc"])
        self.decoder.load_state_dict(state_dict["decoder"])
        # self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

    def get_log_prob(self, h_t, s_t, o_t):
        loc_and_scale = self.forward(h_t, s_t)
        dist = Normal(loc_and_scale['loc'], loc_and_scale['scale'])
        log_prob = dist.log_prob(o_t)
        return log_prob

    def get_mse(self, h_t, s_t, o_t):
        # print("h_t:",h_t)
        # print("s_t:",s_t)
        loc_and_scale = self.forward(h_t, s_t)
        # print("loc:",loc_and_scale['loc'])
        mse = F.mse_loss(loc_and_scale['loc'], o_t, reduction='none')
        return mse


class ObservationModel_with_scale_base(ObservationModel_base):
    def __init__(self,
                 belief_size: int,
                 state_size: int,
                 decoder,
                 mode,
                 activation_function: str = 'relu'
                 ):
        super().__init__(belief_size=belief_size,
                         state_size=state_size,
                         decoder=decoder,
                         mode=mode,
                         activation_function=activation_function)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        # No nonlinearity here
        if self.hf_pgm_use:
            hidden = self.act_fn(self.fc(s_t))
        else:
            hidden = self.act_fn(self.fc(torch.cat([h_t, s_t], dim=1)))
        # hidden = hidden.reshape(-1, self.embedding_size, 1, 1)

        loc, scale = self.decoder(hidden)

        features_shape = loc.size()[1:]

        loc = loc.reshape(T, B, *features_shape)
        scale = scale.reshape(T, B, *features_shape)
        return {'loc': loc, 'scale': scale}


class ObservationModel_dummy(ObservationModel_base):
    def __init__(self,
                 belief_size: int = 1,
                 state_size: int = 1,
                 decoder=None,
                 activation_function: str = 'relu') -> None:
        dummy_decoder = DenseDecoder(1, 1)
        super().__init__(belief_size, state_size, dummy_decoder, activation_function)
        self.modules = []


class DenseDecoder(nn.Module):
    def __init__(self,
                 observation_size: torch.Tensor,
                 embedding_size: int,
                 activation_function: str = 'relu'
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, observation_size)
        self.modules = [self.fc1, self.fc2]

    def forward(self, hidden):
        hidden = self.act_fn(self.fc1(hidden))
        observation = self.fc2(hidden)
        return observation


class PoseDecoder(nn.Module):
    def __init__(self,
                 observation_size: torch.Tensor,
                 embedding_size: int,
                 activation_function: str = 'relu',
                 min_std_dev: float = 0.1
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.min_std_dev = min_std_dev
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, observation_size*2)
        self.modules = [self.fc1, self.fc2]

    def forward(self, hidden):
        hidden = self.act_fn(self.fc1(hidden))
        x = self.fc2(hidden)
        loc, scale = torch.chunk(x, 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return loc, scale


class ImageDecoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 5, stride=2),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(128, 64, 5, stride=2),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(64, 32, 6, stride=2),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          32, image_dim, 6, stride=2)
                                      )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 5, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          128, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          128, 64, 5, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          64, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          64, 32, 6, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          32, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          32, image_dim, 6, stride=2)
                                      )
        else:
            raise NotImplementedError
        self.modules = [self.conv]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        return self.conv(hidden)

def res_block_gen(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=1.0, b_val=1.0):
    upsample = None
    if (kernelsize != 1 and stride != 1) or (in_channels != out_channels):
        upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                 nn.BatchNorm2d(out_channels))
    layers = []
    layers.append(ResidualBlock2dTransposeConv(in_channels, out_channels,
                                               kernelsize=kernelsize,
                                               stride=stride,
                                               padding=padding,
                                               dilation=dilation,
                                               o_padding=o_padding,
                                               upsample=upsample,
                                               a=a_val, b=b_val))
    return nn.Sequential(*layers)


class ImageDecoder_ResNet_64(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 DIM_img=128,
                 a=2.0,
                 b=0.3):
        super(ImageDecoder_ResNet_64, self).__init__()
        self.embedding_size = embedding_size
        self.a = a
        self.b = b
        self.fc = nn.Linear(embedding_size, 5*DIM_img, bias=True)
        self.resblock1 = res_block_gen(
            5*DIM_img, 4*DIM_img, kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock2 = res_block_gen(
            4*DIM_img, 3*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock3 = res_block_gen(
            3*DIM_img, 2*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock4 = res_block_gen(
            2*DIM_img, 1*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.conv = nn.ConvTranspose2d(DIM_img, image_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       output_padding=1)
        self.modules = [self.fc, self.resblock1, self.resblock2,
                        self.resblock3, self.resblock4, self.conv]

    def forward(self, feats):
        d = self.fc(feats)
        d = d.view(d.size(0), d.size(1), 1, 1)
        d = self.resblock1(d)
        d = self.resblock2(d)
        d = self.resblock3(d)
        d = self.resblock4(d)
        d = self.conv(d)
        return d


class ImageDecoder_ResNet_84(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 DIM_img=128,
                 a=2.0,
                 b=0.3):
        super(ImageDecoder_ResNet_84, self).__init__()
        self.embedding_size = embedding_size
        self.a = a
        self.b = b
        self.fc = nn.Linear(embedding_size, 5*DIM_img, bias=True)
        self.resblock1 = res_block_gen(
            5*DIM_img, 4*DIM_img, kernelsize=4, stride=2, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock2 = res_block_gen(
            4*DIM_img, 3*DIM_img, kernelsize=5, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock3 = res_block_gen(
            3*DIM_img, 2*DIM_img, kernelsize=6, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock4 = res_block_gen(
            2*DIM_img, 1*DIM_img, kernelsize=6, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.conv = nn.ConvTranspose2d(DIM_img, image_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       output_padding=1)
        self.modules = [self.fc, self.resblock1, self.resblock2,
                        self.resblock3, self.resblock4, self.conv]

    def forward(self, feats):
        d = self.fc(feats)
        d = d.view(d.size(0), d.size(1), 1, 1)
        d = self.resblock1(d)
        d = self.resblock2(d)
        d = self.resblock3(d)
        d = self.resblock4(d)
        d = self.conv(d)
        return d


class ImageDecoder_ResNet_128(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 DIM_img=128,
                 a=2.0,
                 b=0.3):
        super(ImageDecoder_ResNet_128, self).__init__()
        self.embedding_size = embedding_size
        self.a = a
        self.b = b
        self.fc = nn.Linear(embedding_size, 6*DIM_img, bias=True)
        self.resblock1 = res_block_gen(
            6*DIM_img, 5*DIM_img, kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock2 = res_block_gen(
            5*DIM_img, 4*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock3 = res_block_gen(
            4*DIM_img, 3*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock4 = res_block_gen(
            3*DIM_img, 2*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock5 = res_block_gen(
            2*DIM_img, 1*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.conv = nn.ConvTranspose2d(DIM_img, image_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       output_padding=1)
        self.modules = [self.fc, self.resblock1, self.resblock2,
                        self.resblock3, self.resblock4, self.resblock5, self.conv]

    def forward(self, feats):
        d = self.fc(feats)
        d = d.view(d.size(0), d.size(1), 1, 1)
        d = self.resblock1(d)
        d = self.resblock2(d)
        d = self.resblock3(d)
        d = self.resblock4(d)
        d = self.resblock5(d)
        d = self.conv(d)
        return d


class ImageDecoder_ResNet_256(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 DIM_img=64,
                 a=2.0,
                 b=0.3):
        super(ImageDecoder_ResNet_256, self).__init__()
        self.embedding_size = embedding_size
        self.a = a
        self.b = b
        self.fc = nn.Linear(embedding_size, 5*DIM_img, bias=True)
        self.resblock1 = res_block_gen(
            5*DIM_img, 5*DIM_img, kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock2 = res_block_gen(
            5*DIM_img, 5*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock3 = res_block_gen(
            5*DIM_img, 4*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock4 = res_block_gen(
            4*DIM_img, 3*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock5 = res_block_gen(
            3*DIM_img, 2*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)
        self.resblock6 = res_block_gen(
            2*DIM_img, 1*DIM_img, kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0, a_val=a, b_val=b)

        self.conv = nn.ConvTranspose2d(DIM_img, image_dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       output_padding=1)
        self.modules = [self.fc, self.resblock1, self.resblock2, self.resblock3,
                        self.resblock4, self.resblock5, self.resblock6, self.conv]

    def forward(self, feats):
        d = self.fc(feats)
        d = d.view(d.size(0), d.size(1), 1, 1)
        d = self.resblock1(d)
        d = self.resblock2(d)
        d = self.resblock3(d)
        d = self.resblock4(d)
        d = self.resblock5(d)
        d = self.resblock6(d)
        d = self.conv(d)
        return d


class ImageDecoder_84(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 3, stride=2),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(128, 64, 4, stride=2),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(64, 32, 4, stride=2),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(32, 16, 6, stride=2),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          16, image_dim, 6, stride=2)
                                      )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128, 3, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          128, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          128, 64, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          64, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          64, 32, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          32, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          32, 16, 6, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          16, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          16, image_dim, 6, stride=2)
                                      )
        else:
            raise NotImplementedError
        self.modules = [self.conv]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        return self.conv(hidden)


class ImageDecoder_128(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        scale = 2
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          128*scale, 64*scale, 4, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          64*scale, 32*scale, 4, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          32*scale, 16*scale, 4, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          16*scale, image_dim, 6, stride=2)
                                      )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          128*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          128*scale, 64*scale, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          64*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          64*scale, 32*scale, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          32*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          32*scale, 16*scale, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          16*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          16*scale, image_dim, 6, stride=2)
                                      )
        else:
            raise NotImplementedError
        self.modules = [self.conv]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        return self.conv(hidden)


class ImageDecoder_256(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 embedding_size: int,
                 image_dim=3,
                 normalization=None,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        scale = 2
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          128*scale, 64*scale, 4, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          64*scale, 32*scale, 4, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          32*scale, 16*scale, 4, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          16*scale, 8*scale, 4, stride=2, bias=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          8*scale, image_dim, 6, stride=2)
                                      )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.ConvTranspose2d(embedding_size, 128*scale, 6, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          128*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          128*scale, 64*scale, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          64*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          64*scale, 32*scale, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          32*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          32*scale, 16*scale, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          16*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          16*scale, 8*scale, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(
                                          8*scale, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(
                                          8*scale, image_dim, 6, stride=2)
                                      )
        else:
            raise NotImplementedError
        self.modules = [self.conv]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        return self.conv(hidden)


class SoundDecoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 embedding_size: int,
                 normalization=None,
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        if normalization == None:
            self.conv = nn.Sequential(nn.ConvTranspose2d(5, 64, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2)),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(32, 128, kernel_size=(
                                          5, 5), stride=(1, 1), padding=(1, 2)),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(64, 64, kernel_size=(
                                          4, 8), stride=(2, 2), padding=(1, 3)),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(32, 32, kernel_size=(
                                          4, 8), stride=(2, 2), padding=(1, 3)),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(16, 1, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4)))

        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.ConvTranspose2d(5, 64, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2), bias=False),
                                      nn.BatchNorm2d(
                                          64, affine=True, track_running_stats=True),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(32, 128, kernel_size=(
                                          5, 5), stride=(1, 1), padding=(1, 2), bias=False),
                                      nn.BatchNorm2d(
                                          128, affine=True, track_running_stats=True),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(64, 64, kernel_size=(
                                          4, 8), stride=(2, 2), padding=(1, 3), bias=False),
                                      nn.BatchNorm2d(
                                          64, affine=True, track_running_stats=True),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(32, 32, kernel_size=(
                                          4, 8), stride=(2, 2), padding=(1, 3), bias=False),
                                      nn.BatchNorm2d(
                                          32, affine=True, track_running_stats=True),
                                      nn.GLU(dim=1),
                                      nn.ConvTranspose2d(16, 1, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4), bias=False))

        else:
            raise NotImplementedError
        self.modules = [self.conv]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.reshape(-1, 5, 10, 5)
        observation = self.conv(hidden.reshape(-1, 5, 10, 5))
        return observation.squeeze(1)


# inspired by https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2/blob/master/model.py
class SoundDecoder_v2(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self,
                 embedding_size: int,
                 normalization=None,
                 channels_base=128,
                 ):
        super().__init__()
        self.channels_base = channels_base
        self.embedding_size = int(channels_base*2*32*4)

        # # Up-sampling layers.
        self.up_sample_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=int(channels_base*2), out_channels=int(
                channels_base*4), kernel_size=(3, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=int(channels_base*4),
                           affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=int(channels_base*2), out_channels=int(
                channels_base*2), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=int(channels_base*2),
                           affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels_base, out_channels=channels_base,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels_base,
                           affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Out.
        self.out = nn.Conv2d(in_channels=int(
            channels_base/2), out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

        self.modules = [self.up_sample_0,
                        self.up_sample_1, self.up_sample_2, self.out]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.view(-1, int(self.channels_base*2), 32, 4)
        x = self.up_sample_0(x)
        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        observation = self.out(x)
        return observation.squeeze(1)

def build_ObservationModel(name, observation_shapes, belief_size, state_size, hidden_size, embedding_size, activation_function, mode, normalization=None, use_ResNet=False):
    if "image" in name:
        image_size = observation_shapes[name][1:]
        image_dim = observation_shapes[name][0]
        if use_ResNet:
            if image_size == [256, 256]:
                decoder = ImageDecoder_ResNet_256(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
            elif image_size == [128, 128]:
                decoder = ImageDecoder_ResNet_128(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
            elif image_size == [84, 84]:
                decoder = ImageDecoder_ResNet_84(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
            elif image_size == [64, 64]:
                decoder = ImageDecoder_ResNet_64(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
        else:
            if image_size == [256, 256]:
                decoder = ImageDecoder_256(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
            elif image_size == [128, 128]:
                decoder = ImageDecoder_128(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
            elif image_size == [84, 84]:
                decoder = ImageDecoder_84(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
            elif image_size == [64, 64]:
                decoder = ImageDecoder(
                    embedding_size["image"], image_dim=image_dim, normalization=normalization)
                observation_models = ObservationModel_base(
                    belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["cnn"])
    elif "sound" in name:
        decoder = SoundDecoder_v2(
            embedding_size["sound"], normalization=normalization)
        observation_models = ObservationModel_base(
            belief_size=belief_size, state_size=state_size, decoder=decoder, activation_function=activation_function["cnn"])
    elif "melody" in name:
        decoder = MelodyDecoder(
            observation_shapes[name][0], embedding_size["other"], activation_function["dense"])
        observation_models = ObservationModel_base(
            belief_size=belief_size, state_size=state_size, decoder=decoder, activation_function=activation_function["dense"])
    elif "Pose" == name:
        decoder = PoseDecoder(
            observation_shapes[name][0], embedding_size["other"], activation_function["dense"])
        observation_models = ObservationModel_with_scale_base(
            belief_size=belief_size, state_size=state_size, decoder=decoder, mode=mode, activation_function=activation_function["dense"])
    else:
        decoder = DenseDecoder(
            observation_shapes[name][0], embedding_size["other"], activation_function["dense"])
        observation_models = ObservationModel_base(
            belief_size=belief_size, state_size=state_size, decoder=decoder, activation_function=activation_function["dense"])
    return observation_models


class MultimodalObservationModel:
    __constants__ = ['embedding_size']

    def __init__(self,
                 observation_names_rec,
                 observation_shapes,
                 embedding_size,
                 belief_size: int,
                 state_size: int,
                 hidden_size: int,
                 activation_function,
                 normalization=None,
                 device=torch.device("cpu"),
                 HFPGM_mode=False):
        self.observation_names_rec = observation_names_rec

        self.observation_models = dict()
        self.modules = []
        for name in self.observation_names_rec:
            self.observation_models[name] = build_ObservationModel(
                name, observation_shapes, belief_size, state_size, hidden_size, embedding_size, activation_function, normalization=normalization, mode=HFPGM_mode).to(device)
            self.modules += self.observation_models[name].modules

    def __call__(self, h_t: torch.Tensor, s_t: torch.Tensor):
        return self.forward(h_t, s_t)

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor):
        preds = dict()
        for name in self.observation_models.keys():
            pred = self.observation_models[name](h_t, s_t)
            preds[name] = pred

        return preds

    def get_log_prob(self, h_t, s_t, o_t):
        observation_log_prob = dict()
        for name in self.observation_names_rec:
            log_prob = self.observation_models[name].get_log_prob(
                h_t, s_t, o_t[name])
            observation_log_prob[name] = log_prob
        return observation_log_prob

    def get_mse(self, h_t, s_t, o_t):
        observation_mse = dict()
        for name in self.observation_names_rec:
            mse = self.observation_models[name].get_mse(h_t, s_t, o_t[name])
            observation_mse[name] = mse
        return observation_mse

    def get_pred_value(self, h_t: torch.Tensor, s_t: torch.Tensor, key):
        return self.observation_models[key](h_t, s_t)

    def get_pred_key(self, h_t: torch.Tensor, s_t: torch.Tensor, key):
        return self.get_pred_value(h_t, s_t, key)

    def get_state_dict(self):
        observation_model_state_dict = dict()
        for name in self.observation_models.keys():
            observation_model_state_dict[name] = self.observation_models[name].get_state_dict(
            )
        return observation_model_state_dict

    def _load_state_dict(self, state_dict):
        for name in self.observation_names_rec:
            self.observation_models[name]._load_state_dict(state_dict[name])

    def get_model_params(self):
        model_params = []
        for model in self.observation_models.values():
            model_params += list(model.parameters())
        return model_params

    def eval(self):
        for name in self.observation_models.keys():
            self.observation_models[name].eval()

    def train(self):
        for name in self.observation_models.keys():
            self.observation_models[name].train()
