#!/usr/bin/env python3
import os
import sys
import numpy as np
import warnings
import math
import torch
import torchaudio
import librosa

def wav2mlsp_converter(wav_list, library="librosa", device="cpu"):
    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    n_mels = 128
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))
    top_db = 80.0
    multiplier = 10.0
    amin = 1e-10
    ref_value = np.max
    # db_multiplier = math.log10(max(amin, ref_value))
    device = torch.device(device)
    if library=="torchaudio":
        trans_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=fft_size,
            win_length=None,
            hop_length=hop_length,
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
        ).to(device=device)
    
    sound_array = []
    if library == "librosa":
        for i in range(len(wav_list)):
            mlsp = librosa.feature.melspectrogram(y=wav_list[i], sr=sr, n_fft=fft_size, hop_length=hop_length, htk=False)
            mlsp = librosa.power_to_db(mlsp, ref=ref_value)
            sound_array.append(mlsp[:, :frame_num])
        # sound preprocess [-0 ~ -80] -> [0 ~ 1]
        sound_array = np.array(sound_array).astype(np.float32)
        sound_array = np.divide(np.abs(sound_array), 80).astype(np.float32)
    elif library == "torchaudio":
        for i in range(len(wav_list)):
            temp = torch.FloatTensor(wav_list[i]).to(device=device)
            mlsp_power = trans_mel(temp)
            ref_value = mlsp_power.max(dim=1)[0].max(dim=0)[0]
            mlsp = torchaudio.functional.amplitude_to_DB(trans_mel(temp), multiplier, amin, math.log10(max(amin, ref_value)), top_db)
            # sound preprocess [-0 ~ -80] -> [0 ~ 1]
            mlsp = torch.narrow(mlsp.abs().float().div_(80), 1, 0, frame_num).cpu().detach().numpy()
            sound_array.append(mlsp)
        del trans_mel, temp, mlsp_power, ref_value
    else:
        print("Error : please select library torchaudio or librosa")
        raise NotImplementedError()
    return np.array(sound_array)

def compare_librosa_torchaudio(data):
    mlsp_librosa = wav2mlsp_converter(data["sound"])
    mlsp_torchaudio = wav2mlsp_converter(data["sound"], library="torchaudio", device="cpu")
    mse_loss = torch.nn.functional.mse_loss(torch.FloatTensor(mlsp_librosa), torch.FloatTensor(mlsp_torchaudio))
    print(mse_loss)

if __name__ == '__main__':
    npy_path = sys.argv[1]
    
    if ".bag" == os.path.splitext(npy_path)[1]:
        npy_path = os.path.splitext(npy_path)[0]+".npy"
    
    data = np.load(npy_path, allow_pickle=True, encoding="latin1").item()

    if "sound" in data.keys():
        data["sound"] = wav2mlsp_converter(data["sound"], library="torchaudio", device="cpu")
        print("waveform convert to mel-spectrogram")
        print("sound: {}".format(data["sound"].shape))
        np.save(npy_path, data)
    else:
        print("Sound Not Found")
