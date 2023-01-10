import numpy as np

npy_path = "2022-11-16-07-18-24.npy"
max_length = 1000

epi_data = np.load(npy_path, allow_pickle=True).item()
epi_data_2 = dict()


for key in epi_data.keys():
    epi_data_2[key] = epi_data[key][:max_length]
    if key=="done":
        epi_data_2[key][-1]=1
    print(key)
    print(len(epi_data_2[key]))

np.save("epi_data.npy", epi_data_2, allow_pickle=True)


