import numpy as np

npy_path = "2023-01-21-14-39-17_1.npy"
max_length = 3000

epi_data = np.load(npy_path, allow_pickle=True).item()
epi_data_2 = dict()


for key in epi_data.keys():
    epi_data_2[key] = epi_data[key][300:max_length]
    if key=="done":
        epi_data_2[key][-1]=1
    print(key)
    print(len(epi_data_2[key]))
    print(epi_data_2)

np.save("epi_data_1.npy", epi_data_2, allow_pickle=True)


