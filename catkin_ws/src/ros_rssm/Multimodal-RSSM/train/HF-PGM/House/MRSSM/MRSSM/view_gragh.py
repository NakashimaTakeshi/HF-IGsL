import numpy as np
import matplotlib.pyplot as plt

ht_pca = np.load("ht_pca.npy", allow_pickle=True)
ht_tsne = np.load("ht_tsne.npy", allow_pickle=True)
room = np.load("room.npy", allow_pickle=True)
print(ht_pca.shape)

fig_tsne = plt.figure(figsize=(5,5))
#ax1 = fig_tsne.add_subplot(121, projection="3d")
ax2 = fig_tsne.add_subplot(111, projection="3d")
# ax1.set_aspect('equal')
# ax2.set_aspect('equal')
cmap = plt.get_cmap("tab10") 
# for i in tqdm(range(len(ht_tsne))):
#     ax1.scatter(ht_tsne[i][0], ht_tsne[i][1], color = cmap(room_label[i]), s = 2)
#     ax2.scatter(hx_recon_2d[i],hy_recon_2d[i], color = cmap(room_label[i]), s = 2)
for i in range(len(ht_tsne)-2):
    #ax1.plot(ht_tsne[i:i+2, 0], ht_tsne[i:i+2, 1], ht_tsne[i:i+2, 2], color = cmap(room[i]))
    ax2.plot(ht_pca[0, i:i+2],ht_pca[1, i:i+2],ht_pca[2, i:i+2], color = cmap(room[i]))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    print(ht_pca[2,i])
plt.show()