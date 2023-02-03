import numpy as np
import matplotlib.pyplot as plt
import tqdm

name="RSSM"
ht_pca = np.load("ht_pca_model0.npy", allow_pickle=True)
ht_tsne = np.load("ht_tsne_model0.npy", allow_pickle=True)
room = np.load("room_model1.npy", allow_pickle=True)
print(ht_pca.shape)

print(room.shape)

fig_tsne = plt.figure(figsize=(10,10))
# ax1 = fig_tsne.add_subplot(121, projection="3d")
# ax2 = fig_tsne.add_subplot(111, projection="3d")
ax1 = fig_tsne.add_subplot(222)
ax2 = fig_tsne.add_subplot(221)
ax3 = fig_tsne.add_subplot(224)
ax4 = fig_tsne.add_subplot(223)
# ax1.set_aspect('equal')
# ax2.set_aspect('equal')
cmap = np.array(['r', 'g', 'b', 'y', 'm', 'c'])

for i in range(len(ht_tsne)):
    ax1.scatter(ht_tsne[i][0], ht_tsne[i][1], color = cmap[room[i]], s = 2)
    ax2.scatter(ht_pca[0, i],ht_pca[1, i], color = cmap[room[i]], s = 2)

for i in range(len(ht_tsne)-2):
    # ax3.plot(ht_tsne[i:i+2, 0], ht_tsne[i:i+2, 1], ht_tsne[i:i+2, 2], color = cmap(room[i]))
    # ax4.plot(ht_pca[0, i:i+2],ht_pca[1, i:i+2],ht_pca[2, i:i+2], color = cmap(room[i]))
    ax3.plot(ht_tsne[i:i+2, 0], ht_tsne[i:i+2, 1], color = cmap[room[i]])
    ax4.plot(ht_pca[0, i:i+2],ht_pca[1, i:i+2], color = cmap[room[i]])

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax4.set_xlabel("x")
ax4.set_ylabel("y")



ax1.set_title("ht t-SNE ({})".format(name))
ax2.set_title("ht PCA ({}))".format(name))
ax3.set_title("ht t-SNE ({})".format(name))
ax4.set_title("ht PCA ({})".format(name))
# plt.show()
plt.savefig("{}.png".format(name))
plt.savefig("{}.pdf".format(name))