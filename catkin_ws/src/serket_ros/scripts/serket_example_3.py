# -*- coding: utf-8 -*-

################################################################################

# Example 3: VAE+GMM+MLDA+MM.

################################################################################

import sys
sys.path.append("../Serket/")

# Import modules.
import numpy as np
import tensorflow as tf
import serket as srk
import vae
import gmm
import mlda
import mm

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score

import zipfile

################################################################################

# Extract dataset. 
with zipfile.ZipFile("../Serket/examples/VAE+GMM+MLDA+MM/data.zip") as z:
    z.extractall()

data1 = np.loadtxt("data/data1.txt")
data2 = np.loadtxt("data/data2.txt")
labels = np.loadtxt( "data/category.txt" )

# Check image data.
for i in range(20):
    plt.subplot( 2, 10, i+1 )
    plt.imshow( data1[i].reshape(28,28), cmap="gray" )
plt.show()

# Check speech signal data.
# HAC features (https://www.isca-speech.org/archive/interspeech_2008/i08_2554.html).
plt.bar( range(2400), data2[0] )
plt.show()

################################################################################

# Construct and train the integrated model.

# Define VAE architecture.
encoder_dim = 128
decoder_dim = 128
class vae_model(vae.VAE):
    def build_encoder(self, x, latent_dim):
        h_encoder = tf.keras.layers.Dense(encoder_dim, activation="relu")(x)

        mu = tf.keras.layers.Dense(latent_dim)(h_encoder)
        logvar = tf.keras.layers.Dense(latent_dim)(h_encoder)
        
        return mu, logvar
    
    def build_decoder(self, z):
        h_decoder = tf.keras.layers.Dense(decoder_dim, activation="relu")(z)
        logits = tf.keras.layers.Dense(784)(h_decoder)

        optimizer = tf.train.AdamOptimizer()
        
        return logits, optimizer

# Create instances of the modules.
obs1 = srk.Observation( data1 )
obs2 = srk.Observation( data2 )

vae1 = vae_model( 18, epoch=200, batch_size=500 )
gmm1 = gmm.GMM( 10 )
mlda1 = mlda.MLDA( 10, [200,200] )
mm1 = mm.MarkovModel()

# Connect modules.
vae1.connect( obs1 )
gmm1.connect( vae1 )
mlda1.connect( obs2, gmm1 )
mm1.connect( mlda1 )

# Train the integrated model.
for i in range(1):
    print( i )
    vae1.update()
    gmm1.update()
    mlda1.update()
    mm1.update()

################################################################################

# Evaluation 1: Latent space of VAE.

# Get the latent variables.
feats = vae1.get_forward_msg()

# Compress 18 dims to 2 dims by PCA.
pca = PCA(n_components=2)
pca.fit(feats)
feats_2dim = pca.transform(feats)

# Visualization.
for num in range(10):
    f = feats_2dim[num==labels]
    plt.plot( f[:,0], f[:,1], "o", label=str(num) )

plt.legend()
plt.show()

################################################################################

# Evaluation 2: Classification accuracy.

# Get the probabilities of classification results.
preds = mlda1.get_forward_msg()

# Convert them to the labels by argmax operation.
pred_labels = np.argmax( preds, axis=1 )

# Compute the score.
print("ARI:", adjusted_rand_score(pred_labels, labels))
