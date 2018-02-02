import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

RS = 20150101
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

digits = load_digits()

# nrows, ncols = 2, 5
# plt.figure(figsize=(6,3))
# plt.gray()
# for i in range(ncols * nrows):
#     ax = plt.subplot(nrows, ncols, i + 1)
#     ax.matshow(digits.images[i,...])
#     plt.xticks([]); plt.yticks([])
#     plt.title(digits.target[i])

X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])


y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])

# pca = PCA(n_components=10)
# pca.fit(X)
# X_new = pca.transform(X)
digits_proj = TSNE(random_state=RS).fit_transform(X)


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    
    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
scatter(digits_proj, y)
plt.savefig('digits_tsne-generated.png', dpi=120)
