import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

image_raw = imread("ala_lab.jpg")
print(image_raw.shape)

image_vector = np.array(image_raw.shape)
print("Image vector [H, W, C]:", image_vector)

plt.imshow(image_raw)
plt.axis("off")
plt.title("Original image")
plt.show()

#2
image_sum = image_raw.sum(axis=2)
print("image_sum.shape:", image_sum.shape)

image_bw = image_sum / image_sum.max()
print("image_bw.max():", image_bw.max())

plt.imshow(image_bw, cmap="gray")
plt.axis("off")
plt.title("Black and white img")
plt.show()

#3
H, W = image_bw.shape
pca_full = PCA()
pca_full.fit(image_bw)
explained_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(explained_var) * 100
n_components_95 = np.argmax(cum_var >= 95) + 1
print("Number of components for 95% variance:", n_components_95)
plt.figure(figsize=(6,4))
plt.plot(cum_var)
plt.axhline(95, linestyle="--", color="r")
plt.axvline(n_components_95, linestyle="--", color="g")
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance (%)")
plt.title("Cumulative Explained Variance")
plt.grid(True)
plt.show()

def pca_compress(image_bw, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(image_bw)
    X_reconstructed = pca.inverse_transform(X_reduced)
    return X_reconstructed

X_reconstructed_95 = pca_compress(image_bw, n_components_95)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_bw, cmap="gray")
axes[0].set_title("Original BW")
axes[0].axis("off")

axes[1].imshow(X_reconstructed_95, cmap="gray")
axes[1].set_title(f"Reconstructed (95%, {n_components_95} comps)")
axes[1].axis("off")

plt.tight_layout()
plt.show()

#4
components_list = [5, 15, 25, 75, 100, 170]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, k in zip(axes.ravel(), components_list):
    X_rec = pca_compress(image_bw, k)
    ax.imshow(X_rec, cmap="gray")
    ax.set_title(f"Components: {k}")
    ax.axis("off")

plt.tight_layout()
plt.show()


