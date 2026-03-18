import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os

files = ["dataset1.npy", "dataset2.npy"]

if not os.path.exists("images"):
    os.makedirs("images")

for file in files:
    name = file.replace(".npy", "")
    data = np.load(file)

    print("\nWorking on", name)
    print("Shape:", data.shape)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Task 1 - ground level
    counts, bins = np.histogram(z, bins=100)
    idx = np.argmax(counts)
    ground_level = (bins[idx] + bins[idx + 1]) / 2

    print("Ground level:", ground_level)

    plt.figure(figsize=(8, 5))
    plt.hist(z, bins=100, color="skyblue", edgecolor="black")
    plt.axvline(ground_level, color="red", linestyle="--", label=f"ground = {ground_level:.3f}")
    plt.xlabel("Z")
    plt.ylabel("Frequency")
    plt.title(f"Histogram - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{name}_histogram.png", dpi=200)
    plt.show()

    # keep points above ground
    data2 = data[data[:, 2] > ground_level]
    xy = data2[:, :2]

    print("Points above ground:", data2.shape)

    # Task 2 - elbow plot only for visual support
    nbrs = NearestNeighbors(n_neighbors=5)
    nbrs.fit(xy)
    distances, indices = nbrs.kneighbors(xy)

    k_dist = np.sort(distances[:, 4])

    # chosen eps manually
    if name == "dataset1":
        eps = 0.5
    else:
        eps = 0.5

    print("Chosen eps:", eps)

    plt.figure(figsize=(8, 5))
    plt.plot(k_dist)
    plt.axhline(eps, color="red", linestyle="--", label=f"chosen eps = {eps:.3f}")
    plt.xlabel("Sorted point index")
    plt.ylabel("5-NN distance")
    plt.title(f"Elbow plot - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{name}_elbow.png", dpi=200)
    plt.show()

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(xy)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print("Clusters found:", n_clusters)

    plt.figure(figsize=(8, 6))
    plt.scatter(data2[:, 0], data2[:, 1], c=labels, cmap="tab20", s=2)
    plt.colorbar(label="Cluster label")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"DBSCAN clusters - {name}")
    plt.tight_layout()
    plt.savefig(f"images/{name}_clusters.png", dpi=200)
    plt.show()

    # Task 3 - find largest cluster using x,y span
    best_label = None
    best_points = None
    best_span = -1

    for lab in unique_labels:
        if lab == -1:
            continue

        pts = data2[labels == lab]

        if len(pts) == 0:
            continue

        span_x = pts[:, 0].max() - pts[:, 0].min()
        span_y = pts[:, 1].max() - pts[:, 1].min()
        total_span = span_x + span_y

        if total_span > best_span:
            best_span = total_span
            best_label = lab
            best_points = pts

    if best_points is not None:
        min_x = best_points[:, 0].min()
        min_y = best_points[:, 1].min()
        max_x = best_points[:, 0].max()
        max_y = best_points[:, 1].max()

        print("Catenary cluster label:", best_label)
        print("min(x):", min_x)
        print("min(y):", min_y)
        print("max(x):", max_x)
        print("max(y):", max_y)

        plt.figure(figsize=(8, 6))
        plt.scatter(best_points[:, 0], best_points[:, 1], color="red", s=3)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Catenary cluster - {name}")
        plt.tight_layout()
        plt.savefig(f"images/{name}_catenary.png", dpi=200)
        plt.show()
    else:
        print("No valid cluster found")