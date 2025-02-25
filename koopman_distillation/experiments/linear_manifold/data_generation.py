import numpy as np
from matplotlib import pyplot as plt

centroids = np.array([
    [-3, 3], [3, 3], [-3, -3], [3, -3], [-6, 0], [6, 0],
    [-4, 5], [4, 5], [-4, -5], [4, -5], [-7, 2], [7, 2]
])


def split_gaussian_distribution(n_samples=3000, n_clusters=6):
    """
    Generates a Gaussian distribution with 'n_samples' points and splits it into 'n_clusters' sections.

    Args:
        n_samples (int): Number of points to sample from the Gaussian distribution.
        n_clusters (int): Number of clusters to split the distribution into.

    Returns:
        gaussian_points (numpy.ndarray): The original Gaussian-distributed points.
        cluster_labels (numpy.ndarray): Cluster labels for visualization.
    """
    # Generate Gaussian-distributed points
    gaussian_points = np.random.randn(n_samples, 2) * 2  # Spread out the Gaussian distribution

    # Use KMeans to split the distribution into clusters
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(gaussian_points)

    return gaussian_points, cluster_labels


def get_non_cont_manifold_data(plot=False):
    groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]

    gaussian_points, cluster_labels = split_gaussian_distribution(30000, 12)
    origin_pairs = list(zip(gaussian_points, cluster_labels))

    data = []
    for p, l in origin_pairs:
        if l in groups[0]:
            # get a random point either for the first or second Gaussian centroid
            random_centroid = centroids[np.random.choice([0, 1])] + np.random.randn(2) * 0.5
            data.append((p, random_centroid, 0))
        elif l in groups[1]:
            random_centroid = centroids[np.random.choice([2, 3])] + np.random.randn(2) * 0.5
            data.append((p, random_centroid, 1))
        elif l in groups[2]:
            random_centroid = centroids[np.random.choice([4, 5])] + np.random.randn(2) * 0.5
            data.append((p, random_centroid, 2))
        elif l in groups[3]:
            random_centroid = centroids[np.random.choice([6, 7])] + np.random.randn(2) * 0.5
            data.append((p, random_centroid, 3))
        elif l in groups[4]:
            random_centroid = centroids[np.random.choice([8, 9])] + np.random.randn(2) * 0.5
            data.append((p, random_centroid, 4))
        elif l in groups[5]:
            random_centroid = centroids[np.random.choice([10, 11])] + np.random.randn(2) * 0.5
            data.append((p, random_centroid, 5))
        else:
            raise ValueError(f" {l} is in Invalid group")

    if plot:
        # plot in two subplots the original and the target points
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.scatter([data[i][0][0] for i in range(len(data))], [data[i][0][1] for i in range(len(data))],
                    c=[data[i][2] for i in range(len(data))], cmap="tab10", alpha=0.7)
        ax1.set_title("Original Points")
        ax1.set_xlabel("X-axis")
        ax1.set_ylabel("Y-axis")
        ax2.scatter([data[i][1][0] for i in range(len(data))], [data[i][1][1] for i in range(len(data))],
                    c=[data[i][2] for i in range(len(data))], cmap="tab10", alpha=0.7)
        ax2.set_title("Target Points")
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        plt.show()

    return data