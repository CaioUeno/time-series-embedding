from dtw import dtw
import matplotlib.pyplot as plt
import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_processing import from_nested_to_2d_array, from_nested_to_3d_numpy
from sklearn.manifold import TSNE


def load_from_sktime_univariate(file_path):

    """
    Load sktime univariate dataset (.ts) and return X and y as arrays.
    """

    # load X and y using sktime function
    X, y = load_from_tsfile_to_dataframe(file_path)

    # transform into numpy array
    X = from_nested_to_2d_array(X, return_numpy=True)

    # reshape to (# instances, length, 1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def plot_embedding(X, y, title=None, filename=None):

    """
    Two dimensional plot instances segmented by classes (color). Save it as filename (without extension).
    """

    # dimensional reduction
    two_dimensional = TSNE(n_components=2, init="pca", random_state=0).fit_transform(X)

    # normalize values into [0, 1]
    x_min, x_max = np.min(two_dimensional, 0), np.max(two_dimensional, 0)
    two_dimensional = (two_dimensional - x_min) / (x_max - x_min)

    # config colors 
    if len(np.unique(y)) == 2:
        list_colors = ["crimson", "mediumseagreen"]
    else:
        list_colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(y))))
    
    map_color = {label:list_colors[i] for i, label in enumerate(np.sort(np.unique(y)))}
    colors = list(map(lambda x: map_color[x], y))

    # plot and save figure
    plt.figure(figsize=(12, 12))
    plt.scatter(two_dimensional[:, 0], two_dimensional[:, 1], color=colors)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.savefig(f"{filename}.png")

# DTW distance
def distance_time_warping(a, b):

    """
    DTW distance between two time series a and b.
    """

    return dtw(a, b, distance_only=True).distance
