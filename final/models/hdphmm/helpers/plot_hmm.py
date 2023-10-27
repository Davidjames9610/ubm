import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from final import useful
from hmmlearn.hmm import GaussianHMM

def plot_ellipse(ax, mu, sigma, color):
    vals, vecs = np.linalg.eigh(sigma)
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, facecolor=color, edgecolor='black')  # edgecolor for better visibility
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.4)
    ellipse = ax.add_artist(ellipse)
    return ellipse


def plot_hmm_learn(data, hmm: GaussianHMM, percent=10, feature_a=0, feature_b=1):

    if len(data) > 1000:
        data = data[:1000]

    _, z = hmm.decode(data)

    states_set = set(np.arange(hmm.n_components))

    counts = np.zeros(len(states_set))
    for k in range(len(states_set)):
        counts[k] = np.sum(z == k)

    colors_array = useful.get_colors()

    if colors_array is not None and len(colors_array) >= len(states_set):
        colors_array = colors_array[:len(states_set)]
    else:
        colors_array = np.random.rand(len(states_set), 3)  # 3 for RGB values

    state_color_mapping = dict(zip(states_set, colors_array))
    # Get the colors for each data point based on its state
    colors_z = [state_color_mapping[state] for state in z]

    fig, ax = plt.subplots()

    plt.scatter(data[:, feature_a], data[:, feature_b], c=colors_z, marker='o')

    for state in range(len(states_set)):
        state_mean = np.array([hmm.means_[state][feature_a], hmm.means_[state][feature_b]])
        state_cov = np.diag(np.array([hmm.covars_[state][feature_a][feature_a], hmm.covars_[state][feature_b][feature_b]]))
        if counts[state] > (len(z) * percent / 100):
            plot_ellipse(plt.gca(), state_mean, state_cov, color=state_color_mapping[state])

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # plt.title('HMM scatter plot')

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5) for color in
               colors_array]
    labels = [f'Component {state}' for state in states_set]
    ax.legend(handles[:10], labels[:10], loc='upper right')
    plt.show()

# update this with PCA so can be used w/ more than 2 dimensions
def plot_hmm_data(data, ss, k, means, covars, counts=None, percent=10, feature_a=0, feature_b=1):

    states_set = set(ss)

    colors_array = useful.get_colors()

    if colors_array is not None and len(colors_array) >= len(states_set):
        colors_array = colors_array[:len(states_set)]
    else:
        colors_array = np.random.rand(len(states_set), 3)  # 3 for RGB values
    # Scatter plot
    # Create a mapping between states and colors
    state_color_mapping = dict(zip(states_set, colors_array))
    # Get the colors for each data point based on its state
    colors_z = [state_color_mapping[state] for state in ss]

    fig, ax = plt.subplots()

    plt.scatter(data[:, feature_a], data[:, feature_b], c=colors_z, marker='o')

    for state in range(k):
        state_mean = np.array([means[state][feature_a], means[state][feature_b]])
        state_cov = np.diag(np.array([covars[state][feature_a][feature_a], covars[state][feature_b][feature_b]]))
        if counts is not None:
            if counts[state] > (len(ss) * percent / 100):
                plot_ellipse(plt.gca(), state_mean, state_cov, color=state_color_mapping[state])
        else:
            plot_ellipse(plt.gca(), state_mean, state_cov, color=state_color_mapping[state])

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of HMM-generated Data')

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5) for color in colors_array]
    labels = [f'Component {state}' for state in states_set]
    ax.legend(handles[:10], labels[:10], loc='upper right')
    # Show colorbar for states
    # cbar = plt.colorbar()
    # cbar.set_label('State')

    # Show the plot
    plt.show()