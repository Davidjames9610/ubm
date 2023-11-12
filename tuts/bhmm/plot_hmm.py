import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from final import useful


def plot_ellipse(ax, mu, sigma, color="b"):
    vals, vecs = np.linalg.eigh(sigma)
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, facecolor=color, edgecolor='black')  # edgecolor for better visibility
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ellipse = ax.add_artist(ellipse)
    return ellipse

def plot_hmm_data(data, ss, k, means, covars):
    states_set = set(np.arange(k))

    colors_array = useful.get_colors()

    if colors_array is not None and len(colors_array) >= len(states_set):
        colors_array = colors_array[:len(states_set)]
    else:
        colors_array = np.random.rand(len(states_set), 3)  # 3 for RGB values

    state_color_mapping = dict(zip(states_set, colors_array))

    colors_z = [state_color_mapping[state] for state in ss]

    # Scatter plot
    plt.scatter(data[:, 0], data[:, 1], c=colors_z, cmap='viridis', marker='o')

    for state in range(k):
        state_data = data[ss == state]
        state_mean = means[state]
        state_cov = covars[state]
        plot_ellipse(plt.gca(), state_mean, state_cov, color=state_color_mapping[state])

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of HMM-generated Data')

    # Show colorbar for states
    cbar = plt.colorbar()
    cbar.set_label('State')

    # Show the plot
    plt.show()