import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

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

# update this with PCA so can be used w/ more than 2 dimensions
def plot_hmm_data(data, ss, k, means, covars):
    # Scatter plot
    plt.scatter(data[:, 0], data[:, 1], c=ss, cmap='viridis', marker='o')

    for state in range(k):
        state_data = data[ss == state]
        state_mean = means[state][:2]
        state_cov = covars[state][:2, :2]
        plot_ellipse(plt.gca(), state_mean, state_cov, color=f'C{state}')

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of HMM-generated Data')

    # Show colorbar for states
    cbar = plt.colorbar()
    cbar.set_label('State')

    # Show the plot
    plt.show()