import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import ipywidgets as widgets
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D

def plot_larva_data(larva_data, larva_id, style_path=None):
    """Plots speed, midline, vel_x, and vel_y over time for a single larva.
    
    Args:
        larva_data (dict): Dictionary containing time, speed, midline, vel_x, and vel_y.
        larva_id (str): Identifier of the larva (for title).
        style_path (str, optional): Path to the .mplstyle file.
    """
    if style_path:
        plt.style.use(style_path)

    # Extract data
    time = np.array(larva_data["time"])
    speed = np.array(larva_data["speed"])
    midline = np.array(larva_data["midline"])
    vel_x = np.array(larva_data["vel_x"])
    vel_y = np.array(larva_data["vel_y"])

    # Create the subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig.tight_layout(pad=3.0)

    # Plot speed
    axs[0, 0].plot(time, speed, label="Speed", color="blue", linewidth=2)
    axs[0, 0].set_ylabel("Speed")

    # Plot midline
    axs[0, 1].plot(time, midline, label="Midline", color="green", linewidth=2)
    axs[0, 1].set_ylabel("Midline")

    # Plot vel_x
    axs[1, 0].plot(time, vel_x, label="Velocity X", color="red", linewidth=2)
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Velocity X")

    # Plot vel_y
    axs[1, 1].plot(time, vel_y, label="Velocity Y", color="purple", linewidth=2)
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Velocity Y")

    # Set the title for the entire figure
    fig.suptitle(f"Larva {larva_id}")

    # Show the plot
    plt.show()

def plot_trajectory_2d_interactive(larva_data, larva_id):
    """Plots the 2D trajectory of a larva with an interactive slider to control time.
    
    Args:
        larva_data (dict): Dictionary containing loc_x, loc_y, and time.
        larva_id (str): Identifier of the larva (for title).
    """
    # Extract data
    loc_x = np.array(larva_data["loc_x"])
    loc_y = np.array(larva_data["loc_y"])
    time = np.array(larva_data["time"])

    # Determine the limits based on the final time point
    x_min, x_max = loc_x.min(), loc_x.max()
    y_min, y_max = loc_y.min(), loc_y.max()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(loc_x, loc_y, c=time, cmap='viridis', s=5)  # Smaller dots with colormap
    ax.set_xlim(x_min-1, x_max+1)
    ax.set_ylim(y_min-1, y_max+1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Trajectory of Larva {larva_id}')
    plt.colorbar(scatter, ax=ax, label='Time')
    plt.grid(False)  # Ensure grid is not shown

    # Create the slider
    time_slider = widgets.FloatSlider(
        value=time[0], 
        min=time[0], 
        max=time[-1], 
        step=(time[-1] - time[0]) / len(time), 
        description='Time', 
        continuous_update=True,
        layout=widgets.Layout(width='800px')  # Make the slider bigger
    )

    def update_plot(change):
        current_time = time_slider.value
        current_time_index = np.searchsorted(time, current_time)
        ax.clear()
        scatter = ax.scatter(loc_x[:current_time_index+1], loc_y[:current_time_index+1], c=time[:current_time_index+1], cmap='viridis', s=5)  # Smaller dots with colormap
        ax.set_xlim(x_min-1, x_max+1)
        ax.set_ylim(y_min-1, y_max+1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Trajectory of Larva {larva_id}')
        plt.grid(False)  # Ensure grid is not shown
        fig.canvas.draw_idle()

    time_slider.observe(update_plot, names='value')

    display(time_slider)
    plt.show()
    
def plot_clusters(data, larva_id, n_components=3):
    """Plots the clusters on the PCA-reduced data.
    
    Args:
        data (pd.DataFrame): DataFrame with the original data, PCA components, and cluster labels.
        larva_id (str): Identifier of the larva (for title).
        n_components (int, optional): Number of PCA components to plot. Defaults to 3.
    """
    if n_components == 2:
        plt.figure(figsize=(8, 5))
        unique_clusters = data['cluster'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            cluster_data = data[data['cluster'] == cluster]
            plt.scatter(cluster_data['pca_1'], cluster_data['pca_2'], label=f'Cluster {cluster}', color=color, s=30)
        
        plt.title(f'Clusters of Behaviours for Larva {larva_id}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Cluster')
        plt.grid(False)  # Ensure grid is not shown
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')
        unique_clusters = data['cluster'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            cluster_data = data[data['cluster'] == cluster]
            ax.scatter(cluster_data['pca_1'], cluster_data['pca_2'], cluster_data['pca_3'], label=f'Cluster {cluster}', color=color, s=30)
        
        ax.set_title(f'Clusters of Behaviours for Larva {larva_id}')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.legend(title='Cluster')
        plt.show()


def plot_trajectory_with_clusters(larva_data, clustered_data, larva_id):
    """Plots the 2D trajectory of a larva with a color gradient representing cluster assignment.
    
    Args:
        larva_data (dict): Dictionary containing loc_x, loc_y, and time.
        clustered_data (pd.DataFrame): DataFrame with the original data, PCA components, and cluster labels.
        larva_id (str): Identifier of the larva (for title).
    """
    # Extract data
    loc_x = np.array(larva_data["loc_x"])
    loc_y = np.array(larva_data["loc_y"])
    time = np.array(larva_data["time"])
    clusters = np.array(clustered_data["cluster"])

    # Create the plot
    plt.figure(figsize=(10, 6))
    unique_clusters = np.unique(clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_indices = np.where(clusters == cluster)
        plt.scatter(loc_x[cluster_indices], loc_y[cluster_indices], label=f'Cluster {cluster}', color=color, s=10)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Trajectory of Larva {larva_id} with Cluster Colors')
    plt.legend(title='Cluster')
    plt.grid(False)  # Ensure grid is not shown
    plt.show()


def plot_navigational_index_histogram(ni_dict, bins=20, density=True, fit_distribution=True):
    """Function to plot the histogram of navigational index
    
    Args:
        ni_dict (dict): Dictionary containing the navigational index for each larva
        bins (int, optional): Number of bins for the histogram. Defaults to 20.
        density (bool, optional): Whether to normalize the histogram. Defaults to True.
        fit_distribution (bool, optional): Whether to fit and plot a distribution over the histogram. Defaults to True.
    """
    # Compute the mean navigational index for each larva
    ni_means = [np.nanmean(ni_df["NI"]) for ni_df in ni_dict.values()]
    
    # Calculate mean and standard deviation
    mean_ni = np.mean(ni_means)
    std_ni = np.std(ni_means)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ni_means, bins=bins, density=density, color='blue', edgecolor='black', alpha=0.7, label='Histogram')
    
    if fit_distribution:
        # Fit a normal distribution to the data
        mu, std = stats.norm.fit(ni_means)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label=fr'Fit: $\mu={mu:.2f}$, $\sigma={std:.2f}$')
    
    plt.title(f'Navigational Index Distribution (Total Larvae: {len(ni_means)})')
    plt.xlabel('Mean Navigational Index')
    plt.ylabel('Density' if density else 'Frequency')
    plt.legend()
    plt.grid(False)  # Ensure grid is not shown
    plt.show()
    
    print(f'Mean Navigational Index: {mean_ni:.2f}')
    print(f'Standard Deviation of Navigational Index: {std_ni:.2f}')


def plot_navigational_index_time_series_single(ni_df, larva_id, window_size=5):
    plt.figure(figsize=(10, 6))
    
    # Apply rolling window average
    ni_df["NI_smooth"] = ni_df["NI"].rolling(window=window_size, min_periods=1).mean()
    ni_df["NI_norm_smooth"] = ni_df["NI_norm"].rolling(window=window_size, min_periods=1).mean()
    
    plt.plot(ni_df["time"], ni_df["NI_smooth"], label="Smoothed Navigational Index", color="blue")
    plt.plot(ni_df["time"], ni_df["NI_norm_smooth"], label="Smoothed Normalized Navigational Index", color="red")
    plt.xlabel("Time")
    plt.ylabel("Navigational Index")
    plt.title(f"Navigational Index Over Time for Larva {larva_id}")
    plt.legend()
    plt.show()

def plot_normalized_navigational_index_time_series(ni_dict, window_size=5):
    plt.figure(figsize=(12, 8))
    
    all_normalized_times = []
    all_smoothed_ni = []
    
    for larva_id, ni_df in ni_dict.items():
        # Normalize the time axis
        normalized_time = (ni_df["time"] - ni_df["time"].min()) / (ni_df["time"].max() - ni_df["time"].min())
        
        # Apply rolling window average
        ni_df["NI_smooth"] = ni_df["NI"].rolling(window=window_size, min_periods=1).mean()
        
        # Store the normalized time and smoothed NI
        all_normalized_times.append(normalized_time)
        all_smoothed_ni.append(ni_df["NI_smooth"])
        
        # Plot each larva's smoothed NI with low alpha
        plt.plot(normalized_time, ni_df["NI_smooth"], color='blue', alpha=0.1)
    
    # Interpolate all smoothed NI to a common time axis
    common_time = np.linspace(0, 1, 1000)
    interpolated_ni = np.array([np.interp(common_time, t, ni) for t, ni in zip(all_normalized_times, all_smoothed_ni)])
    
    # Compute the mean and standard error of the mean
    mean_ni = np.nanmean(interpolated_ni, axis=0)
    sem_ni = np.nanstd(interpolated_ni, axis=0) / np.sqrt(interpolated_ni.shape[0])
    
    # Plot the mean NI with shaded area for SEM
    plt.plot(common_time, mean_ni, color='red', label='Mean NI')
    plt.fill_between(common_time, mean_ni - sem_ni, mean_ni + sem_ni, color='red', alpha=0.3, label='SEM')
    
    plt.xlabel("Normalized Time")
    plt.ylabel("Navigational Index")
    plt.title("Normalized Navigational Index Over Time for All Larvae")
    plt.grid(False)  # Ensure grid is not shown
    plt.show()

def plot_navigational_index_time_series_together(ni_dict, window_size=5):
    plt.figure(figsize=(12, 8))
    
    all_times = []
    all_smoothed_ni = []
    
    for larva_id, ni_df in ni_dict.items():
        # Subtract the least time value so they all start at zero
        ni_df["time"] = ni_df["time"] - ni_df["time"].min()
        
        # Apply rolling window average
        ni_df["NI_smooth"] = ni_df["NI"].rolling(window=window_size, min_periods=1).mean()
        
        # Store the time and smoothed NI
        all_times.append(ni_df["time"].values)
        all_smoothed_ni.append(ni_df["NI_smooth"].values)
        
        # Plot each larva's smoothed NI with low alpha
        plt.plot(ni_df["time"], ni_df["NI_smooth"], color='blue', alpha=0.1)
    
    # Determine the common time axis
    common_time = np.linspace(min([t.min() for t in all_times]), max([t.max() for t in all_times]), num=1000)
    
    # Interpolate all smoothed NI to the common time axis
    interpolated_ni = np.array([np.interp(common_time, t, ni, left=np.nan, right=np.nan) for t, ni in zip(all_times, all_smoothed_ni)])
    
    # Compute the mean and standard error of the mean
    mean_ni = np.nanmean(interpolated_ni, axis=0)
    sem_ni = np.nanstd(interpolated_ni, axis=0) / np.sqrt(interpolated_ni.shape[0])
    
    # Plot the mean NI with shaded area for SEM
    plt.plot(common_time, mean_ni, color='red', label='Mean NI')
    plt.fill_between(common_time, mean_ni - sem_ni, mean_ni + sem_ni, color='red', alpha=0.3, label='SEM')
    
    plt.xlabel("Time")
    plt.ylabel("Navigational Index")
    plt.title("Navigational Index Over Time for All Larvae")
    plt.show()
