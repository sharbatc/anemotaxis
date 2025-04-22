import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import ipywidgets as widgets
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D

import os
import pandas as pd
from matplotlib import cm 
from matplotlib.collections import LineCollection
from matplotlib import animation
plt.style.use('../anemotaxis.mplstyle')
from IPython.display import display
import ipywidgets as widgets


def plot_larva_data(data, columns=None, larva_id=None, style_path=None):
    """
    Plots available data components over time for a single larva.
    
    Args:
        data (dict): Dictionary containing data columns
        columns (list, optional): List of column names to plot. If None, plots all except 'time' and 'id'
        larva_id (str, optional): Identifier of the larva (for title)
        style_path (str, optional): Path to the .mplstyle file
    """
    if style_path:
        plt.style.use(style_path)

    # Define default colors for different data types
    default_colors = {
        'persistence': 'purple',
        'speed': 'blue',
        'midline': 'green',
        'loc_x': 'orange',
        'loc_y': 'brown',
        'vel_x': 'red',
        'vel_y': 'pink',
        'orient': 'cyan',
        'pathlen': 'gray'
    }

    # Extract time data
    time = np.array(data["time"])
    
    # Determine which columns to plot
    if columns is None:
        # Plot all columns except time and id
        plot_columns = [col for col in data.keys() 
                       if col not in ['time', 'id']]
    else:
        # Plot specified columns
        plot_columns = [col for col in columns 
                       if col in data.keys() and col not in ['time', 'id']]
    
    n_plots = len(plot_columns)
    if n_plots == 0:
        raise ValueError("No valid columns to plot")
    
    # Create subplot layout
    n_rows = (n_plots + 1) // 2  # Round up division
    n_cols = min(n_plots, 2)
    fig = plt.figure(figsize=(7*n_cols, 5*n_rows))
    
    # Create grid for subplots
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    axes = []
    
    # Create subplots
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        # Plot data
        column = plot_columns[i]
        color = default_colors.get(column, 'black')
        ax.plot(time, np.array(data[column]), 
                color=color, linewidth=1.5, label=column)
        
        # Customize subplot
        ax.set_ylabel(column.replace('_', ' ').title())
        if row == n_rows - 1:  # If in last row
            ax.set_xlabel("Time (s)")
        ax.legend(loc='upper right')
    
    # Set title
    if larva_id is not None:
        fig.suptitle(f"Larva {larva_id} Behavioral Components", 
                    fontsize=20, fontweight='bold')
    
    return fig, axes


def plot_single_trajectory(data, larva_id):
    """Plot larva trajectory with interactive time slider."""
    plt.ioff()
    
    # Get data
    larva = data[larva_id]['data']
    time = np.array(larva['time'])
    x = np.array(larva['loc_x'])
    y = np.array(larva['loc_y'])
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(12,6))
    
    # Create main axis for trajectory
    main_ax = plt.axes([0.1, 0.15, 0.8, 0.75])
    cbar_height = 0.6  # Reduce this value to make colorbar shorter
    cbar_y = 0.15 + (0.75 - cbar_height)/2  # Center the colorbar vertically
    cbar_ax = plt.axes([0.92, cbar_y, 0.02, cbar_height])
    
    # Plot trajectory with color gradient
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap='magma', norm=norm)
    lc.set_array(time)
    main_ax.add_collection(lc)
    
    plt.colorbar(lc, cax=cbar_ax, label='Time (s)')
    
    # Create current position marker
    point, = main_ax.plot([], [], 'o', color='white', markersize=8,
                         markeredgecolor='black', markeredgewidth=2,
                         label='Current Position')
    
    # Set axis limits with padding
    x_padding = (np.max(x) - np.min(x)) * 0.1
    y_padding = (np.max(y) - np.min(y)) * 0.1
    main_ax.set_xlim(np.min(x) - x_padding, np.max(x) + x_padding)
    main_ax.set_ylim(np.min(y) - y_padding, np.max(y) + y_padding)
    
    # Make slider width match the plot width
    slider_width = f"{int(fig.get_figwidth() * 100)}px"
    
    time_slider = widgets.FloatSlider(
        value=time[0],
        min=time[0],
        max=time[-1],
        step=(time[-1] - time[0]) / len(time),
        description='Time (s):',
        style={'description_width': 'initial'},
        readout_format='.1f',
        layout=widgets.Layout(width=slider_width)
    )
    
    def update_plot(change):
        if change['name'] == 'value':
            current_time = change['new']
            time_index = np.searchsorted(time, current_time)
            point.set_data([x[time_index]], [y[time_index]])
            main_ax.set_title(f'Trajectory of Larva {larva_id} (Time: {current_time:.1f}s)')
            fig.canvas.draw_idle()
    
    # Set axes properties
    main_ax.set_aspect('equal')
    main_ax.set_xlabel('X Position')
    main_ax.set_ylabel('Y Position')
    main_ax.grid(False)
    main_ax.legend()
    
    # Connect events
    time_slider.observe(update_plot)
    
    # Display slider and figure
    display(time_slider)
    display(fig.canvas)
    
    # Initialize plot
    update_plot({'name': 'value', 'new': time[0]})


def plot_multiple_trajectories(data, larva_ids=None, output_path='trajectory_animation.mp4'):
    """Plot multiple larval trajectories with interactive time slider and save as video."""
    plt.ioff()
    
    if larva_ids is None:
        larva_ids = list(data.keys())
    
    # Create figure with square-like aspect ratio
    fig = plt.figure(figsize=(6,6))  # Make height equal to width
    main_ax = plt.axes([0, 0.15, 0.9, 0.75])
    cbar_height = 0.6
    cbar_y = 0.15 + (0.75 - cbar_height)/2
    cbar_ax = plt.axes([0.80, cbar_y, 0.02, cbar_height])
    
    # Find global time range and store larval data
    max_time = 0
    trajectories = {}
    
    for larva_id in larva_ids:
        larva = data[larva_id]['data']
        time = np.array(larva['time'])
        x = np.array(larva['loc_x']) - np.min(larva['loc_x'])
        y = np.array(larva['loc_y'])
        
        max_time = max(max_time, np.max(time))
        trajectories[larva_id] = {'time': time, 'x': x, 'y': y}
    
    # After collecting all trajectory data, find y-range
    y_min = float('inf')
    y_max = float('-inf')
    for traj in trajectories.values():
        y_min = min(y_min, np.min(traj['y']))
        y_max = max(y_max, np.max(traj['y']))
    
    # Set axis limits
    main_ax.set_xlim(-30, 80)
    main_ax.set_ylim(0, 200)
 
    # After plotting trajectories but before the colorbar, add vertical line
    main_ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5)

    # Add wind direction arrows
    arrow_y_positions = np.linspace(20, 180, 10)  # 10 arrows along y-axis
    arrow_length = 5  # Length of arrow in data units
    for y_pos in arrow_y_positions:
        main_ax.arrow(70, y_pos, -arrow_length, 0,
                    head_width=3,      
                    head_length=4,    
                    width=0.5,         # Add width to make arrow shaft thicker
                    fc='blue',         # Fill color
                    ec='blue',         # Edge color
                    alpha=0.5,
                    overhang=0.3) 
    
    # Add text for wind direction
    main_ax.text(70, 80, 'Wind direction', color='blue', 
                rotation=-90,  # Rotate text 90 degrees counterclockwise
                ha='center',   # Horizontal alignment
                va='bottom',   # Vertical alignment
                alpha=0.7,
                fontsize=15)   # Adjust font size if needed
    
    # Define marker colors for different larvae
    marker_colors = plt.cm.Set2(np.linspace(0, 1, len(larva_ids)))
    
    # Plot full trajectories with time-based color gradient
    norm = plt.Normalize(0, max_time)
    markers = {}
    
    for i, larva_id in enumerate(larva_ids):
        traj = trajectories[larva_id]
        points = np.array([traj['x'], traj['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Use 'magma' colormap for trajectories with time gradient
        lc = LineCollection(segments, cmap='magma', norm=norm)
        lc.set_array(traj['time'])
        main_ax.add_collection(lc)
        
        # Create current position markers with distinct colors
        marker, = main_ax.plot([], [], 'o', color=marker_colors[i], markersize=8,
                             markeredgecolor='black', markeredgewidth=2)
        markers[larva_id] = marker
        
        # Add small text label with matching marker color
        main_ax.text(traj['x'][0]+2, traj['y'][0]+2, f' {larva_id}', 
                    color=marker_colors[i], fontsize=10)
    
    plt.colorbar(lc, cax=cbar_ax, label='Time (s)')
    
    # Set axes properties
    main_ax.set_aspect('equal')
    main_ax.set_xlabel('X Position')
    main_ax.set_ylabel('Y Position')
    main_ax.grid(False)
    
    # Animation function
    def update_frame(time):
        for larva_id, marker in markers.items():
            traj = trajectories[larva_id]
            time_index = np.searchsorted(traj['time'], time)
            if time_index < len(traj['time']):
                marker.set_data([traj['x'][time_index]], [traj['y'][time_index]])
            else:
                marker.set_data([], [])
        
        main_ax.set_title(f'Larval Trajectories (Time: {time:.1f}s)')
        return tuple(markers.values()) + (main_ax,)
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=np.linspace(0, max_time, 500), blit=True)
    
    # Save animation as video
    ani.save(output_path, writer='ffmpeg', fps=30)
    
    plt.close(fig)  # Close the figure to prevent display in notebook
    print(f"Animation saved to {output_path}")

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
            ax.scatter(cluster_data['pca_1'], cluster_data['pca_2'], cluster_data['pca_3'], label=f'Cluster {cluster}', color=color, s=5)
        
        ax.set_title(f'Clusters of Behaviours for Larva {larva_id}')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.legend(title='Cluster')
        plt.show()


def plot_trajectory_with_clusters(larva_data, clustered_data, larva_id):
    """Plots the 2D trajectory of a larva with cluster colors and interactive time slider."""
    plt.ioff()
    
    # Extract data
    loc_x = np.array(larva_data["loc_x"])
    loc_y = np.array(larva_data["loc_y"])
    time = np.array(larva_data["time"])
    clusters = np.array(clustered_data["cluster"])

    # Create figure with specific layout
    fig = plt.figure(figsize=(12, 6))
    main_ax = plt.axes([0.1, 0.15, 0.8, 0.75])
    
    # Plot full trajectory with cluster colors
    unique_clusters = np.unique(clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = dict(zip(unique_clusters, colors))
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_indices = np.where(clusters == cluster)
        main_ax.scatter(loc_x[cluster_indices], loc_y[cluster_indices], 
                       color=color, s=10, alpha=0.5, 
                       label=f'Cluster {cluster}')
    
    # Create current position marker
    point, = main_ax.plot([], [], 'o', color='white', markersize=10,
                         markeredgecolor='black', markeredgewidth=2,
                         label='Current Position')
    
    # Set axis limits with padding
    x_padding = (np.max(loc_x) - np.min(loc_x)) * 0.1
    y_padding = (np.max(loc_y) - np.min(loc_y)) * 0.1
    main_ax.set_xlim(np.min(loc_x) - x_padding, np.max(loc_x) + x_padding)
    main_ax.set_ylim(np.min(loc_y) - y_padding, np.max(loc_y) + y_padding)
    
    # Create time slider
    slider_width = f"{int(fig.get_figwidth() * 100)}px"
    time_slider = widgets.FloatSlider(
        value=time[0],
        min=time[0],
        max=time[-1],
        step=(time[-1] - time[0]) / len(time),
        description='Time (s):',
        style={'description_width': 'initial'},
        readout_format='.1f',
        layout=widgets.Layout(width=slider_width)
    )
    
    def update_plot(change):
        if change['name'] == 'value':
            current_time = change['new']
            time_index = np.searchsorted(time, current_time)
            
            # Update marker position and color based on current cluster
            point.set_data([loc_x[time_index]], [loc_y[time_index]])
            current_cluster = clusters[time_index]
            point.set_color(cluster_colors[current_cluster])
            
            main_ax.set_title(f'Trajectory of Larva {larva_id}\n'
                             f'Time: {current_time:.1f}s, Cluster: {current_cluster}')
            fig.canvas.draw_idle()
    
    # Set axes properties
    main_ax.set_aspect('equal')
    main_ax.set_xlabel('X Position')
    main_ax.set_ylabel('Y Position')
    main_ax.grid(False)
    main_ax.legend(loc='lower right')
    
    # Connect events
    time_slider.observe(update_plot)
    
    # Display slider and figure
    display(time_slider)
    display(fig.canvas)
    
    # Initialize plot
    update_plot({'name': 'value', 'new': time[0]})


def plot_navigational_indices_comparison(ni_dict_x, ni_dict_y, bins=20, density=True, fit_distribution=True):
    """Plot x and y navigational indices histograms side by side for comparison."""
    # Compute mean navigational indices, filtering out NaN values
    ni_means_x = np.array([x for x in [np.nanmean(df["NI"]) for df in ni_dict_x.values()] 
                          if np.isfinite(x)])
    ni_means_y = np.array([y for y in [np.nanmean(df["NI"]) for df in ni_dict_y.values()]
                          if np.isfinite(y)])
    
    if len(ni_means_x) == 0 or len(ni_means_y) == 0:
        print("No valid navigational index values found.")
        return
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot X-axis navigational index
    ax1.hist(ni_means_x, bins=bins, density=density, color='blue', 
             edgecolor='black', alpha=0.7, label='X-axis $N_{\mathrm{ind}}$')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Plot Y-axis navigational index
    ax2.hist(ni_means_y, bins=bins, density=density, color='green', 
             edgecolor='black', alpha=0.7, label='Y-axis $N_{\mathrm{ind}}$')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    if fit_distribution:
        try:
            # Fit distributions for X-axis NI
            mu_x, std_x = stats.norm.fit(ni_means_x)
            xmin, xmax = ax1.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            px = stats.norm.pdf(x, mu_x, std_x)
            ax1.plot(x, px, 'k', linewidth=2, 
                    label=fr'Fit: $\mu={mu_x:.2f}$, $\sigma={std_x:.2f}$')
            
            # Fit distributions for Y-axis NI
            mu_y, std_y = stats.norm.fit(ni_means_y)
            ymin, ymax = ax2.get_xlim()
            y = np.linspace(ymin, ymax, 100)
            py = stats.norm.pdf(y, mu_y, std_y)
            ax2.plot(y, py, 'k', linewidth=2, 
                    label=fr'Fit: $\mu={mu_y:.2f}$, $\sigma={std_y:.2f}$')
        except Exception as e:
            print(f"Could not fit distribution: {str(e)}")
    
    # Set titles and labels
    ax1.set_title(f'X-axis Nind Distribution (n={len(ni_means_x)})')
    ax2.set_title(f'Y-axis Nind Distribution (n={len(ni_means_y)})')
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Mean Navigational Index ($N_{\mathrm{ind}}$)')
        ax.set_ylabel('Density' if density else 'Frequency')
        ax.legend()
        ax.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Summary Statistics:")
    print(f'X-axis - Mean: {np.mean(ni_means_x):.2f}, Std: {np.std(ni_means_x):.2f}')
    print(f'Y-axis - Mean: {np.mean(ni_means_y):.2f}, Std: {np.std(ni_means_y):.2f}')

def plot_navigational_index_time_series_single(ni_df, larva_id, window_size=5):
    plt.figure(figsize=(8, 6))
    
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

def prepare_ni_data(ni_dict_x, ni_dict_y, window_size=500):
    """Prepare and smooth navigational index data for both x and y axes.
    
    Args:
        ni_dict_x (dict): Dictionary containing x-axis navigational indices
        ni_dict_y (dict): Dictionary containing y-axis navigational indices
        window_size (int): Window size for rolling mean smoothing
    
    Returns:
        tuple: (sorted larva IDs, smoothed x data, smoothed y data, axis limits)
    """
    larva_ids = sorted(list(ni_dict_x.keys()))
    smoothed_data_x = {}
    smoothed_data_y = {}
    y_min_x, y_max_x = float('inf'), float('-inf')
    y_min_y, y_max_y = float('inf'), float('-inf')
    
    for larva_id in larva_ids:
        # Process X data
        ni_df_x = ni_dict_x[larva_id]
        smoothed_data_x[larva_id] = {
            'NI_smooth': ni_df_x["NI"].rolling(window=window_size, min_periods=1).mean(),
            'NI_norm_smooth': ni_df_x["NI_norm"].rolling(window=window_size, min_periods=1).mean(),
            'time': ni_df_x["time"]
        }
        y_min_x = min(y_min_x, smoothed_data_x[larva_id]['NI_smooth'].min(), 
                      smoothed_data_x[larva_id]['NI_norm_smooth'].min())
        y_max_x = max(y_max_x, smoothed_data_x[larva_id]['NI_smooth'].max(), 
                      smoothed_data_x[larva_id]['NI_norm_smooth'].max())
        
        # Process Y data
        ni_df_y = ni_dict_y[larva_id]
        smoothed_data_y[larva_id] = {
            'NI_smooth': ni_df_y["NI"].rolling(window=window_size, min_periods=1).mean(),
            'NI_norm_smooth': ni_df_y["NI_norm"].rolling(window=window_size, min_periods=1).mean(),
            'time': ni_df_y["time"]
        }
        y_min_y = min(y_min_y, smoothed_data_y[larva_id]['NI_smooth'].min(), 
                      smoothed_data_y[larva_id]['NI_norm_smooth'].min())
        y_max_y = max(y_max_y, smoothed_data_y[larva_id]['NI_smooth'].max(), 
                      smoothed_data_y[larva_id]['NI_norm_smooth'].max())
    
    axis_limits = {
        'x': (y_min_x - 0.1, y_max_x + 0.1),
        'y': (y_min_y - 0.1, y_max_y + 0.1)
    }
    
    return larva_ids, smoothed_data_x, smoothed_data_y, axis_limits

def plot_ni_interactive(ni_dict_x, ni_dict_y, window_size=500):
    """Create interactive plot of navigational indices with play/pause controls.
    
    Args:
        ni_dict_x (dict): Dictionary containing x-axis navigational indices
        ni_dict_y (dict): Dictionary containing y-axis navigational indices
        window_size (int): Window size for rolling mean smoothing
    """
    plt.ioff()
    
    # Prepare data
    larva_ids, smoothed_data_x, smoothed_data_y, axis_limits = prepare_ni_data(
        ni_dict_x, ni_dict_y, window_size)
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Initialize empty lines
    line1_x, = ax1.plot([], [], label="Smoothed $N_{\mathrm{ind}}$", color="blue")
    line2_x, = ax1.plot([], [], label="Smoothed Normalized $N_{\mathrm{ind}}$", color="red")
    line1_y, = ax2.plot([], [], label="Smoothed $N_{\mathrm{ind}}$", color="blue")
    line2_y, = ax2.plot([], [], label="Smoothed Normalized $N_{\mathrm{ind}}$", color="red")
    
    # Set axes properties
    ax1.set_ylim(*axis_limits['x'])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("X-axis $N_{\mathrm{ind}}$")
    ax1.legend()
    
    ax2.set_ylim(*axis_limits['y'])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y-axis $N_{\mathrm{ind}}$")
    ax2.legend()
    
    plt.tight_layout(w_pad=3.0)
    
    # Create controls
    play = widgets.Play(
        value=0,
        min=0,
        max=len(larva_ids) - 1,
        step=1,
        interval=500,
        description="Play"
    )
    
    slider = widgets.IntSlider(
        min=0,
        max=len(larva_ids) - 1,
        description='Larva:',
        value=0,
        style={'description_width': 'initial'},
        readout_format='d',
        layout=widgets.Layout(width='1000px')
    )
    
    # Link controls
    widgets.jslink((play, 'value'), (slider, 'value'))
    
    def update_plot(change):
        if change['type'] == 'change' and change['name'] == 'value':
            index = change['new']
            larva_id = larva_ids[index]
            
            # Update X data
            data_x = smoothed_data_x[larva_id]
            line1_x.set_data(data_x['time'], data_x['NI_smooth'])
            line2_x.set_data(data_x['time'], data_x['NI_norm_smooth'])
            ax1.set_xlim(data_x['time'].min(), data_x['time'].max())
            ax1.set_title(f"X-axis $N_ind$ (Larva {larva_id})")
            
            # Update Y data
            data_y = smoothed_data_y[larva_id]
            line1_y.set_data(data_y['time'], data_y['NI_smooth'])
            line2_y.set_data(data_y['time'], data_y['NI_norm_smooth'])
            ax2.set_xlim(data_y['time'].min(), data_y['time'].max())
            ax2.set_title(f"Y-axis $N_ind$ (Larva {larva_id})")
            
            fig.canvas.draw_idle()
    
    # Register callback
    slider.observe(update_plot)
    
    # Display
    display(widgets.HBox([play, slider]))
    display(fig.canvas)
    
    # Initialize plot
    update_plot({'type': 'change', 'name': 'value', 'new': 0})

def save_ni_animation(ni_dict_x, ni_dict_y, output_path='navigational_indices.mp4', fps=2, window_size=500):
    """Save navigational indices animation as video.
    
    Args:
        ni_dict_x (dict): Dictionary containing x-axis navigational indices
        ni_dict_y (dict): Dictionary containing y-axis navigational indices
        output_path (str): Path where to save the video file
        fps (int): Frames per second for the video
        window_size (int): Window size for rolling mean smoothing
    """
    plt.ioff()
    
    # Prepare data
    larva_ids, smoothed_data_x, smoothed_data_y, axis_limits = prepare_ni_data(
        ni_dict_x, ni_dict_y, window_size)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup video writer
    writer = animation.FFMpegWriter(fps=fps)
    
    with writer.saving(fig, output_path, dpi=100):
        for idx, larva_id in enumerate(larva_ids):
            ax1.clear()
            ax2.clear()
            
            # Get and plot data
            data_x = smoothed_data_x[larva_id]
            data_y = smoothed_data_y[larva_id]
            
            ax1.plot(data_x['time'], data_x['NI_smooth'], 'b-', 
                    label="Smoothed $N_{\mathrm{ind}}$")
            ax1.plot(data_x['time'], data_x['NI_norm_smooth'], 'r-', 
                    label="Smoothed Normalized $N_{\mathrm{ind}}$")
            ax1.set_ylim(*axis_limits['x'])
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("X-axis $N_ind$")
            ax1.set_title(f"X-axis $N_ind$ (Larva {larva_id})")
            ax1.legend()
            
            ax2.plot(data_y['time'], data_y['NI_smooth'], 'b-', 
                    label="Smoothed $N_{\mathrm{ind}}$")
            ax2.plot(data_y['time'], data_y['NI_norm_smooth'], 'r-', 
                    label="Smoothed Normalized $N_{\mathrm{ind}}$")
            ax2.set_ylim(*axis_limits['y'])
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Y-axis $N_ind$")
            ax2.set_title(f"Y-axis $N_ind$ (Larva {larva_id})")
            ax2.legend()
            
            plt.tight_layout(w_pad=3.0)
            writer.grab_frame()
            
            print(f"Processing larva {idx+1}/{len(larva_ids)}", end='\r')
        
        print("\nVideo saved successfully!")