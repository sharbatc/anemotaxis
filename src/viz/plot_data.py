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

import core.data_loader as data_loader
import core.data_processor as data_processor
import utils.preprocessing as preprocessing

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

##### Global Behavior Matrix Plotting #####

def plot_global_behavior_matrix(trx_data, show_plot=True, ax=None):
    """
    Plot global behavior using the global state.
    
    This function visualizes behavioral states across time for all larvae.
    It uses global_state_small_large_state which contains both small and large behaviors.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    import os
    from datetime import datetime

    # Create axis if none provided
    if ax is None and show_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        created_fig = True
    else:
        created_fig = False
    
    # Define state value mapping
    base_colors = {
        'run': [0.0, 0.0, 0.0],      # Black
        'cast': [1.0, 0.0, 0.0],     # Red
        'stop': [0.0, 0.5, 0.0],     # Green
        'hunch': [0.0, 0.0, 1.0],    # Blue
        'backup': [0.0, 1.0, 1.0],   # Cyan
        'roll': [1.0, 1.0, 0.0],     # Yellow
    }
    
    state_info = {
        # Large behaviors (integer values)
        1.0: ('run', 1.0), 2.0: ('cast', 1.0), 3.0: ('stop', 1.0),
        4.0: ('hunch', 1.0), 5.0: ('backup', 1.0), 6.0: ('roll', 1.0),
        # Small behaviors (half-integer values)
        0.5: ('run', 0.4), 1.5: ('cast', 0.4), 2.5: ('stop', 0.4),
        3.5: ('hunch', 0.4), 4.5: ('backup', 0.4), 5.5: ('roll', 0.4),
    }

    # Process larvae data and extract date from metadata
    if isinstance(trx_data, dict) and 'data' in trx_data:
        larvae_data = trx_data['data']
        # Use date_str from metadata
        title_date = trx_data.get('metadata', {}).get('date_str', 'Unknown Date')
    else:
        larvae_data = trx_data
        title_date = 'Unknown Date'
        
    larva_ids = sorted(larvae_data.keys())
    n_larvae = len(larva_ids)
    
    if n_larvae == 0:
        ax.text(0.5, 0.5, "No larva data available", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Get global time range
    t_min, t_max = float('inf'), float('-inf')
    valid_larvae = []
    
    for lid in larva_ids:
        if ('global_state_small_large_state' in larvae_data[lid] and 
            't' in larvae_data[lid] and 
            len(larvae_data[lid]['t']) > 0):
            times = np.array(larvae_data[lid]['t']).flatten()
            t_min = min(t_min, np.min(times))
            t_max = max(t_max, np.max(times))
            valid_larvae.append(lid)
    
    if not valid_larvae:
        ax.text(0.5, 0.5, "No valid data available", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Plot each larva using vectorized operations
    for i, larva_id in enumerate(valid_larvae):
        times = np.array(larvae_data[larva_id]['t']).flatten()
        states = np.array(larvae_data[larva_id]['global_state_small_large_state']).flatten()
        
        # Ensure same length and sort if needed
        min_len = min(len(times), len(states))
        times, states = times[:min_len], states[:min_len]
        
        if not np.all(np.diff(times) >= 0):
            sorted_idx = np.argsort(times)
            times, states = times[sorted_idx], states[sorted_idx]
        
        # Get this larva's actual end time
        larva_t_max = times[-1] if len(times) > 0 else t_max
        
        # Create end times (next time point or this larva's max time)
        end_times = np.append(times[1:], larva_t_max)
        durations = end_times - times
        
        # Round states to nearest 0.5 once
        rounded_states = np.round(states * 2) / 2
        
        # Group consecutive identical states for efficiency
        state_changes = np.where(np.diff(rounded_states) != 0)[0] + 1
        segment_starts = np.concatenate(([0], state_changes))
        segment_ends = np.concatenate((state_changes, [len(times)]))
        
        # Plot segments with full height (no gaps between rows)
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            if start_idx >= len(times):
                continue
                
            state_val = rounded_states[start_idx]
            segment_start_time = times[start_idx]
            
            # Use the actual end time for this segment, not global t_max
            if end_idx - 1 < len(end_times):
                segment_end_time = end_times[end_idx - 1]
            else:
                segment_end_time = larva_t_max
                
            segment_duration = segment_end_time - segment_start_time
            
            # Get color and alpha
            if state_val in state_info:
                behavior, alpha = state_info[state_val]
                color = base_colors[behavior]
            else:
                color, alpha = [0.5, 0.5, 0.5], 0.3  # Gray for unknown
            
            # Use height=1.0 and align='edge' to eliminate gaps
            ax.barh(i, segment_duration, left=segment_start_time, height=1.0,
                   color=color, alpha=alpha, edgecolor='none', align='edge')
    
    # Set up y-axis with larva numbers and IDs - use exact integer positions
    ax.set_yticks([i + 0.5 for i in range(len(valid_larvae))])  # Center labels in bars
    y_labels = [f"#{i+1} (ID: {larva_id})" for i, larva_id in enumerate(valid_larvae)]
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Larva')
    ax.set_xlabel('Time (seconds)')
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, len(valid_larvae))  # No padding on y-axis
    
    # Set title with experimental date and larva count
    ax.set_title(f'Behavioral States - {title_date}\n({len(valid_larvae)} larvae)', 
                fontsize=14, pad=20)
    
    # Create legend at bottom
    legend_elements = []
    for behavior in ['run', 'cast', 'stop', 'hunch', 'backup', 'roll']:
        # Large behavior
        legend_elements.append(
            Patch(facecolor=base_colors[behavior], alpha=1.0, 
                  label=f"large {behavior}"))
        # Small behavior  
        legend_elements.append(
            Patch(facecolor=base_colors[behavior], alpha=0.4, 
                  label=f"small {behavior}"))
    
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False)
    
    if created_fig and show_plot:
        plt.tight_layout()
        plt.show()
    
    # Return processed data
    result_data = {}
    for larva_id in valid_larvae:
        result_data[larva_id] = {
            'times': np.array(larvae_data[larva_id]['t']).flatten(),
            'states': np.array(larvae_data[larva_id]['global_state_small_large_state']).flatten()
        }
    
    return result_data

##### Orientation Histogram Plotting #####

def plot_orientation_histogram(analysis_results, ax=None, show_plot=True, control=False, 
                              show_se=True, se_alpha=0.3, color=None, label=None,
                              xlabel=None, ylabel='Probability', show_xlabel=True, show_ylabel=True,
                              title=None, xlim=(-185, 185), show_legend=True, ylim=None, min_amplitude=None):

    """
    General function to plot any histogram with error bars across orientations.
    
    Args:
        analysis_results: Dictionary containing histogram data with keys:
                         - 'mean_hist': mean histogram values
                         - 'se_hist': standard error values
                         - 'bin_centers': bin center positions
                         - 'n_larvae': number of subjects/larvae
        ax: Matplotlib axis to plot on
        show_plot: Whether to display the plot
        control: If True, use dashed line style
        show_se: Whether to show standard error shading
        se_alpha: Alpha transparency for standard error shading
        color: Color for the plot (default: 'black' if control, 'blue' otherwise)
        label: Custom label for the plot
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        xlim: X-axis limits as tuple (min, max)
    
    Returns:
        The matplotlib axis used for plotting
    """
    if ax is None and show_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        created_fig = False

    # Handle empty results
    if analysis_results.get('n_larvae', 0) == 0 or len(analysis_results.get('mean_hist', [])) == 0:
        ax.text(0.5, 0.5, f"No {ylabel.lower()} data available", 
                ha='center', va='center', transform=ax.transAxes)
        return ax

    # Set plot style
    linestyle = '--' if control else '-'
    if color is None:
        color = 'black'

    # Extract results and convert to numpy arrays
    mean_hist = np.array(analysis_results['mean_hist'])
    se_hist = np.array(analysis_results.get('se_hist', np.zeros_like(mean_hist)))
    bin_centers = np.array(analysis_results['bin_centers'])
    
    # Create mask for valid (non-NaN) data
    valid_mask = ~np.isnan(mean_hist)
    
    if not np.any(valid_mask):
        ax.text(0.5, 0.5, f"No valid {ylabel.lower()} data available", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Filter to only valid data for plotting
    valid_bins = bin_centers[valid_mask]
    valid_mean = mean_hist[valid_mask]
    valid_se = se_hist[valid_mask]
    
    # Plot mean line (only valid points)
    ax.plot(valid_bins, valid_mean, color=color, linewidth=2, 
           linestyle=linestyle, label=label)
    
    # Plot standard error as shaded region (only valid points)
    if show_se and len(valid_se) > 0:
        ax.fill_between(valid_bins, 
                       valid_mean - valid_se, 
                       valid_mean + valid_se,
                       color=color, alpha=se_alpha)
    
    # Remove all spines except bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Detach axes from each other
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))

    # Set axis limits and ticks
    ax.set_xlim(xlim)
    ax.set_xticks([-180, -90, 0, 90, 180])
    
    # Set y-axis limits - handle amplitude plots differently
    if ylim:
        ax.set_ylim(ylim)
        y_max = ylim[1]
    elif 'Turn Amplitude' in ylabel:
        # For amplitude plots, set appropriate range
        data_max = np.max(mean_hist + se_hist) if len(se_hist) > 0 else np.max(mean_hist)
        
        # Use provided min_amplitude or default to 60
        y_min = min_amplitude if min_amplitude is not None else 60
        y_max = max(180, np.ceil(data_max / 10) * 10)  # Round up to nearest 10°, min 180°
        ax.set_ylim(y_min, y_max)
        y_max = y_max
    else:
        # For probability plots, start from 0
        y_min, y_max = ax.get_ylim()
        y_max_rounded = np.ceil(y_max * 50) / 50  # Round up to nearest 0.02
        ax.set_ylim(0, y_max_rounded)
        y_max = y_max_rounded

    # Set y-ticks with appropriate intervals
    if 'Turn Amplitude' in ylabel:
        # For amplitude plots, use 20° or 30° intervals
        y_min = min_amplitude if min_amplitude is not None else 60
        if y_max <= 120:
            tick_interval = 20
        else:
            tick_interval = 30
        y_ticks = np.arange(y_min, y_max + tick_interval, tick_interval)
    elif 'Head Cast Number' in ylabel:
        # For head cast number, use 1 as interval
        tick_interval = 1
        n_ticks = int(y_max / tick_interval)
        y_ticks = np.arange(0, (n_ticks + 1) * tick_interval, tick_interval)

    else:
        # For probability plots
        if y_max <= 0.15:
            tick_interval = 0.02
        elif y_max <= 0.30:
            tick_interval = 0.05
        else:
            tick_interval = 0.10
        n_ticks = int(y_max / tick_interval)
        y_ticks = np.arange(0, (n_ticks + 1) * tick_interval, tick_interval)
    
    ax.set_yticks(y_ticks)
    
    # Add reference lines at key orientations (behind the plot)
    for x_pos in [-180, -90, 0, 90, 180]:
        if xlim[0] <= x_pos <= xlim[1]:
            ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5, zorder=0)
    
    # Labels and title
    if show_xlabel:
        ax.set_xlabel(xlabel if xlabel else 'Body Orientations (°)')
    if show_ylabel:
        ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)

    # Only show legend if explicitly requested
    if show_legend and label:
        ax.legend(fontsize=8)

    if created_fig and show_plot:
        plt.tight_layout()
        plt.show()

    return ax


def plot_orientation_histogram_polar(analysis_results, ax=None, show_plot=True, control=False, 
                                   show_se=True, se_alpha=0.3, color=None, label=None,
                                   title=None, bar_style=True, tick_fontsize=8, n_radial_ticks=3,
                                   is_amplitude=False, min_amplitude=None):
    """
    Plot orientation histogram in polar coordinates as bars with optional standard error lines.
    """
    if ax is None and show_plot:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        created_fig = True
    else:
        created_fig = False
    n_subjects = analysis_results.get('n_larvae', 
                                    analysis_results.get('n_subjects', 0))
    # Handle empty results
    if n_subjects == 0 or len(analysis_results.get('mean_hist', [])) == 0:
        ax.text(0, 0, "No orientation data available", 
                ha='center', va='center', transform=ax.transAxes)
        return ax

    # Extract results and convert to numpy arrays
    mean_hist = np.array(analysis_results['mean_hist'])
    se_hist = np.array(analysis_results.get('se_hist', np.zeros_like(mean_hist)))
    bin_centers = np.array(analysis_results['bin_centers'])
    
    # Create mask for valid (non-NaN) data
    valid_mask = ~np.isnan(mean_hist)
    
    if not np.any(valid_mask):
        ax.text(0, 0, "No valid orientation data available", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Filter to only valid data
    valid_bins = bin_centers[valid_mask]
    valid_mean = mean_hist[valid_mask]
    valid_se = se_hist[valid_mask]
    
    # Convert to radians
    theta = np.deg2rad(valid_bins)
    
    # Calculate bin width for bar plotting
    bin_width_deg = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 20
    bin_width_rad = np.deg2rad(bin_width_deg)

    # Set plot style
    if color is None:
        color = 'black'

    if bar_style:
        # Plot as polar bars - USE FILTERED DATA
        bars = ax.bar(theta, valid_mean, width=bin_width_rad, 
                     color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Plot standard error as lines on top of bars
        if show_se and n_subjects > 1 and len(valid_se) > 0:
            upper_bound = valid_mean + valid_se
            lower_bound = valid_mean - valid_se
            
            for i, (th, mean_val, se_val) in enumerate(zip(theta, valid_mean, valid_se)):
                ax.plot([th, th], [lower_bound[i], upper_bound[i]], 
                       color=color, linewidth=1.5, alpha=0.8)
                cap_width = bin_width_rad * 0.3
                ax.plot([th - cap_width/2, th + cap_width/2], 
                       [upper_bound[i], upper_bound[i]], 
                       color=color, linewidth=1.5, alpha=0.8)
                ax.plot([th - cap_width/2, th + cap_width/2], 
                       [lower_bound[i], lower_bound[i]], 
                       color=color, linewidth=1.5, alpha=0.8)
    
    # Polar plot formatting
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
    # Set radial limits based on data type - USE NANMAX TO HANDLE NaN VALUES
    if is_amplitude:
        # For amplitude data, start from min_amplitude and adjust maximum
        data_max = np.nanmax(valid_mean + valid_se) if len(valid_se) > 0 else np.nanmax(valid_mean)
        
        # Handle case where all values are NaN
        if np.isnan(data_max):
            data_max = min_amplitude if min_amplitude is not None else 180
            
        y_min = min_amplitude if min_amplitude is not None else 60
        r_max = max(180, np.ceil(data_max / 10) * 10)  # Round up to nearest 10°, min 180°
        ax.set_ylim(y_min, r_max)
    else:
        # For probability data, start from 0
        r_max = np.nanmax(valid_mean + valid_se) if len(valid_se) > 0 else np.nanmax(valid_mean)
        
        # Handle case where all values are NaN
        if np.isnan(r_max) or r_max <= 0:
            r_max = 1.0  # Default safe value
            
        r_max_rounded = np.ceil(r_max * 100) / 100  # Round up to nearest 0.01
        ax.set_ylim(0, r_max_rounded)
        r_max = r_max_rounded
    
    # Customize radial ticks
    if n_radial_ticks > 0:
        if is_amplitude:
            # For amplitudes, use 20° or 30° intervals
            y_min = min_amplitude if min_amplitude is not None else 60
            if r_max <= 120:
                tick_interval = 20
            else:
                tick_interval = 30
            # Start from the first tick at or above y_min
            first_tick = np.ceil(y_min / tick_interval) * tick_interval
            n_ticks = min(n_radial_ticks, int((r_max - first_tick) / tick_interval) + 1)
            radial_ticks = np.arange(first_tick, first_tick + n_ticks * tick_interval, tick_interval)
        else:
            # For probabilities
            if r_max <= 0.10:
                tick_interval = 0.02
            elif r_max <= 0.30:
                tick_interval = 0.05
            else:
                tick_interval = 0.10
                
            n_ticks = min(n_radial_ticks, int(r_max / tick_interval))
            radial_ticks = np.arange(tick_interval, (n_ticks + 1) * tick_interval, tick_interval)
        
        ax.set_rticks(radial_ticks)
    else:
        ax.set_rticks([])
    
    # Customize tick formatting with smaller fonts
    ax.set_thetagrids(np.arange(0, 360, 45), 
                     ['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'], fontsize=tick_fontsize)
    ax.tick_params(axis='x', labelsize=tick_fontsize, pad=3)
    ax.tick_params(axis='y', labelsize=tick_fontsize-1)
    
    # Set radial label position to avoid overlap
    ax.set_rlabel_position(135)
    
    # Add grid with limited radial lines
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, pad=20, fontsize=12)

    if created_fig and show_plot:
        plt.tight_layout()
        plt.show()

    return ax




##### Metric Over Time Plotting #####


def plot_metric_over_time(analysis_results, show_plot=True, ax=None, show_error=True, 
                         color='blue', alpha=0.3, show_individuals=False, label=None,
                         xlabel='Time (s)', ylabel='Metric', title=None, ylim=None,
                         show_slope=True, slope_color='blue', show_xlabel=True, show_ylabel=True, min_amplitude=None):
    """
    General function to plot any metric over time with error bars across individuals.
    """
    # Extract data using standardized keys
    time_centers = analysis_results.get('time_centers', [])
    mean_values = analysis_results.get('mean_metric', [])
    se_values = analysis_results.get('se_metric', [])
    individual_arrays = analysis_results.get('metric_arrays', [])
    n_subjects = analysis_results.get('n_larvae', 0)
    
    if len(time_centers) == 0 or len(mean_values) == 0:
        if ax is None and show_plot:
            fig, ax = plt.subplots(figsize=(8, 4))
        if ax is not None:
            ax.text(0.5, 0.5, f"No {ylabel.lower()} data available for plotting.", 
                   ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Create axis if none provided
    if ax is None and show_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        created_fig = False
    
    # Convert to numpy arrays for calculations
    time_centers = np.array(time_centers)
    mean_values = np.array(mean_values)
    
    # Plot individual traces if requested
    if show_individuals and len(individual_arrays) > 0:
        for i in range(len(individual_arrays)):
            ax.plot(time_centers, individual_arrays[i], 
                   alpha=0.2, color=color, linewidth=0.5)
    
    # Plot mean line
    ax.plot(time_centers, mean_values, color=color, linewidth=2)
    
    # Plot error bars if requested
    if show_error and len(se_values) > 0 and n_subjects > 1:
        ax.fill_between(time_centers, 
                       np.array(mean_values) - np.array(se_values), 
                       np.array(mean_values) + np.array(se_values),
                       alpha=alpha, color=color)
    
    # Fit and plot slope if requested
    slope_text = ""
    if show_slope and len(time_centers) > 1:
        # Remove NaN values for fitting
        valid_mask = ~(np.isnan(time_centers) | np.isnan(mean_values))
        if np.sum(valid_mask) > 1:
            time_valid = time_centers[valid_mask]
            values_valid = mean_values[valid_mask]
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_valid, values_valid)
            
            # Generate fitted line
            fitted_line = slope * time_centers + intercept
            
            # Plot fitted line
            ax.plot(time_centers, fitted_line, 
                   color=slope_color, linestyle='--', linewidth=1.5)
            
            # Create legend text
            slope_text = f'Slope: {slope:.2e}\nR²={r_value**2:.3f}, p={p_value:.3f}'
                
    # Remove all spines except bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Detach axes from each other
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    
    # Set axis limits and ticks
    ax.set_xlim(0, 600)
    ax.set_xticks([0, 150, 300, 450, 600])
    
    # Set y-axis limits and ensure it ends at a tick
    if ylim:
        ax.set_ylim(ylim)
        y_max = ylim[1]
    elif 'Turn Amplitude' in ylabel:
        # For amplitude plots, set appropriate range
        data_max = np.nanmax(mean_values + np.array(se_values)) if len(se_values) > 0 else np.nanmax(mean_values)
        y_min = 60  # Minimum turn amplitude
        y_max = max(180, np.ceil(data_max / 10) * 10)  # Round up to nearest 10°, min 180°
        ax.set_ylim(y_min, y_max)
    else:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max)
    
    # Set y-ticks to nice intervals
    if 'Turn Amplitude' in ylabel:
        # For amplitude plots, use 20° or 30° intervals
        if y_max <= 120:
            tick_interval = 20
        else:
            tick_interval = 30
        y_ticks = np.arange(60, y_max + tick_interval, tick_interval)
    elif ylim and ylim[1] <= 0.1:
        # For small ranges (like backup), use 0.025 intervals
        n_ticks = int(y_max / 0.025)
        y_ticks = np.linspace(0, n_ticks * 0.025, n_ticks + 1)
    elif ylim and ylim[1] <= 0.5:
        # For medium ranges, use 0.1 intervals
        n_ticks = int(y_max / 0.1)
        y_ticks = np.linspace(0, n_ticks * 0.1, n_ticks + 1)
    else:
        # Auto-determine reasonable ticks
        n_ticks = 5
        y_ticks = np.linspace(0, y_max, n_ticks + 1)
    
    ax.set_yticks(y_ticks)
    
    # Set labels
    if show_xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    # Remove grid
    ax.grid(False)
    
    # Show slope legend with custom positioning and formatting
    if slope_text:
        ax.text(0.98, 0.98, slope_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=10)
    
    # Show plot if we created our own figure
    if created_fig and show_plot:
        plt.tight_layout()
        plt.show()
    
    return ax




### Specific Plotting Functions Using the Generalized Functions ###

def plot_run_probabilities(analysis_results, show_xlabel=True, show_ylabel=True, show_legend=False, **kwargs):
    return plot_orientation_histogram(
        analysis_results, 
        color='black',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_legend=show_legend,
        **kwargs
    )

def plot_turn_probabilities(analysis_results, show_xlabel=True, show_ylabel=True, show_legend=False, **kwargs):
    return plot_orientation_histogram(
        analysis_results, 
        color='red',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_legend=show_legend,
        **kwargs
    )

def plot_backup_probabilities(analysis_results, show_xlabel=True, show_ylabel=True, show_legend=False, **kwargs):
    return plot_orientation_histogram(
        analysis_results, 
        color='cyan',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_legend=show_legend,
        **kwargs
    )

def plot_turn_amplitudes(analysis_results, show_xlabel=True, show_ylabel=True, show_legend=False, min_amplitude=None, **kwargs):
    return plot_orientation_histogram(
        analysis_results, 
        color='red',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_legend=show_legend,
        min_amplitude=min_amplitude,
        **kwargs
    )

def plot_run_velocity(analysis_results, show_xlabel=True, show_ylabel=True, show_legend=False, **kwargs):
    return plot_orientation_histogram(
        analysis_results, 
        color='black',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_legend=show_legend,
        **kwargs
    )

def plot_head_cast_frequency(analysis_results, show_xlabel=True, show_ylabel=True, show_legend=False, **kwargs):
    """Plot head cast frequency by orientation."""
    return plot_orientation_histogram(
        analysis_results, 
        color='purple',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_legend=show_legend,
        **kwargs
    )


# Update the polar plotting functions
def plot_run_probabilities_polar(analysis_results, n_radial_ticks=3, **kwargs):
    return plot_orientation_histogram_polar(
        analysis_results, 
        color='black',
        n_radial_ticks=n_radial_ticks,
        **kwargs
    )

def plot_turn_probabilities_polar(analysis_results, n_radial_ticks=3, **kwargs):
    return plot_orientation_histogram_polar(
        analysis_results, 
        color='red',
        n_radial_ticks=n_radial_ticks,
        **kwargs
    )

def plot_backup_probabilities_polar(analysis_results, n_radial_ticks=3, **kwargs):
    return plot_orientation_histogram_polar(
        analysis_results, 
        color='cyan',
        n_radial_ticks=n_radial_ticks,
        **kwargs
    )

def plot_turn_amplitudes_polar(analysis_results, n_radial_ticks=3, min_amplitude=None, **kwargs):
    return plot_orientation_histogram_polar(
        analysis_results, 
        color='red',
        n_radial_ticks=n_radial_ticks,
        is_amplitude=True,
        min_amplitude=min_amplitude,
        **kwargs
    )

def plot_run_velocity_polar(analysis_results, n_radial_ticks=3, **kwargs):
    """
    Plot run velocity by orientation in polar coordinates.
    """    
    return plot_orientation_histogram_polar(
        analysis_results, 
        color='black',
        n_radial_ticks=n_radial_ticks,
        **kwargs
    )

def plot_head_cast_frequency_polar(analysis_results, n_radial_ticks=3, **kwargs):
    """Plot head cast frequency by orientation in polar coordinates."""
    return plot_orientation_histogram_polar(
        analysis_results, 
        color='purple',
        n_radial_ticks=n_radial_ticks,
        **kwargs
    )

### Plots over time ###

def plot_run_probability_over_time(analysis_results, show_xlabel=True, show_ylabel=True, **kwargs):
    return plot_metric_over_time(
        analysis_results,
        color='black',
        slope_color='black',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        **kwargs
    )

def plot_turn_probability_over_time(analysis_results, show_xlabel=True, show_ylabel=True, **kwargs):
    return plot_metric_over_time(
        analysis_results,
        color='red',
        slope_color='red',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        **kwargs
    )

def plot_backup_probability_over_time(analysis_results, show_xlabel=True, show_ylabel=True, **kwargs):
    return plot_metric_over_time(
        analysis_results,
        color='cyan',
        slope_color='cyan',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        **kwargs
    )

def plot_turn_amplitudes_over_time(analysis_results, show_xlabel=True, show_ylabel=True, min_amplitude=None, **kwargs):
    return plot_metric_over_time(
        analysis_results, 
        color='red',
        slope_color='red',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        min_amplitude=min_amplitude,
        **kwargs
    )

def plot_run_velocity_over_time(analysis_results, show_xlabel=True, show_ylabel=True, **kwargs):
    """Plot run velocity over time with error bars."""
    return plot_metric_over_time(
        analysis_results,
        color='black',
        slope_color='black',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        **kwargs
    )


def plot_head_cast_frequency_over_time(analysis_results, show_xlabel=True, show_ylabel=True, **kwargs):
    """Plot head cast frequency over time with error bars."""
    return plot_metric_over_time(
        analysis_results,
        color='purple',
        slope_color='purple',
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        **kwargs
    )

##### Head Cast Detection Plotting #####
def plot_cast_detection_results(experiments_data, cast_events_data, larva_ids=None, 
                               figsize=(20,20), save_path=None, show_plots=True,
                               time_range=None, smooth_sigma=4.0, max_larvae_per_figure=4):
    """
    Plot cast detection results showing head angles, body orientation, behavioral states, 
    and detected head casts for multiple larvae as subplots.
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        cast_events_data: Output from detect_head_casts_in_casts function
        larva_ids: List of larva IDs to plot (if None, plots all)
        figsize: Figure size for the entire plot
        save_path: Directory to save plots (optional)
        show_plots: Whether to display plots
        time_range: Tuple (start_time, end_time) to limit plotting range
        smooth_sigma: Smoothing parameter (should match detection)
        max_larvae_per_figure: Maximum number of larvae per figure
        
    Returns:
        List of figure objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import math
    
    # Handle data structure
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data
    
    # Determine which larvae to plot
    if larva_ids is None:
        larva_ids = list(cast_events_data.keys())
    
    # Filter to only larvae that have data
    valid_larva_ids = []
    for larva_id in larva_ids:
        if larva_id in data_to_process and larva_id in cast_events_data:
            valid_larva_ids.append(larva_id)
        else:
            print(f"⚠️  Skipping larva {larva_id} - missing data")
    
    if not valid_larva_ids:
        print("No valid larvae found for plotting")
        return []
    
    # Split larvae into groups if there are too many
    larva_groups = []
    for i in range(0, len(valid_larva_ids), max_larvae_per_figure):
        larva_groups.append(valid_larva_ids[i:i + max_larvae_per_figure])
    
    figures = []
    
    for group_idx, larva_group in enumerate(larva_groups):
        n_larvae = len(larva_group)
        
        # Calculate subplot layout
        n_cols = 1  # Single column
        n_rows = n_larvae
        
        # Create figure
        fig = plt.figure(figsize=(figsize[0], figsize[1] * n_larvae / 4))
        
        for subplot_idx, larva_id in enumerate(larva_group):
            larva_data = data_to_process[larva_id]
            cast_events = cast_events_data[larva_id]
            
            try:
                # Extract required data
                if not all(k in larva_data for k in ['t', 'global_state_small_large_state', 'angle_upper_lower_smooth_5']):
                    print(f"⚠️  Skipping larva {larva_id} - missing required fields")
                    continue
                    
                t = np.array(larva_data['t']).flatten()
                states = np.array(larva_data['global_state_small_large_state']).flatten()
                head_angles = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
                
                # Get orientation
                orientations = data_processor.get_larva_orientation_array(larva_data)
                if orientations is None:
                    print(f"⚠️  Skipping larva {larva_id} - could not compute orientation")
                    continue
                
                # Ensure same length
                min_len = min(len(t), len(states), len(head_angles), len(orientations))
                t = t[:min_len]
                states = states[:min_len]
                head_angles = head_angles[:min_len]
                orientations = orientations[:min_len]
                
                # Apply time range filter if specified
                if time_range is not None:
                    time_mask = (t >= time_range[0]) & (t <= time_range[1])
                    t = t[time_mask]
                    states = states[time_mask]
                    head_angles = head_angles[time_mask]
                    orientations = orientations[time_mask]
                    
                    # Adjust indices in cast events
                    adjusted_cast_events = []
                    for cast_event in cast_events:
                        if (cast_event['cast_start_time'] >= time_range[0] and 
                            cast_event['cast_start_time'] <= time_range[1]):
                            adjusted_cast_events.append(cast_event)
                    cast_events = adjusted_cast_events
                
                # Convert head angles to degrees and apply same smoothing as detection
                head_angles_deg = np.degrees(head_angles)
                head_angles_smooth = gaussian_filter1d(head_angles_deg, smooth_sigma/3.0)
                
                # Create subplot with two y-axes
                ax1 = plt.subplot(n_rows, n_cols, subplot_idx + 1)
                ax2 = ax1.twinx()
                
                # Define colors for behavioral states
                state_colors = {
                    1: 'black',    # run
                    2: 'red',      # cast
                    3: 'green',    # stop
                    4: 'blue',     # hunch
                    5: 'cyan',     # backup
                    6: 'yellow'    # roll
                }
                
                # Plot behavioral states as background bars on bottom axis
                y_bottom = -160  # Bottom of the behavioral state bars
                bar_height = 40  # Height of behavioral state bars
                
                # Group consecutive states for efficiency
                i = 0
                while i < len(states):
                    current_state = states[i]
                    start_time = t[i]
                    
                    # Find end of this state segment
                    while i < len(states) and states[i] == current_state:
                        i += 1
                    end_time = t[i-1] if i-1 < len(t) else t[-1]
                    
                    # Plot state bar
                    color = state_colors.get(current_state, 'gray')
                    alpha = 0.7 if current_state in [1, 2, 5] else 0.3  # Highlight run, cast, backup
                    ax1.barh(y_bottom, end_time - start_time, left=start_time, 
                            height=bar_height, color=color, alpha=alpha, edgecolor='none')
                
                # Plot head angles (smoothed) on primary axis
                ax1.plot(t, head_angles_smooth, 'purple', linewidth=1.5, label='Head Angle (smoothed)', alpha=0.8)
                
                # Plot body orientation on secondary axis
                ax2.plot(t, orientations, 'orange', linewidth=1.5, label='Body Orientation', alpha=0.8)
                
                # Plot detected head casts as asterisks
                for cast_event in cast_events:
                    for head_cast in cast_event['head_cast_details']:
                        peak_time = head_cast['peak_time']
                        bend_angle = head_cast['bend_angle']
                        direction = head_cast['direction']
                        
                        # Color asterisk by direction
                        color = 'red' if direction == 'upstream' else 'blue'
                        marker_size = 10
                        
                        ax1.plot(peak_time, bend_angle, '*', color=color, markersize=marker_size,
                                markeredgecolor='black', markeredgewidth=0.5, zorder=10)
                
                # Plot cast event boundaries as vertical lines
                for cast_event in cast_events:
                    ax1.axvline(cast_event['cast_start_time'], color='red', linestyle='--', 
                               alpha=0.6, linewidth=1, zorder=5)
                    ax1.axvline(cast_event['cast_end_time'], color='red', linestyle='--', 
                               alpha=0.6, linewidth=1, zorder=5)
                
                # Formatting
                ax1.set_ylabel('Head Angle (°)', color='purple', fontsize=10)
                ax2.set_ylabel('Body Orientation (°)', color='orange', fontsize=10)
                
                # Set y-axis limits for both to -180 to 180
                ax1.set_ylim(-180, 180)
                ax2.set_ylim(-180, 180)
                
                # Color y-axis labels to match plots
                ax1.tick_params(axis='y', labelcolor='purple', labelsize=8)
                ax2.tick_params(axis='y', labelcolor='orange', labelsize=8)
                
                # Add horizontal reference lines
                ax1.axhline(0, color='gray', linestyle='-', alpha=0.3, zorder=1)
                ax2.axhline(0, color='gray', linestyle='-', alpha=0.3, zorder=1)
                ax2.axhline(90, color='gray', linestyle='--', alpha=0.3, zorder=1)
                ax2.axhline(-90, color='gray', linestyle='--', alpha=0.3, zorder=1)
                ax2.axhline(180, color='gray', linestyle='--', alpha=0.3, zorder=1)
                ax2.axhline(-180, color='gray', linestyle='--', alpha=0.3, zorder=1)
                
                # Add larva ID and summary stats as title
                n_cast_events = len(cast_events)
                total_head_casts = sum(cast_event['total_head_casts'] for cast_event in cast_events)
                total_upstream = sum(cast_event['n_upstream_head_casts'] for cast_event in cast_events)
                total_downstream = sum(cast_event['n_downstream_head_casts'] for cast_event in cast_events)
                
                title = (f'Larva {larva_id}: {n_cast_events} casts, {total_head_casts} head casts '
                        f'({total_upstream}↑, {total_downstream}↓)')
                ax1.set_title(title, fontsize=11, pad=10)
                
                # Only add x-label to bottom subplot
                if subplot_idx == len(larva_group) - 1:
                    ax1.set_xlabel('Time (s)', fontsize=10)
                else:
                    ax1.set_xticklabels([])
                
                # Add legend only to first subplot
                if subplot_idx == 0:
                    from matplotlib.lines import Line2D
                    from matplotlib.patches import Patch
                    
                    legend_elements = [
                        Line2D([0], [0], color='purple', linewidth=2, label='Head Angle'),
                        Line2D([0], [0], color='orange', linewidth=2, label='Body Orientation'),
                        Line2D([0], [0], marker='*', color='red', linewidth=0, markersize=8, 
                              label='Upstream Head Cast'),
                        Line2D([0], [0], marker='*', color='blue', linewidth=0, markersize=8, 
                              label='Downstream Head Cast'),
                        Line2D([0], [0], color='red', linestyle='--', label='Cast Boundaries'),
                        Patch(facecolor='black', alpha=0.7, label='Run'),
                        Patch(facecolor='red', alpha=0.7, label='Cast'),
                        Patch(facecolor='cyan', alpha=0.7, label='Backup')
                    ]
                    
                    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                              fontsize=8, ncol=2)
                
                # Add behavioral state labels at bottom of last subplot
                if subplot_idx == len(larva_group) - 1:
                    ax1.text(0.02, -0.15, 'Behaviors: Black=Run, Red=Cast, Cyan=Backup, Green=Stop, Blue=Hunch, Yellow=Roll',
                            transform=ax1.transAxes, fontsize=8, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"❌ Error plotting larva {larva_id}: {e}")
                continue
        
        # Overall figure title
        if len(larva_groups) > 1:
            fig.suptitle(f'Cast Detection Results - Group {group_idx + 1}/{len(larva_groups)}', 
                        fontsize=14, y=0.98)
        else:
            fig.suptitle('Cast Detection Results', fontsize=14, y=0.98)
        
        # Tight layout with padding for suptitle
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save if requested
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            if len(larva_groups) > 1:
                fig_path = os.path.join(save_path, f'cast_detection_group_{group_idx + 1}.pdf')
            else:
                fig_path = os.path.join(save_path, 'cast_detection_results.pdf')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', 
                       transparent=True, facecolor='none')
            print(f"💾 Saved: {fig_path}")
        
        figures.append(fig)
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    return figures

def plot_cast_detection_summary(cast_events_data, experiments_data, figsize=(12, 8), 
                               save_path=None, show_plot=True):
    """
    Create a summary plot showing statistics across all larvae.
    
    Args:
        cast_events_data: Output from detect_head_casts_in_casts function
        experiments_data: Dictionary of larva data
        figsize: Figure size
        save_path: Path to save figure (optional)
        show_plot: Whether to display the plot
        
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Collect summary statistics
    larva_stats = []
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        n_cast_events = len(cast_events)
        total_head_casts = sum(cast_event['total_head_casts'] for cast_event in cast_events)
        total_upstream = sum(cast_event['n_upstream_head_casts'] for cast_event in cast_events)
        total_downstream = sum(cast_event['n_downstream_head_casts'] for cast_event in cast_events)
        
        # Calculate average cast duration
        avg_cast_duration = np.mean([cast_event['cast_duration'] for cast_event in cast_events])
        
        larva_stats.append({
            'larva_id': larva_id,
            'n_cast_events': n_cast_events,
            'total_head_casts': total_head_casts,
            'total_upstream': total_upstream,
            'total_downstream': total_downstream,
            'avg_cast_duration': avg_cast_duration,
            'upstream_bias': total_upstream / total_head_casts if total_head_casts > 0 else 0
        })
    
    if not larva_stats:
        print("No cast data found for summary plot")
        return None
    
    # Create summary figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Cast events per larva
    larva_ids = [stat['larva_id'] for stat in larva_stats]
    n_cast_events = [stat['n_cast_events'] for stat in larva_stats]
    
    ax1.bar(range(len(larva_ids)), n_cast_events, color='red', alpha=0.7)
    ax1.set_xlabel('Larva ID')
    ax1.set_ylabel('Number of Cast Events')
    ax1.set_title('Cast Events per Larva')
    ax1.set_xticks(range(len(larva_ids)))
    ax1.set_xticklabels(larva_ids, rotation=45)
    
    # Plot 2: Head casts per larva
    total_head_casts = [stat['total_head_casts'] for stat in larva_stats]
    
    ax2.bar(range(len(larva_ids)), total_head_casts, color='purple', alpha=0.7)
    ax2.set_xlabel('Larva ID')
    ax2.set_ylabel('Total Head Casts')
    ax2.set_title('Head Casts per Larva')
    ax2.set_xticks(range(len(larva_ids)))
    ax2.set_xticklabels(larva_ids, rotation=45)
    
    # Plot 3: Upstream vs Downstream bias
    upstream_bias = [stat['upstream_bias'] for stat in larva_stats]
    
    colors = ['red' if bias > 0.5 else 'blue' for bias in upstream_bias]
    ax3.bar(range(len(larva_ids)), upstream_bias, color=colors, alpha=0.7)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Larva ID')
    ax3.set_ylabel('Upstream Bias')
    ax3.set_title('Head Cast Direction Bias per Larva')
    ax3.set_xticks(range(len(larva_ids)))
    ax3.set_xticklabels(larva_ids, rotation=45)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Average cast duration
    avg_durations = [stat['avg_cast_duration'] for stat in larva_stats]
    
    ax4.bar(range(len(larva_ids)), avg_durations, color='green', alpha=0.7)
    ax4.set_xlabel('Larva ID')
    ax4.set_ylabel('Average Cast Duration (frames)')
    ax4.set_title('Average Cast Duration per Larva')
    ax4.set_xticks(range(len(larva_ids)))
    ax4.set_xticklabels(larva_ids, rotation=45)
    
    # Add overall statistics as text
    total_larvae = len(larva_stats)
    total_casts = sum(n_cast_events)
    total_hc = sum(total_head_casts)
    overall_upstream_bias = sum(stat['total_upstream'] for stat in larva_stats) / total_hc if total_hc > 0 else 0
    
    stats_text = (f"Overall Statistics:\n"
                 f"Larvae: {total_larvae}\n"
                 f"Cast Events: {total_casts}\n"
                 f"Head Casts: {total_hc}\n"
                 f"Upstream Bias: {overall_upstream_bias:.2f}")
    
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   transparent=True, facecolor='none')
        print(f"💾 Summary plot saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

def plot_head_cast_bias_perpendicular(bias_results, figsize=(4, 6), save_path=None, ax=None, title=None):
    """
    Plot analysis of head cast bias when larvae are perpendicular to flow.
    Shows bias towards upstream vs downstream using box plots.
    
    Args:
        bias_results: Output from analyze_first_head_cast_bias_perpendicular
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        ax: Optional matplotlib axis to plot on
        title: Optional custom title for the plot
        
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Create axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
    
    # Check if we have data
    if not bias_results or not bias_results.get('head_cast_data'):
        ax.text(0.5, 0.5, 'No perpendicular cast events found', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Head Cast Bias Analysis (Perpendicular to Flow)')
        return fig
    
    # Get per-larva data
    larva_upstream_biases = [summary['upstream_bias'] for summary in bias_results['larva_summaries']]
    larva_downstream_biases = [summary['downstream_bias'] for summary in bias_results['larva_summaries']]
    
    # Data for box plot
    data = [larva_upstream_biases, larva_downstream_biases]
    positions = [1, 2]
    labels_box = ['Towards Wind', 'Away from Wind']
    color = 'purple'  # Single color for both bars
    
    # Create box plot - show outliers but not individual points
    bp = ax.boxplot(data, positions=positions, notch=True, patch_artist=True, 
                   widths=0.6, showfliers=True)
    
    # Color boxes the same purple color
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=color, alpha=0.6)
        bp['medians'][i].set(color='black', linewidth=2)
    
    # Add median values as text on each box
    for i, (pos, d) in enumerate(zip(positions, data)):
        if len(d) > 0:
            median = np.median(d)
            
            # Display median
            ax.text(pos, median, f"{median:.2f}", ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
    
    # Add p-value annotation for difference between upstream and downstream
    if len(larva_upstream_biases) > 1 and len(larva_downstream_biases) > 1:
        tstat, pval = stats.ttest_rel(larva_upstream_biases, larva_downstream_biases)
        if pval < 0.001:
            ptext = "***"
        elif pval < 0.01:
            ptext = "**"
        elif pval < 0.05:
            ptext = "*"
        else:
            ptext = "ns"
            
        # Add horizontal line with ticks at the ends
        y_pos = 0.9
        ax.plot([1, 2], [y_pos, y_pos], 'k-', lw=1)
        # Add ticks at the ends
        ax.plot([1, 1], [y_pos-0.01, y_pos+0.01], 'k-', lw=1)
        ax.plot([2, 2], [y_pos-0.01, y_pos+0.01], 'k-', lw=1)
        ax.text(1.5, y_pos + 0.02, ptext, ha='center', va='bottom', fontsize=12)
    
    # Add significance markers for comparison to chance (0.5)
    for i, (pos, d, label) in enumerate(zip(positions, data, labels_box)):
        if len(d) > 1:
            tstat, pval = stats.ttest_1samp(d, 0.5)
            if pval < 0.05:
                stars = "*" * sum([pval < p for p in [0.05, 0.01, 0.001]])
                ax.text(pos, 0.45, stars, ha='center', va='top', fontsize=16)
    
    # Add a dashed line at 0.5 (chance level)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Configure the plot with detached axes
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel('Probability of Cast Direction')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_box)
    
    # Set y-axis from 0 to 1 with ticks at 0 and 1
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
    
    # Detach axes - remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Move left and bottom spines away from data
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    
    # Title with analysis type
    if title:
        ax.set_title(title, fontsize=12, pad=15)
    else:
        analysis_type = bias_results.get('analysis_type', 'first')
        title_map = {'first': 'First Head Cast', 'last': 'Last Head Cast', 'all': 'All Head Casts'}
        ax.set_title(f'{title_map.get(analysis_type, "Head Cast")} Bias\n(Perpendicular to Flow)',
                    fontsize=12, pad=15)
    
    # Add sample size as small text
    n_larvae = bias_results.get('n_larvae', 0)
    total_casts = bias_results.get('total_casts', 0)
    ax.text(0.02, 0.98, f'n = {n_larvae} larvae\n{total_casts} casts', 
           transform=ax.transAxes, fontsize=8, va='top', ha='left', 
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Only save if we created our own figure
    if save_path and created_fig:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Only show if we created our own figure
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    # Print summary to console only if we created our own figure
    if created_fig:
        analysis_type = bias_results.get('analysis_type', 'first')
        print(f"\n=== HEAD CAST DIRECTION BIAS ANALYSIS ({analysis_type.upper()}) ===")
        print(f"Total perpendicular cast events: {bias_results.get('total_casts', 0)}")
        print(f"Number of larvae: {bias_results.get('n_larvae', 0)}")
        print(f"Upstream head casts: {bias_results.get('total_upstream', 0)} ({bias_results.get('overall_upstream_bias', 0):.1%})")
        print(f"Downstream head casts: {bias_results.get('total_downstream', 0)} ({bias_results.get('overall_downstream_bias', 0):.1%})")
        print(f"Binomial test p-value: {bias_results.get('p_value_binomial', 1.0):.4f}")
        
        if bias_results.get('p_value_binomial', 1.0) < 0.05:
            bias_direction = "upstream" if bias_results.get('overall_upstream_bias', 0) > 0.5 else "downstream"
            print(f"RESULT: Significant bias toward {bias_direction} head casts (p < 0.05)")
        else:
            print("RESULT: No significant bias detected (p ≥ 0.05)")
    
    return fig



#### NAVIGATIONAL INDEX CALCULATION ####
def plot_navigational_index_over_time(ni_time_results, figsize=(8, 6), save_path=None, 
                                     show_individuals=False, show_error=True, ax=None):
    """
    Plot NI_x and NI_y over time.
    
    Args:
        ni_time_results: Output from analyze_navigational_index_over_time
        figsize: Figure size
        save_path: Optional path to save figure
        show_individuals: Whether to show individual larva traces
        show_error: Whether to show error bars
        ax: Optional axis to plot on
        
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
    
    time_centers = ni_time_results['time_centers']
    mean_NI_x = ni_time_results['mean_NI_x']
    mean_NI_y = ni_time_results['mean_NI_y']
    
    # Plot individual larvae if requested
    if show_individuals:
        NI_x_arrays = ni_time_results['NI_x_arrays']
        NI_y_arrays = ni_time_results['NI_y_arrays']
        
        for i in range(len(NI_x_arrays)):
            ax.plot(time_centers, NI_x_arrays[i], color='blue', alpha=0.1, linewidth=0.5)
            ax.plot(time_centers, NI_y_arrays[i], color='red', alpha=0.1, linewidth=0.5)
    
    # Plot means with shaded SEM instead of error bars
    if show_error:
        se_NI_x = ni_time_results['se_NI_x']
        se_NI_y = ni_time_results['se_NI_y']
        
        # Plot mean lines
        ax.plot(time_centers, mean_NI_x, color='blue', label='Mean NI_x', linewidth=2)
        ax.plot(time_centers, mean_NI_y, color='red', label='Mean NI_y', linewidth=2)
        
        # Add shaded SEM regions
        ax.fill_between(time_centers, 
                       np.array(mean_NI_x) - np.array(se_NI_x), 
                       np.array(mean_NI_x) + np.array(se_NI_x),
                       color='blue', alpha=0.3, label='NI_x ± SEM')
        ax.fill_between(time_centers, 
                       np.array(mean_NI_y) - np.array(se_NI_y), 
                       np.array(mean_NI_y) + np.array(se_NI_y),
                       color='red', alpha=0.3, label='NI_y ± SEM')
    else:
        ax.plot(time_centers, mean_NI_x, color='blue', label='Mean NI_x', linewidth=2)
        ax.plot(time_centers, mean_NI_y, color='red', label='Mean NI_y', linewidth=2)
    
    # Formatting
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Navigational Index')
    ax.set_title('Navigational Index Over Time')
    ax.set_ylim(-0.5, 0.5)  # Set y-axis range from -0.5 to 0.5
    ax.legend()
    ax.grid(False)  # Remove grid
    
    # Add summary statistics
    mean_NI_x_val = np.nanmean(mean_NI_x)
    mean_NI_y_val = np.nanmean(mean_NI_y)
    n_larvae = ni_time_results['n_larvae']
    
    summary_text = f"""n = {n_larvae} larvae
Mean NI_x: {mean_NI_x_val:.3f}
Mean NI_y: {mean_NI_y_val:.3f}"""
    
    ax.text(0.98, 0.02, summary_text,
           ha='right', va='bottom', transform=ax.transAxes,
           fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    return fig

def plot_navigational_index_boxplot(ni_single_results, figsize=(6, 6), save_path=None, ax=None):
    """
    Plot box plots for NI_x and NI_y values with significance testing.
    Tests if each distribution is significantly different from 0.
    
    Args:
        ni_single_results: Output from analyze_navigational_index_single_values
        figsize: Figure size
        save_path: Optional path to save figure
        ax: Optional axis to plot on
        
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
    
    # Extract data
    NI_x_clean = ni_single_results['NI_x_clean']
    NI_y_clean = ni_single_results['NI_y_clean']
    p_values = ni_single_results['p_values']
    significances = ni_single_results['significances']
    means = ni_single_results['means']
    n_larvae = ni_single_results['n_larvae']
    
    # Prepare data for box plot
    data_to_plot = [NI_x_clean, NI_y_clean]
    labels = ['NI_x', 'NI_y']
    colors = ['blue', 'red']
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True, showfliers=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add significance markers and p-values
    for i, label in enumerate(labels):
        sig = significances[label]
        p_val = p_values[label]
        
        # Get the position for the significance marker
        if len(data_to_plot[i]) > 0:
            y_pos = np.max(data_to_plot[i])
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_offset = y_range * 0.05
            
            # if sig not in ["ns", "insufficient data"]:
            #     ax.text(i + 1, y_pos + y_offset, sig, 
            #            ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            # # Add p-value as smaller text
            # if not np.isnan(p_val):
            #     ax.text(i + 1, y_pos + 2*y_offset, f'p={p_val:.3f}', 
            #            ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # Add median and quartile values
    for i, label in enumerate(labels):
        if len(data_to_plot[i]) > 0:
            median = np.median(data_to_plot[i])
            
            # Display median
            ax.text(i + 1, median, f"{median:.3f}", ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
        
    
    # Formatting
    ax.set_ylabel('Navigational Index')
    ax.set_title('Navigational Index Values (Time-Averaged)')
    ax.grid(False)  # Remove grid
    
    # Set axis limits and styling
    ax.set_ylim(-0.3, 0.2)  # Set y-axis range from -0.5 to 0.2
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    
    # Add sample size information
    ax.text(0.02, 0.98, f'n = {n_larvae} larvae', transform=ax.transAxes, 
           fontsize=10, va='top', ha='left', 
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    return fig

def plot_navigational_index_combined_experiments(result_files, figsize=(15, 6), save_path=None):
    """
    Plot combined NI data across multiple experiment dates.
    Creates two subplots: one for NI_x and one for NI_y.
    Each subplot shows overall combined data followed by individual experiment dates.
    
    Args:
        result_files: List of HDF5 file paths from different experiments
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Load and extract NI data from all files
    all_ni_x_data = []
    all_ni_y_data = []
    experiment_dates = []
    per_date_ni_x = {}
    per_date_ni_y = {}
    
    for filepath in result_files:
        try:
            results = data_loader.load_analysis_results(filepath)
            
            # Extract experiment date from path
            path_parts = filepath.split('/')
            experiment_date = 'unknown'
            for part in path_parts:
                if len(part) == 15 and part.startswith('202'):  # Format: 20240226_145653
                    experiment_date = part[:8]  # Take just the date part
                    break
            
            if 'ni_single_results' in results:
                ni_data = results['ni_single_results']
                ni_x_clean = ni_data.get('NI_x_clean', [])
                ni_y_clean = ni_data.get('NI_y_clean', [])
                
                if len(ni_x_clean) > 0 and len(ni_y_clean) > 0:
                    all_ni_x_data.extend(ni_x_clean)
                    all_ni_y_data.extend(ni_y_clean)
                    
                    # Store per-date data
                    if experiment_date not in per_date_ni_x:
                        per_date_ni_x[experiment_date] = []
                        per_date_ni_y[experiment_date] = []
                        experiment_dates.append(experiment_date)
                    
                    per_date_ni_x[experiment_date].extend(ni_x_clean)
                    per_date_ni_y[experiment_date].extend(ni_y_clean)
                    
        except Exception as e:
            print(f"⚠️  Could not load data from {filepath}: {e}")
    
    # Sort experiment dates
    experiment_dates = sorted(list(set(experiment_dates)))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Function to perform significance testing
    def test_significance(values, test_value=0):
        if len(values) < 3:
            return np.nan, "insufficient data"
        t_stat, p_value = stats.ttest_1samp(values, test_value)
        if p_value < 0.001:
            return p_value, "***"
        elif p_value < 0.01:
            return p_value, "**"
        elif p_value < 0.05:
            return p_value, "*"
        else:
            return p_value, "ns"
    
    # Plot NI_x data
    if len(all_ni_x_data) > 0:
        # Prepare data for box plot: overall + per-date
        ni_x_plot_data = [all_ni_x_data]  # Overall data first
        labels_x = ['Overall']
        colors_x = ['blue']
        
        # Add per-date data
        for date in experiment_dates:
            if date in per_date_ni_x and len(per_date_ni_x[date]) > 0:
                ni_x_plot_data.append(per_date_ni_x[date])
                labels_x.append(date)
                colors_x.append('lightblue')
        
        # Create box plot for NI_x
        bp1 = ax1.boxplot(ni_x_plot_data, labels=labels_x, patch_artist=True, 
                         notch=True, showfliers=True)
        
        # Color the boxes
        for i, (patch, color) in enumerate(zip(bp1['boxes'], colors_x)):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add significance testing for overall data
        p_val, sig = test_significance(all_ni_x_data)
        if sig not in ["ns", "insufficient data"]:
            y_pos = np.max(all_ni_x_data)
            y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
            y_offset = y_range * 0.05
            ax1.text(1, y_pos + y_offset, sig, ha='center', va='bottom', 
                    fontsize=16, fontweight='bold')
        
        # Add median and quartile annotations for overall data
        if len(all_ni_x_data) > 0:
            median = np.median(all_ni_x_data)
            q25 = np.percentile(all_ni_x_data, 25)
            q75 = np.percentile(all_ni_x_data, 75)
            
            ax1.text(1, median, f"{median:.3f}", ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
            
            ax1.text(1.4, q25, f"Q1: {q25:.3f}", ha='left', va='center',
                    fontsize=8, color='black')
            ax1.text(1.4, q75, f"Q3: {q75:.3f}", ha='left', va='center',
                    fontsize=8, color='black')
    
    # Plot NI_y data
    if len(all_ni_y_data) > 0:
        # Prepare data for box plot: overall + per-date
        ni_y_plot_data = [all_ni_y_data]  # Overall data first
        labels_y = ['Overall']
        colors_y = ['red']
        
        # Add per-date data
        for date in experiment_dates:
            if date in per_date_ni_y and len(per_date_ni_y[date]) > 0:
                ni_y_plot_data.append(per_date_ni_y[date])
                labels_y.append(date)
                colors_y.append('lightcoral')
        
        # Create box plot for NI_y
        bp2 = ax2.boxplot(ni_y_plot_data, labels=labels_y, patch_artist=True, 
                         notch=True, showfliers=True)
        
        # Color the boxes
        for i, (patch, color) in enumerate(zip(bp2['boxes'], colors_y)):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add significance testing for overall data
        p_val, sig = test_significance(all_ni_y_data)
        if sig not in ["ns", "insufficient data"]:
            y_pos = np.max(all_ni_y_data)
            y_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
            y_offset = y_range * 0.05
            ax2.text(1, y_pos + y_offset, sig, ha='center', va='bottom', 
                    fontsize=16, fontweight='bold')
        
        # Add median and quartile annotations for overall data
        if len(all_ni_y_data) > 0:
            median = np.median(all_ni_y_data)
            q25 = np.percentile(all_ni_y_data, 25)
            q75 = np.percentile(all_ni_y_data, 75)
            
            ax2.text(1, median, f"{median:.3f}", ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
            
            ax2.text(1.4, q25, f"Q1: {q25:.3f}", ha='left', va='center',
                    fontsize=8, color='black')
            ax2.text(1.4, q75, f"Q3: {q75:.3f}", ha='left', va='center',
                    fontsize=8, color='black')
    
    # Formatting for both subplots
    for ax, title, ylabel in [(ax1, 'NI_x Distribution', 'NI_x'), 
                              (ax2, 'NI_y Distribution', 'NI_y')]:
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(-0.5, 0.2)
        ax.grid(False)
        
        # Style axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    # Add overall sample size information
    total_larvae = len(all_ni_x_data)
    n_experiments = len(experiment_dates)
    
    fig.suptitle(f'Navigational Index Across Experiments\n'
                f'n = {total_larvae} larvae from {n_experiments} experiments', 
                fontsize=14, y=0.95)
    
    # Add experiment info text
    info_text = f"""Experiments: {', '.join(experiment_dates)}
Per-experiment n: {', '.join([f"{date}: {len(per_date_ni_x.get(date, []))}" for date in experiment_dates])}"""
    
    fig.text(0.02, 0.02, info_text, fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== NAVIGATIONAL INDEX COMBINED ANALYSIS ===")
    print(f"Total larvae: {total_larvae}")
    print(f"Number of experiments: {n_experiments}")
    print(f"Experiment dates: {', '.join(experiment_dates)}")
    print(f"Overall NI_x: mean = {np.mean(all_ni_x_data):.3f}, median = {np.median(all_ni_x_data):.3f}")
    print(f"Overall NI_y: mean = {np.mean(all_ni_y_data):.3f}, median = {np.median(all_ni_y_data):.3f}")
    
    # Per-experiment summaries
    for date in experiment_dates:
        if date in per_date_ni_x and len(per_date_ni_x[date]) > 0:
            ni_x_mean = np.mean(per_date_ni_x[date])
            ni_y_mean = np.mean(per_date_ni_y[date])
            n_larvae_date = len(per_date_ni_x[date])
            print(f"{date}: n={n_larvae_date}, NI_x={ni_x_mean:.3f}, NI_y={ni_y_mean:.3f}")
    
    return fig

def plot_navigational_index_time_series_combined(result_files, figsize=(12, 6), save_path=None):
    """
    Plot combined NI time series data across multiple experiment dates.
    Shows overall mean with SEM and individual experiment means.
    
    Args:
        result_files: List of HDF5 file paths from different experiments
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load and extract time series data from all files
    all_time_data = []
    experiment_dates = []
    per_date_time_data = {}
    common_time_centers = None
    
    for filepath in result_files:
        try:
            results = data_loader.load_analysis_results(filepath)
            
            # Extract experiment date from path
            path_parts = filepath.split('/')
            experiment_date = 'unknown'
            for part in path_parts:
                if len(part) == 15 and part.startswith('202'):
                    experiment_date = part[:8]
                    break
            
            if 'ni_time_results' in results:
                ni_time_data = results['ni_time_results']
                time_centers = ni_time_data.get('time_centers', [])
                mean_NI_x = ni_time_data.get('mean_NI_x', [])
                mean_NI_y = ni_time_data.get('mean_NI_y', [])
                
                if len(mean_NI_x) > 0 and len(mean_NI_y) > 0:
                    if common_time_centers is None:
                        common_time_centers = time_centers
                    
                    all_time_data.append({
                        'date': experiment_date,
                        'time_centers': time_centers,
                        'mean_NI_x': mean_NI_x,
                        'mean_NI_y': mean_NI_y
                    })
                    
                    if experiment_date not in per_date_time_data:
                        per_date_time_data[experiment_date] = []
                        experiment_dates.append(experiment_date)
                    
                    per_date_time_data[experiment_date].append({
                        'mean_NI_x': mean_NI_x,
                        'mean_NI_y': mean_NI_y
                    })
                    
        except Exception as e:
            print(f"⚠️  Could not load time series data from {filepath}: {e}")
    
    if not all_time_data:
        print("No time series data found!")
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Combine all experiments for overall means
    all_ni_x_means = []
    all_ni_y_means = []
    
    for data in all_time_data:
        all_ni_x_means.append(data['mean_NI_x'])
        all_ni_y_means.append(data['mean_NI_y'])
    
    # Calculate overall statistics
    overall_mean_x = np.nanmean(all_ni_x_means, axis=0)
    overall_mean_y = np.nanmean(all_ni_y_means, axis=0)
    overall_sem_x = stats.sem(all_ni_x_means, axis=0, nan_policy='omit')
    overall_sem_y = stats.sem(all_ni_y_means, axis=0, nan_policy='omit')
    
    # Plot NI_x time series
    # Individual experiments (light lines)
    for data in all_time_data:
        ax1.plot(data['time_centers'], data['mean_NI_x'], 
                color='blue', alpha=0.3, linewidth=1)
    
    # Overall mean with SEM
    ax1.plot(common_time_centers, overall_mean_x, 
            color='blue', linewidth=3, label='Overall Mean')
    ax1.fill_between(common_time_centers, 
                    overall_mean_x - overall_sem_x, 
                    overall_mean_x + overall_sem_x,
                    color='blue', alpha=0.3, label='Overall ± SEM')
    
    # Plot NI_y time series
    # Individual experiments (light lines)
    for data in all_time_data:
        ax2.plot(data['time_centers'], data['mean_NI_y'], 
                color='red', alpha=0.3, linewidth=1)
    
    # Overall mean with SEM
    ax2.plot(common_time_centers, overall_mean_y, 
            color='red', linewidth=3, label='Overall Mean')
    ax2.fill_between(common_time_centers, 
                    overall_mean_y - overall_sem_y, 
                    overall_mean_y + overall_sem_y,
                    color='red', alpha=0.3, label='Overall ± SEM')
    
    # Formatting for both subplots
    for ax, title, ylabel in [(ax1, 'NI_x Over Time', 'NI_x'), 
                              (ax2, 'NI_y Over Time', 'NI_y')]:
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(-0.5, 0.5)
        ax.grid(False)
        ax.legend()
    
    n_experiments = len(experiment_dates)
    fig.suptitle(f'Navigational Index Time Series\n'
                f'{n_experiments} experiments: {", ".join(sorted(experiment_dates))}', 
                fontsize=14, y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig