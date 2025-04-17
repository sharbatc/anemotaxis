import pandas as pd
import os
import re
import numpy as np
from math import sqrt, asin
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def parse_protocol(protocol_str):
    """Parses the protocol section of the filename.
    
    Args:
        protocol_str (str): Protocol string from the filename.
    
    Returns:
        dict: Parsed protocol metadata.
    """
    parts = protocol_str.split("_")

    if len(parts) < 2:
        return {"stimulus_type": protocol_str, "raw_protocol": protocol_str}

    stimulus_type = parts[0]
    stimulus_specifications = parts[1]
    
    # Extract timing details
    timing_match = re.match(r"(\d+[a-zA-Z]+)(\d+)x(\d+[a-zA-Z]+)(\d+[a-zA-Z]*)", parts[2]) if len(parts) > 2 else None

    if timing_match:
        prestimulus_duration = timing_match.group(1)
        repetitions = int(timing_match.group(2))
        stimulus_duration = timing_match.group(3)
        interval_between_repetitions = timing_match.group(4) if timing_match.group(4) else "0s"
    else:
        prestimulus_duration, repetitions, stimulus_duration, interval_between_repetitions = None, None, None, None

    return {
        "stimulus_type": stimulus_type,
        "stimulus_specifications": stimulus_specifications,
        "prestimulus_duration": prestimulus_duration,
        "number_of_repetitions": repetitions,
        "stimulus_duration": stimulus_duration,
        "interval_between_repetitions": interval_between_repetitions,
        "raw_protocol": protocol_str  # Keep original for reference
    }

def parse_filename(file_name):
    """Parses the filename to extract metadata including experiment details and larva number.
    
    Args:
        file_name (str): The name of the .dat file.
    
    Returns:
        dict: Metadata containing date, genotype, effector, tracker, protocol details, and larva number.
    """
    parts = file_name.split("@")
    
    if len(parts) < 6:
        raise ValueError(f"Unexpected filename format: {file_name}")
    
    metadata = {
        "date": parts[0],         # Date of experiment
        "genotype": parts[1],     # Genotype
        "effector": parts[2],     # Effector
        "tracker": parts[3],      # Tracker system used
    }

    # Parse protocol section
    protocol_metadata = parse_protocol(parts[4])
    metadata.update(protocol_metadata)

    # Extract larva number (last part after '@', before ".dat")
    larva_number = os.path.splitext(parts[-1])[0]  

    return metadata, larva_number

def extract_larva_data(file_path, columns=["time", "speed", "length", "curvature"]):
    """Extracts time, speed, length, and curvature from a single .dat file.
    
    Args:
        file_path (str): Path to the .dat file.
    
    Returns:
        dict: A dictionary with keys and corresponding values.
    """
    df = pd.read_csv(file_path, sep=r"\s+", names=columns, header=None)
    return df

def get_all_larva_ids(data_folder):
    """Extract all larva IDs from the data files in a folder.
    
    Args:
        data_folder (str): Path to the folder containing .dat files
        
    Returns:
        list: Sorted list of all larva IDs
    """
    larva_ids = []
    
    for file in os.listdir(data_folder):
        if file.endswith(".dat"):
            _, larva_number = parse_filename(file)
            larva_ids.append(larva_number)
    
    return sorted(larva_ids)

def compute_summary(df, columns):
    """Computes summary statistics for each variable in the larva data.
    
    Args:
        df (pd.DataFrame): DataFrame containing time, speed, length, curvature.
    
    Returns:
        dict: Summary statistics including size, mean, max, and min.
    """
    summary = {}
    for column in df.columns:
        summary[column] = {
            "size": len(df[column]),
            "mean": np.nanmean(df[column]),  # Handle potential NaNs
            "max": np.nanmax(df[column]),
            "min": np.nanmin(df[column])
        }
    return summary

def extract_all_larvae(data_folder, columns=["time", "speed", "length", "curvature"]):
    """Extracts larva data from all .dat files in a given folder.
    
    Args:
        data_folder (str): Path to the folder containing .dat files.
    
    Returns:
        dict: A dictionary where keys are larva numbers and values contain metadata, data, and summary.
    """
    larvae_data = {}
    
    for file in os.listdir(data_folder):
        if file.endswith(".dat"):
            file_path = os.path.join(data_folder, file)
            metadata, larva_number = parse_filename(file)
            df = extract_larva_data(file_path, columns)
            summary = compute_summary(df, columns)
            
            larvae_data[larva_number] = {
                "metadata": metadata,
                "data": df.to_dict(orient="list"),  # Store data as lists
                "summary": summary
            }
    
    return larvae_data


def array_with_nan(lst):
    return np.array([np.nan if x is None else x for x in lst])

def compute_v_and_axis(larvae_data):
    for larva_id, larva in larvae_data.items():
        speed = larva["data"]["speed"]
        vel_x = larva["data"]["vel_x"]
        vel_y = larva["data"]["vel_y"]

        # Normalize speed, vel_x, and vel_y by the mean speed
        mean_speed = np.nanmean(array_with_nan(speed))
        larva["data"]["speed_normalized"] = array_with_nan(speed) / mean_speed
        larva["data"]["vel_x_normalized"] = array_with_nan(vel_x) / mean_speed
        larva["data"]["vel_y_normalized"] = array_with_nan(vel_y) / mean_speed

    return larvae_data

def NI_ind(data_ind, ax="x"): 
    """Function that computes the navigational index (normalized or not) for 1 larva

    Args:
        data_ind (pd.DataFrame): Data for 1 larva
        ax (str, optional): Axis along which we compute the NI. Defaults to "x".
    
    Returns:
        dict: The values of the normalized and not normalized navigational index
    """
    # Ensure data_ind is a DataFrame
    if isinstance(data_ind, dict):
        data_ind = pd.DataFrame(data_ind)
    
    # Retrieve the interesting data 
    axis = "vel_x" if ax == "x" else "vel_y"
    vel_axis = data_ind[axis]
    speed = data_ind["speed"]
    midline = data_ind["midline"].mean()  # Mean of the midline
    
    # Normalize
    norm_vel = vel_axis / midline
    norm_speed = speed / midline
    
    # Means
    mean_vel = vel_axis.mean()
    mean_vel_norm = norm_vel.mean()
    mean_speed = speed.mean()
    mean_speed_norm = norm_speed.mean()
    
    # Navigational Index
    NI_time_series = vel_axis / speed
    NI_mean = mean_vel / mean_speed
    NI_norm = mean_vel_norm / mean_speed_norm
    
    ni_df = pd.DataFrame({
        "time": data_ind["time"],
        "NI": NI_time_series,
        "NI_mean": NI_mean,
        "NI_norm": NI_norm
    })
    
    return ni_df

def compute_navigational_index(larvae_data, ax="x"):
    """Function that computes the navigational index for all larvae
    
    Args:
        larvae_data (dict): Dictionary containing data for all larvae
        ax (str, optional): Axis along which we compute the NI. Defaults to "x".
    
    Returns:
        dict: Dictionary containing the navigational index for each larva
    """
    ni_dict = {}
    for larva_id, larva in larvae_data.items():
        ni_dict[larva_id] = NI_ind(larva["data"], ax)
    
    return ni_dict

def determine_pca_components(data, variance_threshold=0.95):
    """Determines the optimal number of PCA components based on explained variance."""
    pca = PCA()
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Create single plot
    plt.figure(figsize=(6, 4))
    plt.plot(cumulative_variance, marker='o', markersize=4)
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Components Analysis')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    return optimal_components

def determine_kmeans_clusters(data, max_clusters=10):
    """Determines the optimal number of K-means clusters."""
    inertia = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, cluster_labels))
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Elbow plot
    ax1.plot(cluster_range, inertia, marker='o', markersize=4)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(False)
    
    # Silhouette plot
    ax2.plot(cluster_range, silhouette_scores, marker='o', markersize=4)
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    return optimal_clusters

def cluster_behaviors(larva_data, variance_threshold=0.95, max_clusters=10):
    """Clusters the time points into different behaviors using PCA and K-means clustering.
    
    Args:
        larva_data (dict): Dictionary containing the variables for clustering.
        variance_threshold (float, optional): Threshold for cumulative explained variance for PCA. Defaults to 0.95.
        max_clusters (int, optional): Maximum number of clusters to consider for K-means. Defaults to 10.
    
    Returns:
        pd.DataFrame, int, int: DataFrame with the original data, PCA components, and cluster labels, 
                                the optimal number of PCA components, and the optimal number of clusters.
    """
    # Extract relevant variables
    df = pd.DataFrame(larva_data)
    variables = ["time", "persistence", "speed", "midline", "loc_x", "loc_y", "vel_x", "vel_y", "orient", "pathlen"]
    data = df[variables].dropna()  # Drop rows with missing values

    # Standardize the data
    data_standardized = (data - data.mean()) / data.std()

    # Determine optimal number of PCA components
    optimal_components = determine_pca_components(data_standardized, variance_threshold)

    # Apply PCA
    pca = PCA(n_components=optimal_components)
    pca_result = pca.fit_transform(data_standardized)
    data_pca = pd.DataFrame(pca_result, columns=[f'pca_{i+1}' for i in range(optimal_components)])

    # Determine optimal number of K-means clusters
    optimal_clusters = determine_kmeans_clusters(data_pca, max_clusters)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data_pca)

    # Merge PCA components back into the original data
    data = pd.concat([data, data_pca], axis=1)

    return data, optimal_components, optimal_clusters


# Script to import the data
import json
import h5py
import scipy.io


def import_trx_sharbat(path):
    """Function that imports a .mat file from the path and formats it into a dictionary of dictionaries."""
    field_rmv = [
        'As_smooth_5', 'back', 'back_large', 'back_strong', 'back_weak', 'ball_proba', 'bend_proba', 'cast',
        'cast_large', 'cast_strong', 'cast_weak', 'curl_proba', 'd_eff_head_norm_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
        'd_eff_tail_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5', 'duration_large', 'duration_large_small',
        'eig_deriv_smooth_5', 'eig_smooth_5', 'full_path', 'global_state', 'hunch', 'hunch_large', 'hunch_strong',
        'hunch_weak', 'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail_head_smooth_5', 'motion_velocity_norm_smooth_5',
        'n_duration', 'n_duration_large', 'n_duration_large_small', 'nb_action', 'nb_action_large', 'nb_action_large_small',
        'neuron', 'numero_larva', 'pipeline', 'proba_global_state', 'prod_scal_1', 'prod_scal_2', 'protocol', 'roll',
        'roll_large', 'roll_strong', 'roll_weak', 'run', 'run_large', 'run_strong', 'run_weak', 'small_motion', 'start_stop',
        'start_stop_large', 'start_stop_large_small', 'stimuli', 'stop', 'stop_large', 'stop_strong', 'stop_weak',
        'straight_and_light_bend_proba', 'straight_proba', 't_start_stop', 't_start_stop_large', 't_start_stop_large_small',
        'tail_velocity_norm_smooth_5', 'x_contour', 'x_neck', 'x_neck_down', 'x_neck_top', 'x_tail', 'y_contour', 'y_neck',
        'y_neck_down', 'y_neck_top', 'y_tail'
    ]

    mat_data = scipy.io.loadmat(path)
    trx_data = mat_data['trx']

    # Extract fields
    fields = [field for field in trx_data.dtype.names if field not in field_rmv]

    trx_extracted = {}  # This will be the dictionary that contains all the larva values

    # Iterate over each larva
    for i in range(trx_data.shape[1]):
        larva = {}
        for field in fields:
            data = trx_data[field][0, i]
            if isinstance(data, np.ndarray):
                data = data.flatten().tolist()
            larva[field] = data
        larva["nb_timestep"] = len(larva["t"])
        larva_id = larva.pop("numero_larva_num")
        trx_extracted[larva_id] = larva

    trx = {"data": trx_extracted}
    return trx


def import_trx(path):
    """Function that imports a trx file from the path and format it into a dictionnary of dictionaries
    Try to read the trx file
    """
    field_rmv = ['As_smooth_5',
#'S',
#'S_deriv_smooth_5',
#'S_smooth_5',
#'angle_downer_upper_deriv_smooth_5',
#'angle_downer_upper_smooth_5',
#'angle_upper_lower_deriv_smooth_5',
#'angle_upper_lower_smooth_5',
'back',
'back_large',
'back_strong',
'back_weak',
'ball_proba',
'bend_proba',
'cast',
'cast_large',
'cast_strong',
'cast_weak',
'curl_proba',
'd_eff_head_norm_deriv_smooth_5',
'd_eff_head_norm_smooth_5',
'd_eff_tail_norm_deriv_smooth_5',
'd_eff_tail_norm_smooth_5',
'duration_large',
'duration_large_small',
'eig_deriv_smooth_5',
'eig_smooth_5',
'full_path',
'global_state',
#'global_state_large_state',
#'global_state_small_large_state',
#'head_velocity_norm_smooth_5',
'hunch',
'hunch_large',
'hunch_strong',
'hunch_weak',
#'id',
#'larva_length_deriv_smooth_5',
#'larva_length_smooth_5',
'motion_to_u_tail_head_smooth_5',
'motion_to_v_tail_head_smooth_5',
'motion_velocity_norm_smooth_5',
'n_duration',
'n_duration_large',
'n_duration_large_small',
'nb_action',
'nb_action_large',
'nb_action_large_small',
'neuron',
'numero_larva',
#'numero_larva_num',
'pipeline',
'proba_global_state',
'prod_scal_1',
'prod_scal_2',
'protocol',
'roll',
'roll_large',
'roll_strong',
'roll_weak',
'run',
'run_large',
'run_strong',
'run_weak',
'small_motion',
'start_stop',
'start_stop_large',
'start_stop_large_small',
'stimuli',
'stop',
'stop_large',
'stop_strong',
'stop_weak',
'straight_and_light_bend_proba',
'straight_proba',
#'t',
't_start_stop',
't_start_stop_large',
't_start_stop_large_small',
'tail_velocity_norm_smooth_5',
#'x_center',
'x_contour',
#'x_head',
'x_neck',
'x_neck_down',
'x_neck_top',
#'x_spine',
'x_tail',
#'y_center',
'y_contour',
#'y_head',
'y_neck',
'y_neck_down',
'y_neck_top',
#'y_spine',
 'y_tail']

    with h5py.File(path,'r') as f:
        fields = list(f['trx'].keys())
        for rmv in field_rmv :
            fields.remove(rmv)
        trx_extracted = dict() #This will be the list the contains all the larva value -> For each larva it will contains all the fields

        #We take the shape of the actual trx file
        nb_larvae = f['trx'][fields[0]].shape[1]
        print(nb_larvae)
        for i in range(nb_larvae) :
            larva = dict()
            for field in fields :
                ref = f['trx'][field][0][i]
                data = np.array(f[ref])
                data = data.tolist()
                data = data[0] if len(data)==1 else data
                larva[field]=data
            larva["nb_timestep"]=len(data[0])
            print(larva["id"])
            _id = larva.pop("numero_larva_num")
            _id=set(_id)
            print(_id)
            #trx_extracted[] = larva
        f.close()
    trx = dict()
    trx["data"] = trx_extracted
    return trx


###All the fields of the trx file are :
# ['As_smooth_5',
# 'S',
# 'S_deriv_smooth_5',
# 'S_smooth_5',
# 'angle_downer_upper_deriv_smooth_5',
# 'angle_downer_upper_smooth_5',
# 'angle_upper_lower_deriv_smooth_5',
# 'angle_upper_lower_smooth_5',
# 'back',
# 'back_large',
# 'back_strong',
# 'back_weak',
# 'ball_proba',
# 'bend_proba',
# 'cast',
# 'cast_large',
# 'cast_strong',
# 'cast_weak',
# 'curl_proba',
# 'd_eff_head_norm_deriv_smooth_5',
# 'd_eff_head_norm_smooth_5',
# 'd_eff_tail_norm_deriv_smooth_5',
# 'd_eff_tail_norm_smooth_5',
# 'duration_large',
# 'duration_large_small',
# 'eig_deriv_smooth_5',
# 'eig_smooth_5',
# 'full_path',
# 'global_state',
# 'global_state_large_state',
# 'global_state_small_large_state',
# 'head_velocity_norm_smooth_5',
# 'hunch',
# 'hunch_large',
# 'hunch_strong',
# 'hunch_weak',
# 'id',
# 'larva_length_deriv_smooth_5',
# 'larva_length_smooth_5',
# 'motion_to_u_tail_head_smooth_5',
# 'motion_to_v_tail_head_smooth_5',
# 'motion_velocity_norm_smooth_5',
# 'n_duration',
# 'n_duration_large',
# 'n_duration_large_small',
# 'nb_action',
# 'nb_action_large',
# 'nb_action_large_small',
# 'neuron',
# 'numero_larva',
# 'numero_larva_num',
# 'pipeline',
# 'proba_global_state',
# 'prod_scal_1',
# 'prod_scal_2',
# 'protocol',
# 'roll',
# 'roll_large',
# 'roll_strong',
# 'roll_weak',
# 'run',
# 'run_large',
# 'run_strong',
# 'run_weak',
# 'small_motion',
# 'start_stop',
# 'start_stop_large',
# 'start_stop_large_small',
# 'stimuli',
# 'stop',
# 'stop_large',
# 'stop_strong',
# 'stop_weak',
# 'straight_and_light_bend_proba',
# 'straight_proba',
# 't',
# 't_start_stop',
# 't_start_stop_large',
# 't_start_stop_large_small',
# 'tail_velocity_norm_smooth_5',
# 'x_center',
# 'x_contour',
# 'x_head',
# 'x_neck',
# 'x_neck_down',
# 'x_neck_top',
# 'x_spine',
# 'x_tail',
# 'y_center',
# 'y_contour',
# 'y_head',
# 'y_neck',
# 'y_neck_down',
# 'y_neck_top',
# 'y_spine',
#  'y_tail']