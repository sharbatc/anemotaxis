import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm
import multiprocessing as mp
import glob
import scipy.stats as stats

def get_behavior_data(f, field, i):
    """Extract behavior-related cell arrays from MATLAB struct.
    
    Args:
        f: HDF5 file object (.mat file in our case for trx data)
        field: Field name (e.g., 'duration_large', 't_start_stop_large', 'duration_large_small')
        i: Larva index
        
    Returns:
        list: List of arrays, one for each behavior type
    """
    try:
        # Get the reference to the cell array
        cell_ref = f['trx'][field][0][i]
        if not isinstance(cell_ref, h5py.h5r.Reference):
            return None
            
        # Get the 1xN cell array (usually, either 1x7 or 1x12)
        cell_array = f[cell_ref]
        
        # Determine if this is a large_small field (has 12 elements) or regular field (7 elements)
        num_elements = cell_array.shape[0]
        
        # Initialize list to store arrays for each behavior
        behavior_data = []
        
        # Extract each behavior's data
        for j in range(num_elements):
            try:
                behavior_ref = cell_array[j,0]
                if isinstance(behavior_ref, h5py.h5r.Reference):
                    behavior_array = f[behavior_ref]
                    behavior_data.append(np.array(behavior_array))
                else:
                    behavior_data.append(np.array([]))
            except Exception as e:
                tqdm.write(f"Error getting behavior {j+1} data: {str(e)}")
                behavior_data.append(np.array([]))
        
        return behavior_data
        
    except Exception as e:
        tqdm.write(f"Error getting behavior data for field {field}: {str(e)}")
        return None

def load_single_trx_file(file_path, show_progress=False):
    """Load and extract data from a single trx.mat file.
    
    Args:
        file_path (str): Path to the trx.mat file
        show_progress (bool): Whether to show progress messages
        
    Returns:
        dict: Dictionary containing:
            - 'date_str': Experiment date string
            - 'data': Dict of larva_id -> larva_data
            - 'metadata': File metadata
    """
    # Define data fields to extract
    TIME_SERIES_FIELDS = [
        't', 'x_spine', 'y_spine', 'x_contour', 'y_contour',
        'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5',
        'motion_velocity_norm_smooth_5', 'angle_upper_lower_smooth_5',
        'angle_downer_upper_smooth_5', 'global_state_large_state',
        'global_state_small_large_state', 'proba_global_state',
        'larva_length_smooth_5'
    ]
    
    BODY_PARTS = ['center', 'neck', 'head', 'tail', 'neck_down', 'neck_top']
    
    BEHAVIOR_FIELDS = {
        'duration': ['large', 'large_small'],
        't_start_stop': ['large', 'large_small'],
        'start_stop': ['large', 'large_small'],
        'nb_action': ['large', 'large_small'],
        'n_duration': ['large', 'large_small']
    }
    
    REQUIRED_BEHAVIOR_FIELDS = ['duration_large_small', 't_start_stop_large_small', 'nb_action_large_small']
    
    def _extract_array_data(f, field_name, larva_idx):
        """Safely extract array data from HDF5 reference."""
        try:
            ref = f['trx'][field_name][0][larva_idx]
            if isinstance(ref, h5py.Dataset):
                return np.array(ref)
            return np.array(f[ref])
        except Exception as e:
            if show_progress:
                tqdm.write(f"Warning: Could not extract {field_name} for larva {larva_idx}: {e}")
            return None
    
    def _extract_larva_id(f, larva_idx):
        """Extract larva ID number."""
        try:
            larva_id_ref = f['trx']['numero_larva_num'][0][larva_idx]
            if isinstance(larva_id_ref, h5py.Dataset):
                return int(np.array(larva_id_ref))
            return int(np.array(f[larva_id_ref]))
        except Exception as e:
            if show_progress:
                tqdm.write(f"Warning: Could not extract larva ID for larva {larva_idx}: {e}")
            return larva_idx  # Fallback to index
    
    def _extract_larva_data(f, larva_idx):
        """Extract all data for a single larva."""
        larva_data = {}
        
        # Extract time series data
        for field in TIME_SERIES_FIELDS:
            if field in f['trx']:
                larva_data[field] = _extract_array_data(f, field, larva_idx)
        
        # Extract body part coordinates
        for part in BODY_PARTS:
            x_field, y_field = f'x_{part}', f'y_{part}'
            if x_field in f['trx'] and y_field in f['trx']:
                larva_data[x_field] = _extract_array_data(f, x_field, larva_idx)
                larva_data[y_field] = _extract_array_data(f, y_field, larva_idx)
        
        # Extract behavior data
        for prefix, suffixes in BEHAVIOR_FIELDS.items():
            for suffix in suffixes:
                field_name = f'{prefix}_{suffix}'
                if field_name in f['trx']:
                    larva_data[field_name] = get_behavior_data(f, field_name, larva_idx)
        
        # Extract larva ID
        larva_data['larva_id'] = _extract_larva_id(f, larva_idx)
        
        return larva_data
    
    def _is_valid_larva(larva_data):
        """Check if larva has required behavior data."""
        return all(
            larva_data.get(field) is not None 
            for field in REQUIRED_BEHAVIOR_FIELDS
        )
    
    def _get_date_from_path(file_path):
        """Extract date string from file path."""
        return os.path.basename(os.path.dirname(file_path))
    
    # Main processing logic
    try:
        with h5py.File(file_path, 'r') as f:
            # Get basic file info
            fields = list(f['trx'].keys())
            n_larvae = f['trx'][fields[0]].shape[1]
            
            if show_progress:
                print(f"\nProcessing file: {file_path}")
                print(f"Number of larvae: {n_larvae}")
            
            # Extract data for each larva
            extracted_data = {}
            progress_iter = tqdm(range(n_larvae), desc="Processing larvae", disable=not show_progress)
            
            for larva_idx in progress_iter:
                try:
                    larva_data = _extract_larva_data(f, larva_idx)
                    
                    # Only include larvae with valid behavior data
                    if _is_valid_larva(larva_data):
                        larva_id = larva_data['larva_id']
                        extracted_data[larva_id] = larva_data
                    
                except Exception as e:
                    if show_progress:
                        tqdm.write(f"Error processing larva {larva_idx}: {e}")
                    continue
            
            # Prepare metadata
            date_str = _get_date_from_path(file_path)
            metadata = {
                'file_path': file_path,
                'date_str': date_str,
                'n_larvae_total': n_larvae,
                'n_larvae_valid': len(extracted_data),
                'date': datetime.strptime(date_str.split('_')[0], '%Y%m%d')
            }
            
            return {
                'date_str': date_str,
                'data': extracted_data,
                'metadata': metadata
            }
            
    except Exception as e:
        error_msg = f"Error loading {file_path}: {e}"
        if show_progress:
            print(error_msg)
        return {
            'date_str': None,
            'data': {},
            'metadata': {'error': error_msg, 'file_path': file_path}
        }

def process_single_file(file_path, show_progress=False):
    """Process a single trx.mat file containing larval tracking data.
    
    Args:
        file_path: Path to the trx.mat file
        show_progress: Whether to show progress messages
        
    Returns:
        tuple: (date_str, extracted_data, metadata)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            fields = list(f['trx'].keys())
            nb_larvae = f['trx'][fields[0]].shape[1]
            
            if show_progress:
                print(f"\nProcessing file: {file_path}")
                print(f"Number of larvae: {nb_larvae}")
            
            extracted_data = {}
            for i in tqdm(range(nb_larvae), desc="Processing larvae", disable=not show_progress):
                larva = {}
                try:
                    # Helper function to safely extract array data
                    def get_array(field):
                        ref = f['trx'][field][0][i]
                        if isinstance(ref, h5py.Dataset):
                            return np.array(ref)
                        return np.array(f[ref])
                    
                    # Extract data fields (time series, coordinates, behavior, etc.)
                    for field in ['t', 'x_spine', 'y_spine', 'x_contour', 'y_contour',
                                 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5',
                                 'motion_velocity_norm_smooth_5', 'angle_upper_lower_smooth_5',
                                 'angle_downer_upper_smooth_5', 'global_state_large_state',
                                 'global_state_small_large_state','proba_global_state','larva_length_smooth_5']:
                        if field in f['trx']:
                            larva[field] = get_array(field)
                    
                    # Get body part positions
                    for point in ['center', 'neck', 'head', 'tail', 'neck_down', 'neck_top']:
                        if f'x_{point}' in f['trx']:
                            larva[f'x_{point}'] = get_array(f'x_{point}')
                            larva[f'y_{point}'] = get_array(f'y_{point}')
                    
                    # Get behavior metrics
                    for field_prefix in ['n_duration','duration', 't_start_stop', 'start_stop', 'nb_action']:
                        for field_suffix in ['large', 'large_small']:
                            field = f'{field_prefix}_{field_suffix}'
                            if field in f['trx']:
                                larva[field] = get_behavior_data(f, field, i)
                    
                    # Get larva ID
                    larva_id_ref = f['trx']['numero_larva_num'][0][i]
                    if isinstance(larva_id_ref, h5py.Dataset):
                        larva_id = int(np.array(larva_id_ref))
                    else:
                        larva_id = int(np.array(f[larva_id_ref]))
                    
                    # Only add larva if we have valid behavior data
                    if all(x is not None for x in [
                        larva.get('duration_large', None),
                        larva.get('t_start_stop_large', None),
                        larva.get('nb_action_large', None)
                    ]):
                        extracted_data[larva_id] = larva
                    
                except Exception as e:
                    tqdm.write(f"Error extracting data for larva {i}: {str(e)}")
                    continue
            
            date_str = os.path.basename(os.path.dirname(file_path))
            
            return date_str, extracted_data, {
                'path': file_path,
                'date': datetime.strptime(date_str.split('_')[0], '%Y%m%d'),
                'n_larvae': len(extracted_data),
                'date_str': date_str
            }
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_all_trx_files(base_path):
    """Process all trx.mat files in a directory tree sequentially with progress bar."""
    # Find all trx files
    file_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'trx.mat':
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    
    print(f"Found {len(file_list)} trx.mat files")
    
    # Initialize results
    all_data = {}
    metadata = {
        'files_processed': [],
        'total_larvae': 0,
        'experiments': {}
    }
    
    # Process files sequentially with progress bar
    for file_path in tqdm(file_list, desc="Processing trx files"):
        result = process_single_file(file_path)
        if result is not None:
            date_str, trx_extracted, exp_info = result
            metadata['experiments'][date_str] = exp_info
            metadata['files_processed'].append(exp_info['path'])
            metadata['total_larvae'] += exp_info['n_larvae']
            
            for larva_id, larva_data in trx_extracted.items():
                unique_id = f"{date_str}_{larva_id}"
                larva_data['experiment_date'] = date_str
                all_data[unique_id] = larva_data
    
    print(f"\nProcessed {len(metadata['files_processed'])} files")
    print(f"Total larvae: {metadata['total_larvae']}")
    
    return {'data': all_data, 'metadata': metadata}

def save_analysis_results(output_dir, single_path, trx_filtered_by_merging, **results):
    """
    Save all analysis results to an HDF5 file for later loading and combining.
    
    Args:
        output_dir: Directory to save the file
        single_path: Path to the original trx.mat file
        trx_filtered_by_merging: Filtered data
        **results: All analysis results as keyword arguments
    """
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_results_{timestamp}.h5"
    filepath = os.path.join(output_dir, filename)
    
    def save_dict_recursively(group, data_dict, level=0):
        """Recursively save a dictionary to HDF5 group, handling nested structures."""
        for key, value in data_dict.items():
            # Convert key to string to avoid HDF5 naming issues
            key_str = str(key)
            
            if value is None:
                continue
                
            # Handle nested dictionaries
            if isinstance(value, dict):
                subgroup = group.create_group(key_str)
                save_dict_recursively(subgroup, value, level + 1)
                continue
            
            # Handle lists and arrays
            if isinstance(value, (list, np.ndarray)):
                try:
                    # Try to convert to numpy array
                    if isinstance(value, list):
                        # Check if it's a list of dictionaries (like cast_events_data)
                        if len(value) > 0 and isinstance(value[0], dict):
                            # Save as a group with numbered sub-entries
                            subgroup = group.create_group(key_str)
                            subgroup.attrs['n_entries'] = len(value)
                            subgroup.attrs['entry_type'] = 'dict_list'
                            for i, entry_dict in enumerate(value):
                                entry_group = subgroup.create_group(f'entry_{i}')
                                save_dict_recursively(entry_group, entry_dict, level + 1)
                            continue
                        
                        # Check if it's a list of arrays with different shapes
                        if len(value) > 0 and hasattr(value[0], '__len__'):
                            lengths = [len(item) if hasattr(item, '__len__') else 1 for item in value]
                            if len(set(lengths)) > 1:  # Different lengths - inhomogeneous
                                subgroup = group.create_group(key_str)
                                subgroup.attrs['n_arrays'] = len(value)
                                subgroup.attrs['entry_type'] = 'array_list'
                                for i, arr in enumerate(value):
                                    if arr is not None:
                                        subgroup.create_dataset(f'array_{i}', data=np.array(arr))
                                continue
                    
                    # Try normal numpy conversion
                    value_array = np.array(value)
                    if value_array.dtype == 'object':
                        # Still object dtype - handle as separate arrays
                        subgroup = group.create_group(key_str)
                        subgroup.attrs['n_arrays'] = len(value)
                        subgroup.attrs['entry_type'] = 'object_array'
                        for i, arr in enumerate(value):
                            if arr is not None:
                                try:
                                    subgroup.create_dataset(f'array_{i}', data=np.array(arr))
                                except Exception as e2:
                                    print(f"❌ Could not save array {i} in {key_str}: {e2}")
                    else:
                        # Regular homogeneous array
                        group.create_dataset(key_str, data=value_array)
                        
                except (ValueError, TypeError) as e:
                    # Handle the inhomogeneous case
                    print(f"⚠️  Handling inhomogeneous data for {key_str}")
                    subgroup = group.create_group(key_str)
                    subgroup.attrs['n_arrays'] = len(value)
                    subgroup.attrs['entry_type'] = 'mixed_data'
                    for i, arr in enumerate(value):
                        if arr is not None:
                            try:
                                if isinstance(arr, dict):
                                    arr_group = subgroup.create_group(f'dict_{i}')
                                    save_dict_recursively(arr_group, arr, level + 1)
                                else:
                                    subgroup.create_dataset(f'array_{i}', data=np.array(arr))
                            except Exception as e2:
                                print(f"❌ Could not save item {i} in {key_str}: {e2}")
                                
            elif isinstance(value, (int, float, np.integer, np.floating)):
                group.attrs[key_str] = value
            elif isinstance(value, str):
                group.attrs[key_str] = value
            elif isinstance(value, bool):
                group.attrs[key_str] = value
            else:
                # Convert other types to string
                try:
                    group.attrs[key_str] = str(value)
                except Exception as e:
                    print(f"⚠️  Could not save {key_str}: {e}")
    
    with h5py.File(filepath, 'w') as f:
        # Add metadata
        f.attrs['created_date'] = datetime.now().isoformat()
        f.attrs['source_file'] = single_path
        f.attrs['n_larvae_total'] = len(trx_filtered_by_merging['data']) if isinstance(trx_filtered_by_merging, dict) and 'data' in trx_filtered_by_merging else len(trx_filtered_by_merging)
        
        # Save each analysis result
        for result_name, result_data in results.items():
            if result_data is None:
                continue
                
            print(f"💾 Saving {result_name}...")
            group = f.create_group(result_name)
            
            if isinstance(result_data, dict):
                save_dict_recursively(group, result_data)
            else:
                # Handle non-dict results
                try:
                    if isinstance(result_data, (list, np.ndarray)):
                        group.create_dataset('data', data=np.array(result_data))
                    else:
                        group.attrs['data'] = str(result_data)
                except Exception as e:
                    print(f"⚠️  Could not save {result_name}: {e}")
    
    print(f"📁 Analysis results saved to: {filepath}")
    return filepath

def load_analysis_results(filepath):
    """Load analysis results from HDF5 file."""
    results = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata = dict(f.attrs)
        results['metadata'] = metadata
        
        # Load each analysis result group
        for group_name in f.keys():
            group = f[group_name]
            result_dict = {}
            
            # Load attributes (scalars)
            for attr_name, attr_value in group.attrs.items():
                result_dict[attr_name] = attr_value
            
            # Load datasets (arrays)
            for dataset_name in group.keys():
                if isinstance(group[dataset_name], h5py.Group):
                    # Handle arrays of arrays (inhomogeneous data)
                    subgroup = group[dataset_name]  # FIXED: Remove typo
                    entry_type = subgroup.attrs.get('entry_type', None)
                    
                    if entry_type == 'dict_list':
                        # Reconstruct list of dictionaries
                        n_entries = subgroup.attrs['n_entries']
                        result_dict[dataset_name] = []
                        for i in range(n_entries):
                            if f'entry_{i}' in subgroup:
                                entry_group = subgroup[f'entry_{i}']
                                entry_data = {}
                                # Load all attributes and datasets from this entry
                                for attr_name, attr_value in entry_group.attrs.items():
                                    entry_data[attr_name] = attr_value
                                for sub_key, sub_item in entry_group.items():
                                    if isinstance(sub_item, h5py.Dataset):
                                        entry_data[sub_key] = np.array(sub_item)
                                result_dict[dataset_name].append(entry_data)
                    
                    elif entry_type in ['array_list', 'object_array', 'mixed_data']:
                        # Handle other special cases
                        n_arrays = subgroup.attrs.get('n_arrays', len(subgroup.keys()))
                        arrays = []
                        for i in range(n_arrays):
                            if f'array_{i}' in subgroup:
                                arrays.append(subgroup[f'array_{i}'][:])
                            else:
                                arrays.append(None)  # Handle missing arrays
                        result_dict[dataset_name] = arrays
                    
                    else:
                        # Regular nested group - recurse
                        nested_result = {}
                        for attr_name, attr_value in subgroup.attrs.items():
                            nested_result[attr_name] = attr_value
                        for sub_key, sub_item in subgroup.items():
                            if isinstance(sub_item, h5py.Dataset):
                                nested_result[sub_key] = np.array(sub_item)
                        result_dict[dataset_name] = nested_result
                else:
                    # Regular homogeneous array
                    result_dict[dataset_name] = group[dataset_name][:]
            
            results[group_name] = result_dict
    
    return results


def combine_analysis_results(result_files, analysis_type):
    """
    Combine multiple analysis results of the same type.
    
    Args:
        result_files: List of HDF5 file paths
        analysis_type: Type of analysis to combine (e.g., 'run_prob_results')
    
    Returns:
        Combined analysis results
    """
    all_hist_arrays = []
    all_metric_arrays = []
    all_ni_x_values = []  # For NI single x values
    all_ni_y_values = []  # For NI single y values
    all_ni_x_time_series = []  # For NI time series x
    all_ni_y_time_series = []  # For NI time series y
    all_larva_summaries = []  # For larva-level bias summaries
    
    # NEW: Store NI values by date for plotting
    ni_x_by_date = {}  # {date: [NI_x_values]}
    ni_y_by_date = {}  # {date: [NI_y_values]}
    
    bin_centers = None
    time_centers = None
    orientation_bins = None
    n_larvae_total = 0
    
    # Special handling for different analysis types
    is_bias_analysis = 'bias_results' in analysis_type
    is_ni_single = analysis_type == 'ni_single_results'
    is_ni_time = analysis_type == 'ni_time_results'
    is_head_cast = 'head_cast' in analysis_type and 'bias' not in analysis_type
    
    for filepath in result_files:
        try:
            results = load_analysis_results(filepath)
            
            if analysis_type not in results:
                continue
                
            data = results[analysis_type]
            n_larvae_total += data.get('n_larvae', 0)
            
            # Extract experiment date from filepath
            experiment_date = 'unknown'
            path_parts = filepath.split('/')
            for part in path_parts:
                if len(part) == 15 and part.startswith('202'):  # Format: 20240226_145653
                    experiment_date = part
                    break
            
            # Get common axes
            if bin_centers is None and 'bin_centers' in data:
                bin_centers = data['bin_centers']
            if time_centers is None and 'time_centers' in data:
                time_centers = data['time_centers']
            if orientation_bins is None and 'orientation_bins' in data:
                orientation_bins = data['orientation_bins']
            
            # Handle different data structures based on analysis type
            if is_bias_analysis:
                # Head cast bias data - collect larva summaries
                if 'larva_summaries' in data and data['larva_summaries'] is not None:
                    # Filter out None values and extend the list of larva summaries
                    valid_summaries = [s for s in data['larva_summaries'] if s is not None]
                    all_larva_summaries.extend(valid_summaries)
                            
            elif is_ni_single:
                # NI single values - collect individual NI values per larva AND by date
                if 'NI_x_clean' in data and len(data['NI_x_clean']) > 0:
                    ni_x_values = data['NI_x_clean']
                    all_ni_x_values.extend(ni_x_values)
                    # Store by date
                    if experiment_date not in ni_x_by_date:
                        ni_x_by_date[experiment_date] = []
                    ni_x_by_date[experiment_date].extend(ni_x_values)
                    
                if 'NI_y_clean' in data and len(data['NI_y_clean']) > 0:
                    ni_y_values = data['NI_y_clean']
                    all_ni_y_values.extend(ni_y_values)
                    # Store by date
                    if experiment_date not in ni_y_by_date:
                        ni_y_by_date[experiment_date] = []
                    ni_y_by_date[experiment_date].extend(ni_y_values)
                    
            elif is_ni_time:
                # NI time series - collect arrays for each larva
                if 'NI_x_arrays' in data and data['NI_x_arrays'] is not None:
                    ni_x_arrays = np.array(data['NI_x_arrays'])
                    if len(ni_x_arrays.shape) == 2:  # Ensure 2D (n_larvae, n_time_bins)
                        all_ni_x_time_series.append(ni_x_arrays)
                        
                if 'NI_y_arrays' in data and data['NI_y_arrays'] is not None:
                    ni_y_arrays = np.array(data['NI_y_arrays'])
                    if len(ni_y_arrays.shape) == 2:  # Ensure 2D (n_larvae, n_time_bins)
                        all_ni_y_time_series.append(ni_y_arrays)
                        
            else:
                # Standard histogram/metric data (run prob, turn prob, velocity, turn amp, head cast, backup)
                if 'hist_arrays' in data and len(data['hist_arrays']) > 0:
                    hist_arrays = np.array(data['hist_arrays'])
                    if len(hist_arrays.shape) == 2:
                        all_hist_arrays.append(hist_arrays)
                        
                if 'metric_arrays' in data and len(data['metric_arrays']) > 0:
                    metric_arrays = np.array(data['metric_arrays'])
                    if len(metric_arrays.shape) == 2:
                        all_metric_arrays.append(metric_arrays)
                        
        except Exception as e:
            print(f"⚠️  Error processing {filepath} for {analysis_type}: {e}")
            continue
    
    # Combine all arrays based on data type
    combined_result = {}
    
    # Standard histogram data
    if all_hist_arrays:
        combined_hist_arrays = np.concatenate(all_hist_arrays, axis=0)
        combined_result.update({
            'hist_arrays': combined_hist_arrays,
            'mean_hist': np.nanmean(combined_hist_arrays, axis=0),
            'se_hist': stats.sem(combined_hist_arrays, axis=0, nan_policy='omit'),
            'bin_centers': bin_centers,
            'n_larvae': len(combined_hist_arrays)
        })
    
    # Time series metric data
    if all_metric_arrays:
        combined_metric_arrays = np.concatenate(all_metric_arrays, axis=0)
        combined_result.update({
            'metric_arrays': combined_metric_arrays,
            'mean_metric': np.nanmean(combined_metric_arrays, axis=0),
            'se_metric': stats.sem(combined_metric_arrays, axis=0, nan_policy='omit'),
            'time_centers': time_centers,
            'n_larvae': len(combined_metric_arrays)
        })
    
    # Head cast bias analysis - Fixed version
    # Head cast bias analysis - CORRECTED version
    if all_larva_summaries:
        # Recalculate combined statistics from all larva summaries
        combined_towards_biases = []
        combined_away_biases = []
        
        for summary in all_larva_summaries:
            if summary is not None and isinstance(summary, dict):
                if 'towards_bias' in summary and summary['towards_bias'] is not None and not np.isnan(summary['towards_bias']):
                    combined_towards_biases.append(summary['towards_bias'])
                if 'away_bias' in summary and summary['away_bias'] is not None and not np.isnan(summary['away_bias']):
                    combined_away_biases.append(summary['away_bias'])

        # Calculate totals from larva summaries
        total_towards = sum(summary.get('towards_count', 0) for summary in all_larva_summaries
                        if summary is not None and isinstance(summary, dict))
        total_away = sum(summary.get('away_count', 0) for summary in all_larva_summaries
                    if summary is not None and isinstance(summary, dict))
        total_casts = total_towards + total_away
        overall_towards_bias = total_towards / total_casts if total_casts > 0 else np.nan
        overall_away_bias = total_away / total_casts if total_casts > 0 else np.nan
        
        # CORRECTED Statistical tests
        from scipy.stats import ttest_1samp, wilcoxon
        
        if len(combined_towards_biases) >= 3:  # Need at least 3 for meaningful test
            # Test 1: One-sample t-test against 0.5 (chance level)
            t_stat_ttest, p_value_ttest = ttest_1samp(combined_towards_biases, 0.5)
            
            # Test 2: Wilcoxon signed-rank test against 0.5 (non-parametric)
            deviations_from_chance = np.array(combined_towards_biases) - 0.5
            if np.any(deviations_from_chance != 0):
                w_stat, p_value_wilcoxon = wilcoxon(deviations_from_chance, alternative='two-sided')
            else:
                p_value_wilcoxon = 1.0
        else:
            p_value_ttest = np.nan
            p_value_wilcoxon = np.nan
        
        # Per-larva statistics
        mean_larva_towards_bias = np.nanmean(combined_towards_biases) if combined_towards_biases else np.nan
        se_larva_towards_bias = stats.sem(combined_towards_biases, nan_policy='omit') if combined_towards_biases else np.nan
        mean_larva_away_bias = np.nanmean(combined_away_biases) if combined_away_biases else np.nan
        se_larva_away_bias = stats.sem(combined_away_biases, nan_policy='omit') if combined_away_biases else np.nan
        
        combined_result.update({
            'larva_summaries': all_larva_summaries,
            'total_towards': total_towards,
            'total_away': total_away,
            'total_casts': total_casts,
            'overall_towards_bias': overall_towards_bias,
            'overall_away_bias': overall_away_bias,
            'mean_larva_towards_bias': mean_larva_towards_bias,
            'se_larva_towards_bias': se_larva_towards_bias,
            'mean_larva_away_bias': mean_larva_away_bias,
            'se_larva_away_bias': se_larva_away_bias,
            'p_value_wilcoxon': p_value_wilcoxon,
            'p_value_ttest': p_value_ttest,
            'n_larvae': len([s for s in all_larva_summaries if s is not None]),
            'analysis_type': analysis_type.split('_')[-1] if '_' in analysis_type else 'all'
        })
    
    # NI single values - ENHANCED with date preservation
    if all_ni_x_values or all_ni_y_values:
        combined_ni_x = np.array(all_ni_x_values) if all_ni_x_values else np.array([])
        combined_ni_y = np.array(all_ni_y_values) if all_ni_y_values else np.array([])
        
        # Remove NaN values
        combined_ni_x = combined_ni_x[~np.isnan(combined_ni_x)] if len(combined_ni_x) > 0 else combined_ni_x
        combined_ni_y = combined_ni_y[~np.isnan(combined_ni_y)] if len(combined_ni_y) > 0 else combined_ni_y
        
        # Clean NI values by date (remove NaNs)
        ni_x_by_date_clean = {}
        ni_y_by_date_clean = {}
        
        for date, values in ni_x_by_date.items():
            clean_values = np.array(values)
            clean_values = clean_values[~np.isnan(clean_values)]
            if len(clean_values) > 0:
                ni_x_by_date_clean[date] = clean_values
                
        for date, values in ni_y_by_date.items():
            clean_values = np.array(values)
            clean_values = clean_values[~np.isnan(clean_values)]
            if len(clean_values) > 0:
                ni_y_by_date_clean[date] = clean_values
        
        combined_result.update({
            'NI_x_clean': combined_ni_x,
            'NI_y_clean': combined_ni_y,
            'NI_x_by_date': ni_x_by_date_clean,  # NEW: NI_x values grouped by date
            'NI_y_by_date': ni_y_by_date_clean,  # NEW: NI_y values grouped by date
            'means': {
                'NI_x': np.nanmean(combined_ni_x) if len(combined_ni_x) > 0 else np.nan,
                'NI_y': np.nanmean(combined_ni_y) if len(combined_ni_y) > 0 else np.nan
            },
            'n_larvae': n_larvae_total
        })
        
        # Add significance testing
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
        
        p_x, sig_x = test_significance(combined_ni_x) if len(combined_ni_x) > 0 else (np.nan, "insufficient data")
        p_y, sig_y = test_significance(combined_ni_y) if len(combined_ni_y) > 0 else (np.nan, "insufficient data")
        
        combined_result.update({
            'p_values': {'NI_x': p_x, 'NI_y': p_y},
            'significances': {'NI_x': sig_x, 'NI_y': sig_y}
        })
    
    # NI time series data
    if all_ni_x_time_series or all_ni_y_time_series:
        if all_ni_x_time_series:
            combined_ni_x_time = np.concatenate(all_ni_x_time_series, axis=0)
            mean_ni_x_time = np.nanmean(combined_ni_x_time, axis=0)
            se_ni_x_time = stats.sem(combined_ni_x_time, axis=0, nan_policy='omit')
        else:
            combined_ni_x_time = np.array([])
            mean_ni_x_time = np.array([])
            se_ni_x_time = np.array([])
            
        if all_ni_y_time_series:
            combined_ni_y_time = np.concatenate(all_ni_y_time_series, axis=0)
            mean_ni_y_time = np.nanmean(combined_ni_y_time, axis=0)
            se_ni_y_time = stats.sem(combined_ni_y_time, axis=0, nan_policy='omit')
        else:
            combined_ni_y_time = np.array([])
            mean_ni_y_time = np.array([])
            se_ni_y_time = np.array([])
        
        combined_result.update({
            'NI_x_arrays': combined_ni_x_time,
            'NI_y_arrays': combined_ni_y_time,
            'mean_NI_x': mean_ni_x_time,
            'mean_NI_y': mean_ni_y_time,
            'se_NI_x': se_ni_x_time,
            'se_NI_y': se_ni_y_time,
            'time_centers': time_centers,
            'n_larvae': n_larvae_total
        })
    
    # Add common elements
    if bin_centers is not None:
        combined_result['bin_centers'] = bin_centers
    if time_centers is not None:
        combined_result['time_centers'] = time_centers
    if orientation_bins is not None:
        combined_result['orientation_bins'] = orientation_bins
    
    # Add total larvae count if not already added
    if 'n_larvae' not in combined_result:
        combined_result['n_larvae'] = n_larvae_total
    
    return combined_result if combined_result else None

def get_latest_analysis_files(experiment_dates, base_path=None):
    """
    Get the latest analysis .h5 files for given experiment dates.
    
    Args:
        experiment_dates: List of experiment dates (e.g., ['20240219_143334', '20240223_112627'])
        base_path: Base path to the data directory (optional)
    
    Returns:
        List of paths to the latest .h5 files for each date
    """
    if base_path is None:
        base_path = '/Users/sharbat/Projects/anemotaxis/data/FCF_attP2-40@UAS_TNT_2_0003/p_5gradient2_2s1x600s0s#n#n#n'
    
    result_files = []
    
    for date in experiment_dates:
        # Construct the analyses directory path
        analyses_dir = os.path.join(base_path, date, 'analyses')
        
        if not os.path.exists(analyses_dir):
            print(f"⚠️  Warning: Directory not found for {date}: {analyses_dir}")
            continue
        
        # Find all .h5 files with analysis_results pattern
        pattern = os.path.join(analyses_dir, 'analysis_results_*.h5')
        h5_files = glob.glob(pattern)
        
        if not h5_files:
            print(f"⚠️  Warning: No .h5 files found for {date}")
            continue
        
        # Sort by filename (timestamp is in filename) and get the latest
        h5_files.sort()
        latest_file = h5_files[-1]
        result_files.append(latest_file)
        
        # Extract timestamp from filename for confirmation
        filename = os.path.basename(latest_file)
        timestamp = filename.replace('analysis_results_', '').replace('.h5', '')
        print(f"✅ {date}: Using latest file with timestamp {timestamp}")
    
    return result_files

# Function to load and combine data for a single genotype
def load_genotype_data(genotype_key, genotype_config):
    """Load and combine analysis results for a single genotype."""
    print(f"🔄 Loading {genotype_config['label']} dataset...")
    
    result_files = get_latest_analysis_files(
        genotype_config['experiment_dates'], 
        base_path=genotype_config['base_path']
    )
    
    # Combine all analysis types
    combined_data = {}
    analysis_types = [
        'run_prob_results', 'run_prob_time_results',
        'turn_prob_results', 'turn_prob_time_results',
        'velocity_results', 'velocity_time_results',
        'backup_prob_results', 'backup_prob_time_results',
        'bias_results_first', 'bias_results_turn',
        'turn_amp_results', 'turn_amp_time_results',
        'head_cast_orientation_results', 'head_cast_time_results',
        'ni_single_results', 'ni_time_results'
    ]
    
    for analysis_type in analysis_types:
        combined_data[analysis_type] = combine_analysis_results(result_files, analysis_type)
    
    # Add metadata
    combined_data['metadata'] = {
        'genotype': genotype_config['name'],
        'label': genotype_config['label'],
        'n_files': len(result_files),
        'n_larvae': combined_data['run_prob_results']['n_larvae'] if combined_data['run_prob_results'] else 0,
        'style': {
            'linestyle': genotype_config['linestyle'],
            'se_alpha': genotype_config['alpha']
        }
    }
    
    return combined_data