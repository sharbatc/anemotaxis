import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm
import multiprocessing as mp

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
    """
    Load analysis results from an HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
        
    Returns:
        Dictionary containing all the analysis results
    """
    def load_group_recursively(group):
        """Recursively load data from HDF5 group."""
        result = {}
        
        # Load attributes
        for attr_name, attr_value in group.attrs.items():
            result[attr_name] = attr_value
        
        # Load datasets and subgroups
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                result[key] = np.array(item)
            elif isinstance(item, h5py.Group):
                # Check if this is a special structured group
                entry_type = item.attrs.get('entry_type', None)
                
                if entry_type == 'dict_list':
                    # Reconstruct list of dictionaries
                    n_entries = item.attrs['n_entries']
                    result[key] = []
                    for i in range(n_entries):
                        entry_group = item[f'entry_{i}']
                        result[key].append(load_group_recursively(entry_group))
                        
                elif entry_type == 'array_list':
                    # Reconstruct list of arrays
                    n_arrays = item.attrs['n_arrays']
                    result[key] = []
                    for i in range(n_arrays):
                        if f'array_{i}' in item:
                            result[key].append(np.array(item[f'array_{i}']))
                        
                elif entry_type in ['object_array', 'mixed_data']:
                    # Reconstruct object array or mixed data
                    n_arrays = item.attrs['n_arrays']
                    result[key] = []
                    for i in range(n_arrays):
                        array_key = f'array_{i}'
                        dict_key = f'dict_{i}'
                        if array_key in item:
                            result[key].append(np.array(item[array_key]))
                        elif dict_key in item:
                            result[key].append(load_group_recursively(item[dict_key]))
                else:
                    # Regular nested group
                    result[key] = load_group_recursively(item)
        
        return result
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata = {}
        for attr_name, attr_value in f.attrs.items():
            metadata[attr_name] = attr_value
        
        # Load all analysis results
        results = {}
        for result_name, group in f.items():
            print(f"📖 Loading {result_name}...")
            results[result_name] = load_group_recursively(group)
        
        results['_metadata'] = metadata
        
    return results