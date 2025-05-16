import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm
import multiprocessing as mp

def get_behavior_data(f, field, i):
    """Extract behavior-related cell arrays from MATLAB struct.
    
    Args:
        f: HDF5 file object
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
            
        # Get the 1xN cell array (either 1x7 or 1x12)
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
                                 'global_state_small_large_state']:
                        if field in f['trx']:
                            larva[field] = get_array(field)
                    
                    # Get body part positions
                    for point in ['center', 'neck', 'head', 'tail', 'neck_down', 'neck_top']:
                        if f'x_{point}' in f['trx']:
                            larva[f'x_{point}'] = get_array(f'x_{point}')
                            larva[f'y_{point}'] = get_array(f'y_{point}')
                    
                    # Get behavior metrics
                    for field_prefix in ['duration', 't_start_stop', 'start_stop', 'nb_action']:
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

def process_all_trx_files_parallel(base_path, n_processes=None):
    """Process all trx.mat files in a directory tree using parallel processing."""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    # Find all trx files
    file_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'trx.mat':
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    
    print(f"Found {len(file_list)} trx.mat files")
    print(f"Using {n_processes} processes")
    
    # Initialize results
    all_data = {}
    metadata = {
        'files_processed': [],
        'total_larvae': 0,
        'experiments': {}
    }
    
    # Process files in parallel
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, file_list),
            total=len(file_list),
            desc="Processing files"
        ))
    
    # Combine results
    for result in results:
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

def filter_larvae_by_duration(data, min_total_duration=None, percentile=10):
    """Filter larvae based on their total tracked duration.
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        min_total_duration: Minimum total duration in seconds. If None, uses percentile
        percentile: Percentile threshold (0-100) to use if min_total_duration is None
        
    Returns:
        dict: Filtered data with same structure as input, excluding larvae below threshold
    """
    # Handle data type
    if 'data' in data:
        extracted_data = data['data']
        is_multi_exp = True
    else:
        extracted_data = data
        is_multi_exp = False
    
    # Calculate total duration for each larva
    larva_durations = {}
    for larva_id, larva_data in extracted_data.items():
        total = 0
        for durations in larva_data['duration_large_small']:
            if durations is not None:
                total += float(np.nansum(durations.flatten()))
        larva_durations[larva_id] = total
    
    # Determine threshold
    if min_total_duration is None:
        min_total_duration = np.percentile(list(larva_durations.values()), percentile)
    
    # Filter larvae
    filtered_data = {}
    for larva_id, total_duration in larva_durations.items():
        if total_duration >= min_total_duration:
            filtered_data[larva_id] = extracted_data[larva_id]
    
    # Return with appropriate structure
    if is_multi_exp:
        return {
            'data': filtered_data,
            'metadata': {
                **data['metadata'],
                'total_larvae': len(filtered_data),
                'duration_threshold': min_total_duration,
                'original_total_larvae': len(extracted_data)
            }
        }
    else:
        return filtered_data