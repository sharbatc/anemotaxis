import multiprocessing as mp
from functools import partial
import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm

def process_single_file(file_info):
    """Process a single trx file (multiprocessing-safe version)"""
    file_path, date_str = file_info
    
    try:
        with h5py.File(file_path, 'r') as f:
            fields = list(f['trx'].keys())
            trx_extracted = {}
            nb_larvae = f['trx'][fields[0]].shape[1]
            
            for i in range(nb_larvae):
                larva = {}
                for field in fields:
                    try:
                        ref = f['trx'][field][0][i]
                        data = np.array(f[ref]).flatten()
                        larva[field] = data.tolist() if len(data) > 1 else data[0]
                    except:
                        larva[field] = None
                
                larva_id = str(larva.pop("numero_larva_num", i))
                trx_extracted[larva_id] = larva
        
        return date_str, trx_extracted, {
            'path': file_path,
            'date': datetime.strptime(date_str.split('_')[0], '%Y%m%d'),
            'n_larvae': len(trx_extracted),
            'date_str': date_str
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_all_trx_files(base_path, n_processes=None):
    """Process all trx files using multiprocessing"""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    # Find all trx files
    file_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'trx.mat':
                file_path = os.path.join(root, file)
                date_str = os.path.basename(os.path.dirname(file_path))
                file_list.append((file_path, date_str))
    
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