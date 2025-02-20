### Sharbat 2025, slight modifications of Timothee 2024

### Functions to extract the dat files for each genotype for each experiments. 

### For the analysis to be efficient, you need to take the files that come out of the syncing process from tracker's computer to eq_ncb.
### The files are the one on the eq-ncb.
### For the names to be formatted the right way. 

import subprocess
import pandas as pd
import os
import json
import time 
from functools import wraps
import re
### To put in a new file with all
# name_columns = ["time","id","persistence","speed","midline","loc_x","loc_y","vel_x","vel_y","orient","pathlen"]
#directory to store the computed data

### decorator 
### measures and prints the execution time of a function
def timeit(func):
    """Decorator to time the function

    Args:
        func (func): function to time

    Returns:
        func : wrapper
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{total_time:.4f} seconds :  {func.__name__}{args}')
        return result
    return timeit_wrapper



### Functions 
### __________________________________________________________________

### For one file 

@timeit
def run_chore(experiment_path : str) -> None:
    """Function that launch the analysis of one experiment using the choreography setup to produce the .dat files needed for the analysis
    The command used is in the .sh file called "chore_command"

    Args:
        experiment_path (str): path to the experiments. Should at least contain the needed files to run the chore.jar app
        
    """
    sh_script = "chore_command.sh" #Script that contains the command
    commande = [
                "bash",        
                sh_script,   
                experiment_path 
            ]
    process = subprocess.Popen(
            commande,
            stdout=subprocess.PIPE,  
            stderr=subprocess.PIPE,  
            universal_newlines=True,  
            bufsize=1,                
            shell=False               
        )
    
    process.wait()
    

def dict_expe_(experiment_path : str) -> dict[str,str] :
    """Function that return the different parts of the experiement from path of the experiment
    NB : the path need to contain the actual files output from the tracker

    Args:
        experiment_path (str): path of the experiment

    Returns:
        dict: dict that contain the following data : date, genotype, effector, tracker, protocol 
    """
    list_files = os.listdir(experiment_path)
    list_files = [file for file in list_files if "@" in file] #we remove all the files like the trx
    common = os.path.commonprefix(list_files)
    list_ = common.split("@")
    list_.remove(list_[-1]) #We don't care about the last element
    dict_expe = {"date" : list_[0], "genotype" : list_[1],"effector" : list_[2],"tracker" : list_[3], "protocol" : list_[4]}
    return dict_expe
    
    
def is_analysed(path_experiment : str) -> bool :
    """Function that takes the path of the experiment as an input and return a boolean that is True if the experiment have already been analysed

    Args:
        path_experiment (str): Path of the experiment (absolute path is the best)

    Returns:
        bool : True if the experiment have been analysed (it contains at least 1 .dat file) and False in the other case
    """
    bl = False
    list_files = os.listdir(path_experiment)
    list_ext = []
    for file in list_files :
        ext = os.path.splitext(file)[1]
        list_ext.append(ext)
    
    if ".dat" in list_ext :
         bl = True
    return bl
    

def list_dat(experiment_path : str) -> list[str] :
    """Function that retrieve the absolute paths of all the .dat file from the path of the experiment)

    Args:
        experiment_path (str): Path of the experiment

    Returns:
        str list: list of all the absolute path of the .dat file of 1 experiement
    """
    
    experiment_path = os.path.abspath(experiment_path)
    list_files = os.listdir(experiment_path)
    list_path_dat = [experiment_path+"/"+file for file in list_files if file.endswith(".dat")]
    return list_path_dat 

def save_json(obj : dict ,name : str ,dir :str = "") -> None:
    """Function that save an object at the desired location with the desired name

    Args:
        obj (dict): object to save
        name (str): name to save the file
        dir (str, optional): directory where to save the json file. Defaults to "".
    """
    with open(dir+"/"+name+'.json',"w") as fp:
        json.dump(obj,fp)

@timeit
def concatenate_and_save(experiment_path : str ,name_columns : list[str] ,save_folder : str):
    """Function that retrieve the data for on experiement and save it at the desired location as a json file

    Args:
        experiment_path (str): path to go to the experiement
        name_columns (str list): list of the name of the columns of de .dat files (cf. features name and chore_command.sh)
        save_folder (str): folder to save the json file
        
    """
    list_path_dat = list_dat(experiment_path)
    dict_expe = dict_expe_(experiment_path)
    
    #To produce each dataframe
    def read_csv_format(path_dat):
        return pd.read_csv(sep=r"\s+",names=name_columns,header=None,filepath_or_buffer= path_dat)
    
    
    list_data = [read_csv_format(path) for path in list_path_dat]
    
    concatenated_data = pd.concat(ignore_index=True, objs=list_data)
    dict_data = concatenated_data.to_dict(orient='dict')
    name = "@".join([dict_expe[name] for name in ["genotype","effector","date"]])
    save_json(dict_data,name,save_folder)
    



### For all the files
#_________________________________________________________________________________

def retrieve_path_exp(path_data :str) -> list[str] :
    """Function that takes the path of the data (where you have all the experiments) and output a list of the names of the experiments
    This will allow to avoid errors

    Args:
        path_data (str): Path of the folder with all the experiements as it is on the T7 (in the behavior room)

    Returns:
        str list : list of the name of the experiment folders
    """
    #Pattern of name for the directory
    pattern = re.compile('[0-9]{8}_[0-9]{6}')
    
    #Retrieving all the paths
    all_paths = os.listdir(path_data)
    
    #Sorting the paths that are experiments according to the pattern of their name
    path_exp = [path for path in all_paths if pattern.match(path)]
    return path_exp


def chore_all(path_data : str) -> None:
    """Function run the chore command other all the experiment in the data folder that hasn't been analysed

    Args:
        path_data (str): path of the folder with all the experiments
    """
    # list_path_exp = retrieve_path_exp(path_data)
    # list_path_exp = [path_data + "/" + path_exp for path_exp in list_path_exp]
    # for path in list_path_exp :
    if not is_analysed(path_data) :
        run_chore(path_data)
    else:
        print(os.path.basename(path_data) + " :  Already analysed ")
    

def all_concatenate_save(path_data : str ,name_columns : list[str] ,save_folder : str) -> None:
    """Function that concatenate all the data for each experiments and save the computed (.json) file in the save_folder

    Args:
        path_data (str): path of the data folder
        name_columns (str list): list of the names of the columns of the .dat files (cf. features name for chore command)
        save_folder (str): path of the folder to save the computed .json files
    """
    # list_path_exp = retrieve_path_exp(path_data)
    # list_path_exp = [path_data + "/" + path_exp for path_exp in list_path_exp]
    # for path in list_path_exp : 
    concatenate_and_save(path_data,name_columns,save_folder)





### Main Function ________________________________

def run_compute(path_data :str ,name_columns : list[str] ,save_folder : str) -> None:
    """Function that compute the .json file from the raw data outputed from the MWT

    Args:
        path_data (str): path to the data folder
        name_columns (str list): name to give to the columns (cf. features of chore command)
        save_folder (str ): path to the folder to save the files
    """
    #Run the chore of the experiments
    chore_all(path_data)
    
    #Save everything
    all_concatenate_save(path_data,name_columns,save_folder)