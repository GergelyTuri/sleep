"""metadata info for sparse analysis"""

from src.io import mouse_io as mouse

# clean data sets: mouse id -> list of sessions
CLEAN_DATA = {"dock13a1": ["TSeries-08062024-1156-001.sima", "TSeries-08082024-1850-001.sima"],
              "dock13b1": ["TSeries-08082024-1650-001.sima"],
              "dock13a2": ["TSeries-08082024-1012-001.sima"]
              }

# destination folder for saving the plots and summary calculations
SAVE_FOLDER = "/data2/gergely/invivo_DATA/sleep/summaries/sparse_labeling"   

# find the full path to the TSeries folders
CLEAN_DATA_PATHS = []
for mouse_id, tseries_list in CLEAN_DATA.items():
    mouse_obj  = mouse.MouseData(mouse_id)
    all_tseries = mouse_obj.find_sima_folders()
    
    for tseries in tseries_list:  # Loop over individual TSeries segments
        for tseries_path in all_tseries:  # Loop over PosixPath objects
            if tseries_path.name == tseries:  # Compare the name of the PosixPath with the segment
                CLEAN_DATA_PATHS.append(str(tseries_path))  # Store full path as a string