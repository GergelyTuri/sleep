"""metadata info for sparse analysis"""

from src.classes import mouse_class as mouse

# clean data sets: mouse id -> list of sessions
CLEAN_GC_DATA = {"140502_5": ["TSeries-01082024-0803_GC1-001.sima",
                           "TSeries-01082024-0803_GC2-002.sima",
                           "TSeries-01092024-0803_GC1-001.sima",
                           "TSeries-01092024-0803_GC2-002.sima",
                           "TSeries-01092024-0803_GC3-003.sima",
                           # "TSeries-01112024-0853_GC1-001.sima", bad S/N
                           "TSeries-01112024-0853_GC2-002.sima",                           
                           ],
              "140302_3": ["TSeries-12112023-0825_sess01-001_Cycle00001_Element00001.sima", # not the best S/N
                           ],
              "140503_2": [".sima"],
              "140503_3": []
              }

CLEAN_MC_DATA = {"140302_3": ["TSeries-12112023-0825_sess02_MC-002.sima"]}

CLEAN_DENDRITE_DATA = {"140502_5": ["TSeries-01162024-0853_dendrite-001.sima",
                                    "TSeries-01172024-0745_dendrites-001.sima",
                                    ],
                       "140302_3": ["TSeries-01182024-0805-003.sima"]}

CLEAN_GC_DUALPLANE_DATA = {"140502_5": ["TSeries-02162024-1417-001.sima"],
                        "140302_3": ["TSeries-02192024-1358-001.sima"], # the somatic layer looks like crap
                        }

CLEAN_GC_MC_DUALPLANE_DATA = {}

GROUPING = {"Cort": ["140502_5", "140302_3"],
            "Control": ["140503_2", "140503_3"]
            }

# destination folder for saving the plots and summary calculations
SAVE_FOLDER = "/data2/gergely/invivo_DATA/sleep/summaries/cort_project"   

# find the full path to the TSeries folders
def find_clean_data_paths(clean_data):
    """
    Finds and returns the paths to clean data files.
    This function takes a dictionary where the keys are mouse IDs and the values are lists of time series segments.
    It creates a list of paths to the clean data files corresponding to these segments.
    Args:
        clean_data (dict): A dictionary where keys are mouse IDs (str) and values are lists of time series segments (str).
    Returns:
        list: A list of strings representing the full paths to the clean data files.
    """
    CLEAN_DATA_PATHS = []
    for mouse_id, tseries_list in clean_data.items():
        mouse_obj  = mouse.MouseData(mouse_id)
        all_tseries = mouse_obj.find_sima_folders()
        
        for tseries in tseries_list:  # Loop over individual TSeries segments
            for tseries_path in all_tseries:  # Loop over PosixPath objects
                if tseries_path.name == tseries:  # Compare the name of the PosixPath with the segment
                    CLEAN_DATA_PATHS.append(str(tseries_path))  # Store full path as a string