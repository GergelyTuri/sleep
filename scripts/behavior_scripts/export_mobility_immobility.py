"""
Script to export mobility and mobility data to JSON file.
"""

import logging
from argparse import ArgumentParser as AP
from os.path import exists, join

from src.io import behavior_io as bc
from src.io import mouse_io as mc


def main(args):
    """
    Export mobility and mobility data to JSON files.

    Args:
        args: Parsed command-line arguments containing mouse_ID and overwrite.
    """
    # Initialize MouseData object
    mouse_data = mc.MouseData(args.mouse_ID)
    
    # Find behavior folders
    behavior_folders = mouse_data.find_behavior_folders()
    
    for behavior_folder in behavior_folders:
        behavior = bc.BehaviorData(behavior_folder)
        output_path = join(behavior_folder, "mobility_immobility.json")
        
        # Skip folder if JSON file exists and overwrite flag is not set
        if exists(output_path) and not args.overwrite:
            logging.info("Skipping folder %s as file already exists.", behavior_folder)
            continue

        try:
            logging.info("Processing folder: %s", behavior_folder)

            # Process velocity and define mobility
            processed_velo = behavior.load_processed_velocity()
            mobility = behavior.define_mobility(velocity=processed_velo)

            # Save mobility data to JSON
            mobility.to_json(output_path, orient="records", indent=4)
            logging.info("Successfully processed and saved data for folder: %s", behavior_folder)

        except FileNotFoundError:
            logging.warning("Folder %s not found.", behavior_folder)

        except Exception:
            logging.exception("Failed to process folder %s", behavior_folder)
            # Continue with the next folder without stopping the script

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Parsing command-line arguments
    parser = AP(description="Export mobility and immobility data to JSON files (mobility = 1).")
    parser.add_argument("mouse_ID", help="the ID of the mouse to analyze.")
    parser.add_argument("-o", "--overwrite", help="overwrite existing files", action="store_true")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
