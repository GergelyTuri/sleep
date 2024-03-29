"""
author: GT
date: 1/10/24
This script adds a time-averaged image to suite2p data.

Usage:
python -m add_time_avg -d <directory>

Arguments:
-d, --directory : The directory name under which to find the suite2p data.

Example:
python -m add_time_avg -d /path/to/mouse
"""
import argparse as arg
import logging
import sys
from os import walk
from os.path import dirname, isdir, join

from src.suite2p_class import Suite2p

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_time_avg(directory):
    """
    Add time averaged image to suite2p data.

    Args:
        directory (str): The directory name under which to find the suite2p data.

    Returns:
        None
    """
    if not isdir(directory):
        logging.error(f"Error: The directory '{directory}' does not exist.")
        sys.exit(1)

    for root, dirs, files in walk(directory):
        if "suite2p" in dirs:
            s2p_path = join(root, "suite2p")
            save_dir = dirname(s2p_path)
            print(f"Processing suite2p data in: {s2p_path}")
            s2p = Suite2p(s2p_path)
            time_avg_image = s2p.load_avg_image()                      
            result = s2p.plot_time_avg_image(time_avg_image, save_path=save_dir)
            if result is None:
                logging.warning(f"No time-averaged image was saved in: {save_dir}")
            else:
                logging.info(f"Time-averaged image saved in: {save_dir}")


if __name__ == "__main__":
    argparser = arg.ArgumentParser(description="Add time average to suite2p data")
    argparser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="directory name under which to find the suite2p data",
    )
    args = argparser.parse_args()
    add_time_avg(args.directory)
