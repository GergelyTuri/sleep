"""
author: GT
date: 1/10/24
This script adds a time-averaged image to suite2p data.

Usage:
python -m add_time_avg -d <directory> [-t]

Arguments:
-d, --directory : The directory name under which to find the suite2p data.
-t, --tif       : Optional flag to save the output as a multipage TIFF instead of PNG.

Example:
python -m add_time_avg -d /path/to/mouse -t
"""
import argparse as arg
import logging
import sys
from os import walk
from os.path import dirname, isdir, join

from src.io.suite2p_io import Suite2p


def add_time_avg(directory, save_as_tif=False):
    """
    Add time averaged image to suite2p data.

    Args:
        directory (str): The directory name under which to find the suite2p data.
        save_as_tif (bool): Flag indicating whether to save the output as a TIFF file.

    Returns:
        None
    """
    if not isdir(directory):
        logging.error("Error: The directory '%s' does not exist.", directory)
        sys.exit(1)

    for root, dirs, files in walk(directory):
        if "suite2p" in dirs:
            s2p_path = join(root, "suite2p")
            save_dir = dirname(s2p_path)
            logging.info("Processing suite2p data in: %s", s2p_path)
            s2p = Suite2p(s2p_path)
            time_avg_image = s2p.load_avg_image()                      
            if save_as_tif:
                save_path = join(save_dir, "time_avg_image.tif")
                s2p.save_time_avg_as_tiff(time_avg_image, save_path)
            else:
                result = s2p.plot_time_avg_image(time_avg_image, save_path=save_dir)
                if result is None:
                    logging.warning("No time-averaged image was saved in: %s", save_dir)
                else:
                    logging.info("Time-averaged image saved in: %s", save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    argparser = arg.ArgumentParser(description="Add time average to suite2p data")
    argparser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="directory name under which to find the suite2p data",
    )
    argparser.add_argument(
        "-t",
        "--tif",
        action='store_true',
        help="optional flag to save the output as a multipage TIFF instead of PNG",
    )
    args = argparser.parse_args()
    add_time_avg(args.directory, save_as_tif=args.tif)
