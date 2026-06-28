"""Class for handling Suite2p data.
Attributes:
    s2p_folder (Union[str, Path]): Path to the Suite2p folder.
    iscell_filename (str): Name of the iscell file. Defaults to "iscell".
Methods:
    __post_init__():
        Converts s2p_folder to a Path object if it is a string.
    _load_data_from_dir(subdir_name, signal_source):
    _get_raw_signal(signal_source="F", plane=None):
    get_iscell_indices(plane=None):
        Get the positions (indices) of cells marked as '1' in the iscell.npy file, with an optional plane filter.
    get_cells(plane=None):
    get_npil(plane=None):
    get_spikes(plane=None):
    load_avg_image():
    plot_time_avg_image(ops_array, save_path=None):
    _save_or_display_plot(fig, save_path=None):
    save_time_avg_as_tiff(ops_array, save_path):"""

import datetime
import logging
import warnings
from dataclasses import dataclass
from os.path import exists, isdir, join
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COMBINED_DIR_NAME = "combined"
PLANE0_DIR_NAME = "plane0"

logger = logging.getLogger(__name__)


class DirectoryNotFoundError(Exception):
    """Exception raised for errors in the directory path."""

    pass


@dataclass
class Suite2p:
    """Class for suite2p data
    Initialized with the path to the suite2p folder"""

    s2p_folder: Union[str, Path]
    iscell_filename: str = "iscell"

    def __post_init__(self):
        if isinstance(self.s2p_folder, str):
            self.s2p_folder = Path(self.s2p_folder)

    def _load_data_from_dir(self, subdir_name, signal_source):
        """
        Helper method to load data from a directory.

        Args:
            subdir_name (str): The name of the subdirectory containing the data.
            signal_source (str): The name of the signal source file. "F" - cell fluorescence data,
                "Fneu" - neuropil data, "spks" - deconvolved spike data.

        Returns:
            Tuple: A tuple containing the iscells, raw_signal arrays and the plane index array
                or None if the directory does not exist.
        """

        path = join(self.s2p_folder, subdir_name)
        if not isdir(path):
            raise DirectoryNotFoundError(f"Directory {path} does not exist")

        iscell_path = join(path, f"{self.iscell_filename}.npy")
        if not exists(iscell_path):
            raise FileNotFoundError(f"File {self.iscell_filename} not found in {path}")

        iscells = np.load(iscell_path, allow_pickle=True)
        # Extract and convert the first column to boolean
        is_cell_boolean = iscells[:, 0].astype(bool)

        signal_path = join(path, f"{signal_source}.npy")
        if not exists(signal_path):
            raise FileNotFoundError(f"Expected Suite2p signal file at {signal_path}")
        raw_signal = np.load(signal_path, allow_pickle=True)

        stat_path = join(path, "stat.npy")
        if not exists(stat_path):
            raise FileNotFoundError(f"Expected Suite2p stat file at {stat_path}")
        stat_file = np.load(stat_path, allow_pickle=True)
        values_iplane = [d["iplane"] for d in stat_file if "iplane" in d]
        plane_index = np.array(
            values_iplane
        )  # palne index will be an empty array in case of a single plane dataset
        return is_cell_boolean, raw_signal, plane_index
    
    def _get_raw_signal(
        self, signal_source: str = "F", plane: Optional[int] = None
    ) -> np.ndarray:
        """
        Returns the signal of the `is_cell` cells in the Suite2p output directory.

        Args:
            signal_source (str, optional): The source of the signal. Defaults to "F".
            plane (int, optional): The plane index. If provided, returns the signal of cells in the specified plane.

        Returns:
            np.ndarray: The true signal of the cells.
        """
        try:
            iscells, raw_signal, plane_index = self._load_data_from_dir(
                COMBINED_DIR_NAME, signal_source
            )
        except (DirectoryNotFoundError, FileNotFoundError):
            warnings.warn(
                f"Combined directory not found or incomplete, falling back to {PLANE0_DIR_NAME}."
            )
            iscells, raw_signal, plane_index = self._load_data_from_dir(
                PLANE0_DIR_NAME, signal_source
            )
        if not plane_index.size:
            if plane is not None:
                warnings.warn(
                    f"Single plane dataset detected, but plane {plane} was specified. Ignoring plane parameter."
                )
            signal = np.where(iscells)[0]
        else:
            if plane is not None:
                if plane < 0 or plane > np.max(plane_index):
                    raise ValueError(f"Plane {plane} is out of range")
                signal = np.where(iscells & (plane_index == plane))[0]
            else:
                signal = np.where(iscells)[0]

        return raw_signal[signal, :]
    
    def get_iscell_indices(self, plane: Optional[int] = None) -> np.ndarray:
        """
        Get the positions (indices) of cells marked as '1' in the iscell.npy file,
        with an optional plane filter.

        Args:
            iscell_file (str, optional): The name of the iscell file. Defaults to 'iscell.npy'.
            plane (int, optional): The plane index. If provided, returns positions of cells in the specified plane.

        Returns:
            np.ndarray: An array of indices where the first column of iscell.npy is 1.

        Raises:
            FileNotFoundError: If the iscell file cannot be found in either directory.
        """
        try:
            # Load data from combined directory
            iscells, _, plane_index = self._load_data_from_dir(COMBINED_DIR_NAME, self.iscell_filename)
        except DirectoryNotFoundError:
            # Fallback to plane0 directory if combined not found
            warnings.warn(f"Combined directory not found, falling back to {PLANE0_DIR_NAME}.")
            iscells, _, plane_index = self._load_data_from_dir(PLANE0_DIR_NAME, self.iscell_filename)

        # Extract positions where the first column is 1
        positions_of_ones = np.where(iscells)[0]

        # Filter by plane if applicable
        if plane_index.size == 0:
            if plane is not None:
                warnings.warn(
                    f"Single plane dataset detected, but plane {plane} was specified. Ignoring plane parameter."
                )
        else:
            if plane is not None:
                if plane < 0 or plane > np.max(plane_index):
                    raise ValueError(f"Plane {plane} is out of range")
                positions_of_ones = positions_of_ones[plane_index[positions_of_ones] == plane]

        return positions_of_ones


    def get_cells(self, plane=None):
        """
        Returns the cell signals for the fluorescence data.

        Args:
            plane (int, optional): The plane index. Defaults to None.

        Returns:
            numpy.ndarray: An array of cell signals for the fluorescence data.
        """
        return self._get_raw_signal(signal_source="F", plane=plane)

    def get_npil(self, plane=None):
        """
        Computes the neuropil signal for each cell in the Suite2p object.

        Parameters:
        -----------
        plane : int or None, optional
            The plane index for which to compute the neuropil signal. If None, the signal will be computed for all planes.

        Returns:
        -------
        npil : numpy.ndarray
            The neuropil signal for each cell.
        """
        return self._get_raw_signal(signal_source="Fneu", plane=plane)

    def get_spikes(self, plane=None):
        """
        Returns the spike signal for each cell in the Suite2p object.

        Parameters:
        -----------
        plane: int, optional
            The plane number for which to retrieve the spike signal. If not specified, all planes are considered.

        Returns:
        -------
        numpy.ndarray: Array of shape (num_cells, num_frames) containing the spike signal for each cell.
        """
        return self._get_raw_signal(signal_source="spks", plane=plane)

    def load_avg_image(self):
        """
        TODO: make it plane specific
        TODO: add ROI masks
        Plot and display the time-averaged image(s) from the ops_array.

        Args:
            save_path (str): Optional. The file path to save the plot.

        Returns:
            str or None: If save_path is provided, returns the path where the plot is saved. Otherwise, returns None.
        """
        ops_path = join(self.s2p_folder, "ops1.npy")
        if not exists(ops_path):
            logger.warning("File not found: %s", ops_path)
            return None

        ops_array = np.load(ops_path, allow_pickle=True)
        return ops_array

    def plot_time_avg_image(self, ops_array, save_path=None):
        """
        Plots the time-averaged images from the given ops_array.

        Parameters:
        ops_array (list): A list of dictionaries, each containing an image under the key "meanImg".
        save_path (str, optional): The path where the plot should be saved. If None, the plot will be displayed.

        Returns:
        matplotlib.figure.Figure: The figure object containing the plot, or None if ops_array is None.

        Notes:
        - If ops_array contains only one element, a single plot will be created.
        - If ops_array contains multiple elements, a grid of subplots will be created.
        - The images are flipped upside down to match the field of view (FOV).
        """
        if ops_array is None:
            return None

        num_elements = len(ops_array)

        plot_width_per_image = 5  # Width per image in inches
        plot_height_per_image = 5  # Height per image in inches

        # Single element handling
        if num_elements == 1:
            fig, ax = plt.subplots(
                figsize=(plot_width_per_image, plot_height_per_image)
            )
            # the image is flipped upside down to match the FOV
            image_data = ops_array[0]["meanImg"]
            ax.imshow(image_data, cmap="gray")
            ax.set_title("Mean Image")
            ax.axis("off")

        # Multiple elements handling
        else:
            cols = int(np.ceil(np.sqrt(num_elements)))
            rows = int(np.ceil(num_elements / cols))

            # Calculate the total plot size
            total_width = plot_width_per_image * cols
            total_height = plot_height_per_image * rows

            # Create subplots
            fig, axes = plt.subplots(rows, cols, figsize=(total_width, total_height))
            axes = axes.flatten()

            # Loop through each item and plot
            for i, ops in enumerate(ops_array):
                image_data = ops["meanImg"]
                axes[i].imshow(image_data, cmap="gray")
                axes[i].set_title(f"Mean Image {i+1}")
                axes[i].axis("off")

            # Turn off any unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
        plt.tight_layout()
        self._save_or_display_plot(fig, save_path)
        return fig

    def _save_or_display_plot(self, fig, save_path=None):
        """
        Save or display a matplotlib figure.

        Parameters:
        fig (matplotlib.figure.Figure): The figure to save or display.
        save_path (str, optional): The directory path where the figure should be saved. 
                                   If None, the figure will be displayed instead of saved.

        Returns:
        None
        """
        if save_path is not None:
            filename = "time_avg_image.png"
            full_save_path = join(save_path, filename)
            if not exists(full_save_path):
                try:
                    fig.savefig(full_save_path)
                    logger.info("Saved plot to %s", full_save_path)
                except Exception:
                    logger.exception("Error saving plot to %s", full_save_path)
                finally:
                    plt.close(fig)
            else:
                warnings.warn(
                    f"File already exists: {full_save_path}, saving new file."
                )
                filename = f"time_avg_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                full_save_path = join(save_path, filename)
                try:
                    fig.savefig(full_save_path)
                    logger.info("Saved new plot to %s", full_save_path)
                except Exception:
                    logger.exception("Error saving new plot to %s", full_save_path)
                finally:
                    plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    def save_time_avg_as_tiff(self, ops_array, save_path):
        """
        Save the time-averaged image(s) as a .tiff file.

        Args:
            ops_array (list): A list of dictionaries containing image data.
            save_path (str): Path to save the output .tiff file.

        Returns:
            None
        """
        if ops_array is None:
            return

        num_elements = len(ops_array)
        tif_images = []  # List to store images for saving as multipage TIFF

        # Loop through each item in ops_array and prepare the image data
        for ops in ops_array:
            image_data = ops["meanImg"]
            
            # Normalize the image data to be between 0 and 255, similar to plot_time_avg_image
            image_data = np.clip(image_data, 0, None)  # Clip negative values to 0
            if image_data.max() > 0:
                image_data = (image_data / image_data.max()) * 255  # Normalize to 0-255
            
            # Convert to uint8 for saving
            tif_images.append(Image.fromarray(image_data.astype(np.uint8)))

        # Save as multipage TIFF if there are multiple images
        if num_elements > 1:
            tif_images[0].save(save_path, save_all=True, append_images=tif_images[1:])
        else:
            tif_images[0].save(save_path)

        logger.info("Saved time-averaged image(s) to %s", save_path)