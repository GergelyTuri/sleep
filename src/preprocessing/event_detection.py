import logging
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import numpy.ma as ma
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
from typing import Iterator, Tuple, List

class Denoiser:
    """
    A class to apply denoising methods to calcium imaging signals.

    Attributes:
        method (str): Denoising method to be applied. Options include "savgol", "gaussian", "median", and "none".
        params (dict): Parameters specific to the denoising method.
    """
    def __init__(self, method: str = "savgol", **kwargs):
        """
        Initialize the Denoiser instance with a chosen method and associated parameters.

        Args:
            method (str): The denoising method to apply.
            **kwargs: Parameters specific to the chosen denoising method.
        """
        self.method = method
        self.params = kwargs

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the selected denoising method to a signal.

        Args:
            signal (np.ndarray): Input 1D signal to be denoised.

        Returns:
            np.ndarray: Denoised signal.
        """
        if self.method == "savgol":
            return savgol_filter(signal, self.params.get("window_length", 11), self.params.get("polyorder", 3))
        elif self.method == "gaussian":
            return gaussian_filter1d(signal, self.params.get("sigma", 2))
        elif self.method == "median":
            return medfilt(signal, self.params.get("kernel_size", 5))
        elif self.method == "none":
            return signal  # No denoising applied
        else:
            raise ValueError(f"Unknown denoising method: {self.method}")


class EventDetection:
    """
    A class for detecting transient events in calcium imaging signals.

    Attributes:
        ON_THRESHOLD (float): Threshold above which a signal is considered an event onset.
        OFF_THRESHOLD (float): Threshold below which a signal is considered an event offset.
        MAX_SIGMA (float): Maximum sigma value for event noise estimation.
        MIN_DURATION (float): Minimum duration for an event to be considered valid (in seconds).
        MAX_DURATION (float): Maximum duration for an event to be considered valid (in seconds).
        N_BINS_PER_SIGMA (int): Number of bins per sigma for event classification.
        N_BINS_PER_SEC (int): Number of bins per second for event classification.
        denoiser (Denoiser): Instance of the Denoiser class for signal denoising.
    """
    def __init__(self, config_file: str = "configs.json", denoiser_method: str = "savgol", **denoise_params):
        """
        Initialize the EventDetection instance and load configuration settings.

        Args:
            config_file (str): Path to the JSON configuration file.
            denoiser_method (str): Method for denoising signals.
            **denoise_params: Parameters for the denoising method.
        """
        if os.path.exists(config_file):
            with open(config_file) as f:
                settings = json.load(f).get("calculateTransients", {})
        else:
            settings = {}

        self.ON_THRESHOLD = settings.get("ON_THRESHOLD", 3)
        self.OFF_THRESHOLD = settings.get("OFF_THRESHOLD", 1)
        self.MAX_SIGMA = settings.get("MAX_SIGMA", 5)
        self.MIN_DURATION = settings.get("MIN_DURATION", 0.1)
        self.MAX_DURATION = settings.get("MAX_DURATION", 5)
        self.N_BINS_PER_SIGMA = settings.get("N_BINS_PER_SIGMA", 2)
        self.N_BINS_PER_SEC = settings.get("N_BINS_PER_SEC", 4)

        self.nTimeBins = int((self.MAX_DURATION - self.MIN_DURATION) * self.N_BINS_PER_SEC + 1)
        self.nSigmaBins = int((self.MAX_SIGMA - self.ON_THRESHOLD) * self.N_BINS_PER_SIGMA + 1)

        self.denoiser = Denoiser(method=denoiser_method, **denoise_params)

    def apply_denoising(self, dfof: np.ndarray) -> np.ndarray:
        """
        Apply denoising to an array of signals.

        Args:
            dfof (np.ndarray): 2D array of signals (cells x time points).

        Returns:
            np.ndarray: Denoised signals (same dimensions as input).
        """
        return np.array([self.denoiser.apply(trace) for trace in dfof])

    def estimate_noise(self, dfof: np.ndarray, transients: List = None) -> np.ndarray:
        """
        Estimate noise levels in signals, optionally excluding transients.

        Args:
            dfof (np.ndarray): 2D array of signals (cells x time points).
            transients (List, optional): Transient event data for masking.

        Returns:
            np.ndarray: Noise levels for each signal (1D array of floats).
        """
        if transients is None:
            return np.nanstd(dfof, axis=1)

        mask = np.zeros_like(dfof, dtype=bool)
        for cell_idx, cell_transients in enumerate(transients):
            for start, end, _, _, _ in cell_transients:
                mask[cell_idx, start:end + 1] = True

        return ma.array(dfof, mask=mask).std(axis=1).data

    def identify_events(self, dfof: np.ndarray) -> Iterator[Tuple[int, int, int]]:
        """
        Identify start and end points for transient events in signals.

        Args:
            dfof (np.ndarray): 2D array of signals (cells x time points).

        Yields:
            Tuple[int, int, int]: (cell index, event start index, event end index).
        """
        L = dfof.shape[1]
        for cell_index in range(dfof.shape[0]):
            cell_data = dfof[cell_index]
            start_index = 0
            while start_index < L:
                starts = np.where(cell_data[start_index:] > self.ON_THRESHOLD)[0]
                if len(starts) == 0:
                    break
                abs_start = start_index + starts[0]
                ends = np.where(cell_data[abs_start:] < self.OFF_THRESHOLD)[0]
                if len(ends) == 0:
                    break
                abs_end = abs_start + ends[0]
                yield cell_index, abs_start, abs_end
                start_index = abs_end + 1

    def identify_transients(self, dfof: np.ndarray, frame_period: float, thresholds: np.ndarray, noise: np.ndarray) -> List[List[Tuple]]:
        """
        Detect transient events based on thresholds and noise estimates.

        Args:
            dfof (np.ndarray): 2D array of signals (cells x time points).
            frame_period (float): Time duration of a single frame (seconds).
            thresholds (np.ndarray): Thresholds for event detection.
            noise (np.ndarray): Estimated noise levels for each signal.

        Returns:
            List[List[Tuple]]: List of detected transients for each signal.
        """
        transients = []
        epsilon = 1e-6

        for cell_index in range(dfof.shape[0]):
            cell_transients = []
            cell_data = dfof[cell_index]
            noise_std = max(noise[cell_index], epsilon)
            normalized_data = cell_data / noise_std

            for _, abs_start, abs_end in self.identify_events(np.expand_dims(normalized_data, axis=0)):
                duration = (abs_end - abs_start) * frame_period
                event_segment = normalized_data[abs_start:abs_end + 1]
                amplitude = np.nanmax(event_segment)
                peak_index = np.argmax(event_segment)
                max_time = abs_start + peak_index

                sigma_bin_ind = min(
                    int((amplitude - self.ON_THRESHOLD) * self.N_BINS_PER_SIGMA),
                    self.nSigmaBins - 1,
                )

                if duration > thresholds[sigma_bin_ind, 0]:
                    cell_transients.append((abs_start, abs_end, max_time, amplitude, duration))

            transients.append(cell_transients)

        return transients

    def run_transient_detection(self, dfof: np.ndarray, frame_period: float):
        """
        Run transient detection pipeline: denoise signals, estimate noise, and detect transients.

        Args:
            dfof (np.ndarray): 2D array of signals (cells x time points).
            frame_period (float): Time duration of a single frame (seconds).

        Returns:
            Tuple[List[List[Tuple]], np.ndarray, np.ndarray]: (final transients, final noise levels, denoised signals).
        """
        logging.info("Step 0: Applying denoising...")
        dfof_denoised = self.apply_denoising(dfof)

        logging.info("Step 1: Initial noise estimation...")
        initial_noise = self.estimate_noise(dfof_denoised)

        logging.info("Step 2: Defining thresholds...")
        thresholds = np.full((self.nSigmaBins, 1), self.MIN_DURATION)

        logging.info("Step 3: Initial transient detection...")
        initial_transients = self.identify_transients(dfof_denoised, frame_period, thresholds, initial_noise)

        logging.info("Step 4: Final noise estimation excluding detected transients...")
        final_noise = self.estimate_noise(dfof_denoised, transients=initial_transients)

        logging.info("Step 5: Final transient detection...")
        final_transients = self.identify_transients(dfof_denoised, frame_period, thresholds, final_noise)

        logging.info("Transient detection completed.")
        return final_transients, final_noise, dfof_denoised

    def plot_raw_and_denoised(self, raw_signals: np.ndarray, denoised_signals: np.ndarray, cell_index: int = 0, save_path: str = None):
          """
          Plot raw and denoised signals for a specific cell and optionally save as HTML.

          Args:
              raw_signals (np.ndarray): 2D array of raw signals (cells x time points).
              denoised_signals (np.ndarray): 2D array of denoised signals.
              cell_index (int): Index of the cell to visualize.
              save_path (str, optional): File path to save the plot as an HTML file.
          """
          raw = raw_signals[cell_index]
          denoised = denoised_signals[cell_index]

          fig = go.Figure()
          fig.add_trace(go.Scatter(
              x=np.arange(len(raw)), 
              y=raw, 
              mode='lines', 
              name='Raw Signal', 
              line=dict(color='gray', width=1),
              opacity=0.6
          ))
          fig.add_trace(go.Scatter(
              x=np.arange(len(denoised)), 
              y=denoised, 
              mode='lines', 
              name='Denoised Signal', 
              line=dict(color='blue', width=2)
          ))

          fig.update_layout(
              title=f"Raw vs. Denoised Signal for Cell {cell_index}",
              xaxis_title="Time Points",
              yaxis_title="Signal Intensity",
              template="plotly_white",
              hovermode="x unified",
              xaxis=dict(showgrid=True, zeroline=False),
              yaxis=dict(showgrid=True),
          )

          if save_path:
              fig.write_html(save_path)
          fig.show()
          return fig


    def plot_transient_events(self, dfof_signals: np.ndarray, transients: List[List[Tuple]], cell_index: int = 0, save_path: str = None):
        """
        Plot transient events for a specific cell and optionally save as HTML.

        Args:
            dfof_signals (np.ndarray): 2D array of signals (cells x time points).
            transients (List[List[Tuple]]): List of transient events for all cells.
            cell_index (int): Index of the cell to visualize.
            save_path (str, optional): File path to save the plot as an HTML file.
        """
        signal = dfof_signals[cell_index]
        cell_transients = transients[cell_index] if transients else []

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(signal)), 
            y=signal, 
            mode='lines', 
            name=f'Cell {cell_index}', 
            line=dict(color='blue', width=1),
            hoverinfo="x+y"
        ))

        x_mid, y_amp, customdata = [], [], []
        y0_baseline = np.percentile(signal, 5)

        for start, end, max_time, amplitude, duration in cell_transients:
            fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=y0_baseline,
                y1=amplitude,
                fillcolor='rgba(255, 165, 0, 0.3)',
                line=dict(width=0),
                layer='below'
            )
            x_mid.append((start + end) / 2)
            y_amp.append(amplitude)
            customdata.append((start, end, max_time, amplitude))

        if x_mid:
            fig.add_trace(go.Scatter(
                x=x_mid,
                y=y_amp,
                mode='markers',
                name="Transient Events",
                marker=dict(color='red', size=8, symbol='circle'),
                hovertemplate=(
                    "Transient Event<br>"
                    "Start: %{customdata[0]}<br>"
                    "End: %{customdata[1]}<br>"
                    "Peak: %{customdata[2]}<br>"
                    "Amplitude: %{customdata[3]:.2f}<extra></extra>"
                ),
                customdata=customdata
            ))

        fig.update_layout(
            title=f"Transient Events for Cell {cell_index}",
            xaxis_title="Time Points",
            yaxis_title="Signal Intensity",
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True),
        )

        if save_path:
            fig.write_html(save_path)
        fig.show()
        return fig

    def plot_noise(self, dfof_signals: np.ndarray, noise: np.ndarray, cell_index: int = 0, save_path: str = None):
        """
        Plot noise levels for a specific cell and optionally save as HTML.

        Args:
            dfof_signals (np.ndarray): 2D array of signals (cells x time points).
            noise (np.ndarray): 1D array of noise levels for each cell.
            cell_index (int): Index of the cell to visualize.
            save_path (str, optional): File path to save the plot as an HTML file.
        """
        signal = dfof_signals[cell_index]
        noise_level = noise[cell_index]
        baseline = np.mean(signal)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(signal)), y=signal, mode='lines', name=f'Cell {cell_index}', line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=np.arange(len(signal)), y=[baseline + noise_level] * len(signal), mode="lines", name="+1σ Noise Level", line=dict(color="red", dash="dot")))
        fig.add_trace(go.Scatter(x=np.arange(len(signal)), y=[baseline - noise_level] * len(signal), mode="lines", name="-1σ Noise Level", line=dict(color="red", dash="dot")))

        fig.update_layout(
            title=f"Noise Estimation for Cell {cell_index}",
            xaxis_title="Time Points",
            yaxis_title="Signal Intensity",
            template="plotly_white",
            hovermode="x"
        )

        if save_path:
            fig.write_html(save_path)
        fig.show()
        return fig


    def visualize_all(self, raw_signals: np.ndarray, denoised_signals: np.ndarray, noise: np.ndarray, transients: List[List[Tuple]], cell_index: int = 0, save_prefix: str = None):
        """
        Visualize all data (raw, denoised signals, noise, and transient events) for a specific cell.

        Optionally save each figure as an HTML file using a filename prefix.

        Args:
            raw_signals (np.ndarray): 2D array of raw signals (cells x time points).
            denoised_signals (np.ndarray): 2D array of denoised signals.
            noise (np.ndarray): 1D array of noise levels for each cell.
            transients (List[List[Tuple]]): List of detected transients for all cells.
            cell_index (int): Index of the cell to visualize.
            save_prefix (str, optional): Prefix to save HTML plots (e.g., "cell0" saves as "cell0_raw_denoised.html", etc.)
        """
        self.plot_raw_and_denoised(
            raw_signals, denoised_signals, cell_index,
            save_path=f"{save_prefix}_raw_denoised.html" if save_prefix else None
        )
        self.plot_noise(
            denoised_signals, noise, cell_index,
            save_path=f"{save_prefix}_noise.html" if save_prefix else None
        )
        self.plot_transient_events(
            denoised_signals, transients, cell_index,
            save_path=f"{save_prefix}_transients.html" if save_prefix else None
        )


    def transients_to_dataframe(self, transients: List[List[Tuple]]) -> pd.DataFrame:
        """
        Convert transients data into a pandas DataFrame for analysis.

        Args:
            transients (List[List[Tuple]]): List of transient events for all cells.

        Returns:
            pd.DataFrame: A DataFrame containing transient events data with columns: 
                          'cell_index', 'start', 'end', 'peak', 'amplitude', 'duration'.
        """
        data = []
        for cell_index, cell_transients in enumerate(transients):
            for start, end, peak, amplitude, duration in cell_transients:
                data.append({
                    "cell_index": cell_index,
                    "start": start,
                    "end": end,
                    "peak": peak,
                    "amplitude": amplitude,
                    "duration": duration
                })

        return pd.DataFrame(data)

    def export_transients_to_csv(self, transients: List[List[Tuple]], file_name: str):
        """
        Export transients data to a CSV file.

        Args:
            transients (List[List[Tuple]]): List of transient events for all cells.
            file_name (str): Name of the CSV file to save the data.

        Returns:
            None
        """
        df = self.transients_to_dataframe(transients)  
        df.to_csv(file_name, index=False)
        logging.info("Transients exported to %s.", file_name)
