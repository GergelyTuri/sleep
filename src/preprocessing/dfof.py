"""Class for calculating the delta F over F.
based on lab3
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from . import filters
from src.io.suite2p_io import Suite2p

try:
    from src.config import (
        SUITE2P_DFOF_WINDOW, SUITE2P_DFOF_SIGMA, SUITE2P_DFOF_MIN_PERIODS,
        NPIL_COEFF, JIA_T1, JIA_T2,
    )
except ImportError:
    from config import (
        SUITE2P_DFOF_WINDOW, SUITE2P_DFOF_SIGMA, SUITE2P_DFOF_MIN_PERIODS,
        NPIL_COEFF, JIA_T1, JIA_T2,
    )


class DFOFStrategy(ABC):

    """Base class for dF/F calculation strategies. Derived strategies should
    implement the _filter_final_signal and _calculate_baseline methods."""

    name = "DFOF"

    def __init__(self) -> None:
        self._baseline = None

    def apply_to(self, mouse_id, s2p_folder, signal_source, **kwargs):
        """Apply dF/F analysis to an experiment object.

        Parameters
        ----------
        experiment : lab3.experiment.base.ImagingExperiment
            Instance of ImagingExperiment to analyze
        channel :
        label :

        Additional keyword arguments are passed directly to `calculate`.
        """
        sig_dict = self._load_signals(mouse_id, s2p_folder, signal_source)
        return self.calculate(**sig_dict, **kwargs)

    def calculate(self, signal, *args, **kwargs):
        """Calculate dF/F

        Parameters
        ----------
        signal : array-like (n_rois, n_samples)
            2D array of roi signals

        Returns
        -------
        dfof : pd.DataFrame (n_rois, n_samples)
            2D array of roi dF/F
        """

        if len(signal.shape) != 2:
            raise ValueError("Input signal must be 2D")

        base = self.calculate_baseline(signal, *args, **kwargs)
        return self._filter_final_signal((signal - base) / base, *args, **kwargs)

    # ----------------- Implement these methods to subclass ----------------- #

    @abstractmethod
    def _filter_final_signal(self, signal):
        """Apply smoothing to the final trace. This method should be implemented by subclasses."""
        pass

    @abstractmethod
    def _calculate_baseline(self, signal):
        """Estimate baseline for dFOF calculation. This method should be implemented by subclasses."""
        pass

    def calculate_baseline(self, signal, *args, **kwargs):
        """
        Parameters
        ----------
        signal : array-like (n_rois, n_samples)
            2D array of roi signals

        Returns
        -------
        baseline : pd.DataFrame (n_rois, n_samples)
            2D array of roi baselines
        """
        try:
            return self.baseline
        except AttributeError:
            self.baseline = self._calculate_baseline(signal, *args, **kwargs)
            return self.baseline

    # ------------------------------- Optional ------------------------------ #

    def _load_signals(self, mouse_id, s2p_folder, signal_source):
        """By default only the raw signal dataframe is passed to `calculate`
        when using `apply`. If you override `calculate` to take additional
        arguments in your custom strategy, you should load and add them to the
        dictionary here, which will be unpacked as arguments to `calculate`.
        """
        s2p_instance = Suite2p(mouse_id)
        sig_dict = {"signal": s2p_instance.true_signal(s2p_folder, signal_source)}
        return sig_dict


class SlowTrendMixin:

    """Mixin class that provides a final detrending step at the end of
    dF/F calculation, calculated as a rolling median filter."""

    # TODO should this override the calculate method? It would make composition
    # easier

    def calculate_slow_trend(self, signal, window=None, min_periods=None):
        """
        Parameters
        ----------
        signal : array-like (n_rois, n_samples)
            2D array of roi signals
        window : int, optional
            Size of window in frames to compute rolling median
        min_periods : float (0, 1], optional
            Fraction of frames in rolling window that must be non-NaN to return
            a non-NaN value in the filtered signal

        Returns
        -------
        slow_trend : pd.DataFrame (n_rois, n_samples)
            2D array of roi slow trends (rolling median)
        """
        if window is None:
            return pd.DataFrame(np.zeros(signal.shape))
        else:
            logger.info("Calculating slow trend")
            if not isinstance(signal, pd.DataFrame):
                signal = pd.DataFrame(signal)
            self.slow_trend = signal.rolling(
                window=window,
                min_periods=int(window * min_periods),
                center=True,
                axis=1,
            ).median()
            return self.slow_trend


class Suite2pDFOF(DFOFStrategy):

    """Class for calculating dF/F that takes into account Suite2p neuropil
    estimations. Subtracting the scaled neuropil from the raw trace can be
    valuable for removing bleedthrough from nearby ROIs or other local diffuse
    signals that a low-pass baseline will obscure. However, this initial
    subtraction complicates estimating dF/F, as the resulting residual trace
    will have a baseline close to zero (or negative). Thus it is necessary to
    "add back in" the component of the low-pass baseline that was removed by
    the neuropil correction, to avoid distortions in the dF/F due to small or
    negative denominator terms.

    The unfiltered neuropil is first subtracted from the unfiltered raw trace.
    Two partial baselines are estimated then by filtering the residual raw
    trace, and the neuropil. The residual raw trace minus its partial baseline
    forms the numerator term of the dF/F. The denominator is the sum of the
    partial baselines.

    The partial baselines are calculated by applying a minimax filter to the
    respective (smoothed) traces.

    Parameters
    ----------
    window : int, optional
        Size of the rolling window for min/max/smoothing operations, in frames.
        Defaults to 600
    sigma : float
        Standard deviation of the gaussian smoothing kernel applied prior to
        min/max filtering, in frames. Defaults to 10.
    min_periods : float (0, 1], optional
        Fraction  of frames in each window that must be non-NaN in order to
        return a non-NaN baseline. Defaults to 0.2.
    constant_denominator : bool, optional
        Whether to divide the dF result by a constant denominator when
        calculating dF/F, taken as the median of the total baseline across
        all time points. This is helpful to avoid artifactually increasing
        signal amplitudes over time due to a monotonically decaying baseline,
        for example during photobleaching. Defaults to False
    npil_coeff : float, optional
        Scaling coefficient applied to the neuropil before subtraction
        (F_corrected = F - npil_coeff * Fneu). Suite2p convention is 0.7.
        Defaults to NPIL_COEFF (0.7).

    Attributes
    ----------
    baseline : tuple
        Contains 3 DataFrames containing the total baseline ([0]),
        signal residual baseline ([1]), and neuropil baseline ([2]).

    Examples:
    >>> sig = s2p_mouse.true_signal(s2p_folder=s2p_folder)
    >>> npil = s2p_mouse.true_signal(s2p_folder=s2p_folder, signal_source="Fneu")
    >>> dfof_strategy = Suite2pDFOF()
    >>> dfof_calc = dfof_strategy.calculate(signal=sig, npil="Fneu")
    """

    def __init__(
        self,
        window=SUITE2P_DFOF_WINDOW,
        sigma=SUITE2P_DFOF_SIGMA,
        min_periods=SUITE2P_DFOF_MIN_PERIODS,
        constant_denominator=False,
        npil_coeff=NPIL_COEFF,
    ):
        self.window = window
        self.sigma = sigma
        self.min_periods = min_periods
        self.constant_denominator = constant_denominator
        self.npil_coeff = npil_coeff

    def _filter_final_signal(self, signal):
        """No final filtering"""
        return signal

    def _calculate_baseline(self, signal, npil):
        """Calculates the baseline signal from a raw signal and its neuropil component.

        Parameters
        ----------
        signal : array-like
            The raw signal to be baseline-corrected.
        npil : array-like
            The neuropil (NPIL) component of the raw signal.

        Returns
        -------
        tuple
            A tuple containing the baseline-corrected signal (baseline_total), the baseline signal (baseline_sig),
            and the baseline NPIL component (baseline_npil).

        Notes
        -----
        This function calculates the baseline signal by first subtracting the NPIL component from the raw signal.
        Then, it applies a max-min filter to the residual signal to remove high-frequency noise and extract
        the low-frequency baseline. Finally, it adds back the baseline NPIL component to obtain the total baseline-corrected
        signal.

        If the `constant_denominator` attribute is True, then the function will replace the baseline_total array with its median
        value, effectively setting a constant denominator in the signal processing."""

        sig_residual = signal - self.npil_coeff * npil

        baseline_sig = filters.maxmin_filter(
            sig_residual,
            window=self.window,
            sigma=self.sigma,
            min_periods=self.min_periods,
        )
        baseline_npil = filters.maxmin_filter(
            npil, window=self.window, sigma=self.sigma, min_periods=self.min_periods
        )

        baseline_total = baseline_sig + baseline_npil
        if self.constant_denominator:
            baseline_total = np.median(baseline_total)
        self.baseline = (baseline_total, baseline_sig, baseline_npil)

        return self.baseline

    def calculate(self, signal, npil):
        """Calculate dF/F

        Parameters
        ----------
        signal : array-like (n_rois, n_timepoints)
        npil : array-like (n_rois, n_timepoints)

        Returns
        -------
        dfof : pd.DataFrame
        """

        if len(signal.shape) != 2:
            raise ValueError("Input signal must be 2D")

        try:
            del self.baseline
        except AttributeError:
            pass

        base = self.calculate_baseline(signal, npil)
        return self._filter_final_signal(((signal - self.npil_coeff * npil) - base[1]) / base[0])


class JiaDFOF(DFOFStrategy, SlowTrendMixin):

    """Implements the dF/F calculation method of Jia et al 2010 (Nat. Protocols)
    with additional options to detrend slow changes.

    Parameters
    ----------
    t1 : int, optional
        Window size in frames for initial smoothing. A good value is ~3 sec
        of imaging time. Defaults to 90 frames.
    t2 : int, optional
        Window size in frames for rolling minimum baseline. A good value is ~60
        sec of imaging time. Defaults to 1800 frames.
    exp : float, optional
        Time constamt for exponential filtering of final trace.
        Not implemented
    slow_trend_window : int, optional
        If passed, the final trace is median-filtered in a rolling window of
        this size to remove slow changes (e.g. due to photobleaching). This
        is conventionally larger than the baseline window. Defaults to None
        (no detrending is applied).
    min_periods_t1, min_periods_t2, min_periods_slow : float (0, 1], optional
        Fraction of frames in rolling t1, t2, and slow_trend windows that must
        be non-NaN to return a non-NaN value.

    Attributes
    ----------
    baseline : pd.DataFrame
    slow_trend : pd.DataFrame

    Examples:
    Examples:
    >>> sig = s2p_mouse.true_signal(s2p_folder=s2p_folder)
    >>> dfof_strategy = JiaDFOF()
    >>> dfof_calc = dfof_strategy.calculate(signal=sig)
    """

    def __init__(
        self,
        t1=JIA_T1,
        t2=JIA_T2,
        exp=None,
        min_periods_t1=0.2,
        min_periods_t2=0.2,
        slow_trend_window=None,
        min_periods_slow=0.2,
    ):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.exp = exp
        self.min_periods_t1 = min_periods_t1
        self.min_periods_t2 = min_periods_t2
        self.slow_trend_window = slow_trend_window
        self.min_periods_slow = min_periods_slow
        self.slow_trend = None

    def _filter_final_signal(self, signal):
        """Filter with exponential kernel."""

        # # remove slow changes
        # if self.slow_trend is None:
        #     raise AttributeError("No slow trend to remove")

        return signal - self.calculate_slow_trend(
            signal, self.slow_trend_window, self.min_periods_slow
        )

    def _calculate_baseline(self, signal):
        # first smooth with t1 rolling window
        kwargs = {"center": True, "axis": 1}
        smooth_signal = (
            pd.DataFrame(signal)
            .rolling(
                window=self.t1, min_periods=int(self.min_periods_t1 * self.t1), **kwargs
            )
            .mean()
        )

        # then minimize with t2 rolling window
        return smooth_signal.rolling(
            window=self.t2, min_periods=int(self.min_periods_t2 * self.t2), **kwargs
        ).min()
