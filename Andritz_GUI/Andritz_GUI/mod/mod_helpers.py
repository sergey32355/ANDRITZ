"""TBC"""

import warnings
import itertools
import numpy as np
import scipy as sp
import pandas as pd
from math import floor


# Compute timescale
def get_timescale(data: np.ndarray, sampling_rate: int = 1) -> np.ndarray:
    """
    Args:
        data (np.ndarray): array containing the relevant data;
            the last dimension must correspond to the timesteps.
        sampling_rate (int, optional): in samples per second. Defaults to 1.

    Returns:
        np.ndarray: sampling times in seconds.
    """
    if sampling_rate == 1:
        print("Computing timescale with sampling rate equal to 1.")
        # warnings.warn("Computing timescale with sampling rate equal to 1.")

    timescale = np.linspace(
        start=0,
        stop=data.shape[-1] / sampling_rate,
        num=data.shape[-1],
        endpoint=False,
    )

    return timescale


def reshape_data(data, n_channels):

    # Make sure that there are no unused dimensions (i.e., of length 1)
    data = data.squeeze()

    # Check the number of dimensions and add axes as needed
    if data.ndim == 1 and n_channels == 1:
        data = data.reshape(1, 1, -1)
    elif data.ndim == 2:
        if n_channels == data.shape[0]:
            data = data.reshape(1, *data.shape)
        elif n_channels == 1:
            data = data.reshape(data.shape[0], 1, data.shape[1])
    elif data.ndim == 3 and n_channels == data.shape[1]:
        pass
    else:
        raise ValueError(
            f"Array shape {data.shape} does not match with {n_channels} channels. "
            f"We expect an array of shape signals x channels x timesteps, "
            f"with dimensions collapsed when only one signal or channel is present."
        )

    return data


def moving_average(arr: np.ndarray, window_size: int = 100) -> np.ndarray:
    """TBC"""
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    kernel = np.ones(window_size) / window_size

    # Apply np.convolve for the middle part
    smoothed = np.convolve(arr, kernel, mode="same")

    # Handle edges explicitly
    for i in range(half_window):
        smoothed[i] = np.mean(arr[: i + half_window + 1])  # Start edge
        smoothed[-(i + 1)] = np.mean(arr[-(i + half_window + 1) :])  # End edge

    return smoothed


def compute_fft(
    data: np.ndarray,
    n_channels: int,
    sampling_rate: int,
    remove_mean: bool = False,
    window: None | str = None,  # "hann", "hamming", "blackman", "tukey",...
    log1p: bool = False,
) -> tuple[np.ndarray[float]]:
    """TBC"""

    data = reshape_data(data, n_channels)
    # FFT
    # n and d determine uniquely the sampled frequency bins, which are regularly spaced
    # from 0 to the Nyquist frequency sampling_rate/2 (and then negative)
    fft_freq = np.fft.fftfreq(n=data.shape[-1], d=1 / sampling_rate)
    # For real-valued signals, and if we are only interested in the magnitude,
    # we can discard the negative frequencies and their magnitudes, because the latter are
    # equal to the magnitudes at the corresponding positive frequency values
    fft_freq = fft_freq[: data.shape[-1] // 2 + 1]
    if window is not None:
        window_data = sp.signal.get_window(window, data.shape[-1])
        # Normalize to get out correct amplitude
        window_data = window_data / np.sum(window_data)
    # Before computing the FFT it can make sense to remove the signal's mean,
    # so that the DC component (magnitude at 0) does not dominate the spectrum
    processed_data = (
        data - np.mean(data, axis=-1, keepdims=True) if remove_mean else data
    ) * (window_data if window is not None else 1)
    # Compute FFT
    fft_ampl = np.fft.fft(processed_data, axis=-1)
    # Renormalize to get correct units
    fft_ampl = np.abs(fft_ampl)[..., : data.shape[-1] // 2 + 1] / data.shape[-1]
    # Double frequencies except DC and Nyquist (if it is present) for consistency
    fft_ampl[..., 1 : -((data.shape[-1] + 1) % 2)] *= 2
    if log1p:
        # Logarithmically scale the amplitudes
        fft_ampl = np.log1p(fft_ampl)
    # Reshape fft data for output
    data_fft = fft_ampl.reshape(-1, n_channels, data.shape[-1] // 2 + 1)
    return data_fft, fft_freq


def compute_cross_corr(
    signals_dict: dict[np.ndarray],
    output_threshold: float = 0.6,
    window_size: int = 1,
    mode: str = "manual",  # or "scipy" or "both"
    sampling_rate: int = 1000000,
    max_shift_frac: float = 0.2,
) -> None:
    """Compute the cross-correlation between all pairs of signals.
    "manual" has a more precise normalization, "scipy" is faster (uses FFT).
    The normalization of "scipy" favors constellations where the signals overlap a lot.

    Args:
        signals_dict (dict[np.ndarray]): Dictionary with all the relevant signals.
        output_threshold (float, optional): Min value of the cross-correlation to get an output. Defaults to 0.6.
        window_size (int, optional): If >1, averages the signal (new_length = old_length // window_size). Defaults to 1.
        mode (str, optional): Which computation to use. Either "manual", "scipy" or "both". Defaults to "manual".
        max_shift_frac (float, optional): For "manual" mode: max shift to consider, as frac of the longer signal. Defaults to 0.2.

    Returns:
        None. Get a printout of the results.
    """

    if window_size > 1:
        signal_dicts_list = [
            {
                "name": k,
                "data": np.array(
                    [
                        np.mean(
                            v[
                                idx : (
                                    idx + window_size
                                    if idx + window_size <= len(v)
                                    else len(v)
                                )
                            ]
                        )
                        for idx in range(0, len(v), window_size)
                    ]
                ),
            }
            for k, v in signals_dict.items()
        ]
    else:
        signal_dicts_list = [{"name": k, "data": v} for k, v in signals_dict.items()]

    for sig1, sig2 in itertools.combinations(signal_dicts_list, 2):

        # Make sure that sig1["data"] is not shorter
        if len(sig1["data"]) < len(sig2["data"]):
            sig1, sig2 = sig2, sig1

        # Mean-center the input arrays (and also convert lists or dataframes to arrays)
        long = np.array(sig1["data"] - np.mean(sig1["data"]))
        short = np.array(sig2["data"] - np.mean(sig2["data"]))

        # Manual computation
        if mode in {"manual", "both"}:

            max_shift_m = int(max_shift_frac * len(long))

            # Pad with zeros
            long_pad = np.pad(
                long, pad_width=max_shift_m, mode="constant", constant_values=0
            )

            # Compute cross-correlation
            cross_corr_m = np.array(
                [
                    # Shifted dot product
                    np.dot(
                        long_pad[max_shift_m + k : max_shift_m + k + len(short)], short
                    )
                    # Normalization (so that we get 1 for identical signals)
                    / np.sqrt(
                        np.sum(
                            long_pad[max_shift_m + k : max_shift_m + k + len(short)]
                            ** 2
                        )
                        * np.sum(
                            short[max(0, -k) : min(len(short), len(long) - k)] ** 2
                        )
                    )
                    for k in range(-max_shift_m, max_shift_m + 1)
                ]
            )

            # Output results only if the absolute maximum correlation exceeds the output threshold
            if np.max(cross_corr_m) >= output_threshold:

                shifts_m = np.argwhere(cross_corr_m == np.max(cross_corr_m)).flatten()
                shift_m = np.min(np.abs(shifts_m - max_shift_m))
                shift_ms_m = 1000 * window_size * shift_m / sampling_rate

                print("")
                print(
                    f"For channels {sig1["name"]} and {sig2["name"]} (in MANUAL mode):"
                )
                print("")
                print(
                    f">> Max correlation: {round(float(np.max(cross_corr_m)),3)}, shift: {round(shift_ms_m,3)} ms."
                )
                print("")

        # Scipy computation
        if mode in {"scipy", "both"}:

            max_shift_sp = len(short) // 2

            # Compute cross-correlation
            cross_corr_sp = sp.signal.correlate(
                long, short, mode="same", method="auto"
            ) / np.sqrt(np.sum(long**2) * np.sum(short**2))

            # Output results only if the absolute maximum correlation exceeds the output threshold
            if np.max(cross_corr_sp) >= output_threshold:

                shifts_sp = np.argwhere(
                    cross_corr_sp == np.max(cross_corr_sp)
                ).flatten()
                shift_sp = np.min(np.abs(shifts_sp - max_shift_sp))
                shift_ms_sp = 1000 * window_size * shift_sp / sampling_rate

                print("")
                print(
                    f"For channels {sig1["name"]} and {sig2["name"]} (in SCIPY mode):"
                )
                print("")
                print(
                    f">> Max correlation: {round(float(np.max(cross_corr_sp)),3)}, shift: {round(shift_ms_sp,3)} ms."
                )
                print("")

    return None


def get_stats(
    data: np.ndarray,
    channels: list[str],
    exclude_absolutes: bool = True,
    sampling_rate: int = 1,
    poly_deg: int = 3,
    is_fft: bool = False,
) -> pd.DataFrame:
    """Compute and return a DataFrame containing summary statistics of the given time signals.

    Args:
        data (np.ndarray): array containing the relevant data; the last dimension
            must correspond to the timesteps.
        channels (list[str]): list of the channel names.
        exclude_absolutes (bool, optional): whether to exclude the statistics that
            only pertain to the non-standardized signals. Defaults to True.
        sampling_rate (int, optional): in samples per second. Defaults to 1.
        poly_deg (int, optional): degree for polyfit. Defaults to 3.
            If sampling_rate is 1, it is computed based on the number of samples,
            ignoring actual time.

    Returns:
        pd.DataFrame: summary statistics.
    """

    # # Zero crossings (or rather above / below linear trend line??)
    # stats_dict["z_zcr"] = len(lib.zero_crossing_rate(z_signal, frame_length = len(z_signal)))
    # # Peak-to-peak distance
    # # Chaoticity

    data = reshape_data(data, len(channels))

    stats_dict = {}

    # Compute base stats
    stats_dict["mean"] = np.mean(data, axis=-1, keepdims=True)
    stats_dict["std"] = np.std(data, axis=-1, keepdims=True)
    stats_dict["skewness"] = sp.stats.skew(data, axis=-1)
    stats_dict["kurtosis"] = sp.stats.kurtosis(data, axis=-1)
    stats_dict["min"] = np.min(data, axis=-1)
    stats_dict["max"] = np.max(data, axis=-1)
    stats_dict["change"] = np.mean(np.abs(np.diff(data, axis=-1)), axis=-1)
    signs = np.sign(np.diff(data, axis=-1))
    stats_dict["dir_changes"] = np.sum(signs[...,1:] != signs[...,:-1], axis=-1) / (data.shape[-1]-2)

    # Standardize / compute z-scores
    z_data = (data - stats_dict["mean"]) / stats_dict["std"]

    # Compute standardized base stats
    stats_dict["z_min"] = np.min(z_data, axis=-1)
    stats_dict["z_max"] = np.max(z_data, axis=-1)
    stats_dict["z_change"] = stats_dict["change"] / stats_dict["std"].squeeze()
    stats_dict.update(
        [
            (f"z_per_{per}", np.percentile(z_data, q=per, axis=-1))
            for per in (25, 50, 75)
        ]
    )

    # Compute polyfit
    for deg in range(poly_deg + 1):
        stats_dict[f"poly_fit_{deg}"] = np.zeros(shape=data.shape[:-1])
    for idx in np.ndindex(data.shape[:-1]):  # Loop over all except the last dimension
        sub_z_data = z_data[idx]  # Get the innermost array slice
        for deg, coeff in enumerate(
            np.polyfit(get_timescale(sub_z_data, sampling_rate), sub_z_data, poly_deg)[
                ::-1
            ]
        ):
            stats_dict[f"poly_fit_{deg}"][idx] = coeff

    # # Numerically more stable:
    # from numpy.polynomial.polynomial import Polynomial
    # Polynomial.fit(time, data, deg).convert().coef

    # Compute spectral features
    if is_fft:
        # Make sure that also last freq is positive
        freqs = np.abs(np.fft.fftfreq(n=(data.shape[-1]-1)*2, d=1 / sampling_rate)[: data.shape[-1]])
        # Mean within given bands
        stats_dict["z_mean_freqs_below_1e3"] = np.mean(
            z_data[..., freqs < 1000], axis=-1
        )
        for exponent in range(3,floor(np.log10(max(freqs)))+1):
            stats_dict[f"z_mean_freqs_1e{exponent}_1e{exponent+1}"] = np.mean(
                z_data[..., (10**exponent <= freqs) & (freqs < 10**(exponent+1))], axis=-1
            )
        # Dominant frequency
        dominant_freqs = freqs[np.argmax(z_data, axis=-1)]
        stats_dict["log10_dominant_freq"] = np.where(
            dominant_freqs == 0, 0, np.log10(dominant_freqs)
        )
        # if np.isnan(stats_dict["log10_dominant_freq"]).any():
        #     print(dominant_freqs)
        #     print(stats_dict["log10_dominant_freq"])
        # Spectral entropy
        psd = data ** 2
        psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
        stats_dict["spectral_entropy"] = sp.stats.entropy(
            psd_norm, axis=-1, base=2
        )

    if exclude_absolutes:
        for key in ["mean", "std", "min", "max", "change"]:
            stats_dict.pop(key, None)
    else:
        stats_dict["mean"] = stats_dict["mean"].squeeze()
        stats_dict["std"] = stats_dict["std"].squeeze()

    # Prepare stats_df

    # Step 1: Combine all arrays horizontally along the columns
    combined_array = np.hstack(list(stats_dict.values()))

    # Step 2: Create column names by joining stat name with channel name
    if len(channels) > 1:
        column_names = [f"{channel}/{stat}" for stat in stats_dict for channel in channels]
    else:
        column_names = [stat for stat in stats_dict]

    # Step 3: Convert dict to DataFrame and sort columns by channel
    stats_df = pd.DataFrame(combined_array, columns=column_names)
    stats_df = stats_df.sort_index(axis=1)

    return stats_df

# Not in use currently
def measure_prominence(array, r=10):
    prominences = np.zeros(array.shape)
    for idx in np.ndindex(array.shape[:-1]):
        # Ensure idx is a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)
        for i in range(array.shape[-1]):
            start_idx = max(0, i - r)
            end_idx = min(array.shape[-1], i + r + 1)
            # Avoid empty slices
            left = array[idx + (slice(start_idx, i),)] if i > start_idx else np.array([])
            right = array[idx + (slice(i+1, end_idx),)] if i+1 < end_idx else np.array([])
            neighbors = np.concatenate([left, right], axis=-1) if left.size or right.size else np.array([])
            baseline = np.min(np.concatenate([neighbors, array[idx + (slice(i, i+1),)]], axis=-1), axis=-1, keepdims=True)
            neighbors -= baseline
            neighbor_mean = np.mean(neighbors, axis=-1, keepdims=True)
            if (neighbor_mean == 0).any():
                raise ValueError("Mean of neighbors is zero, cannot compute prominence.")
            prominences[idx + (slice(i, i+1),)] = (array[idx + (slice(i, i+1),)] + baseline) / neighbor_mean
    return prominences


def get_simple_stats(
    data: np.ndarray,
    channels: list[str],
) -> pd.DataFrame:
    """TBC"""

    data = reshape_data(data, len(channels))

    stats_dict = {}

    # Compute base stats
    stats_dict["mean"] = np.mean(data, axis=-1)
    stats_dict["std"] = np.std(data, axis=-1)
    stats_dict["min"] = np.min(data, axis=-1)
    stats_dict["max"] = np.max(data, axis=-1)
    stats_dict.update(
        [
            (f"z_per_{per}", np.percentile(data, q=per, axis=-1))
            for per in (25, 50, 75)
        ]
    )

    # Step 1: Combine all arrays horizontally along the columns
    combined_array = np.hstack(list(stats_dict.values()))

    # Step 2: Create column names by joining stat name with channel name
    if len(channels) > 1:
        column_names = [f"{channel}/{stat}" for stat in stats_dict for channel in channels]
    else:
        column_names = [stat for stat in stats_dict]

    # Step 3: Convert dict to DataFrame and sort columns by channel
    stats_df = pd.DataFrame(combined_array, columns=column_names)
    stats_df = stats_df.sort_index(axis=1)

    return stats_df
