"""TBC"""
import pandas as pd
from mod.mod_plate import combine_plates
from mod.mod_helpers import compute_fft, get_stats
import gui_files.config as config

class FeatureExtractorBPPLines:
    
    def __init__(self, dl, verbose=False):
        self.dl = dl
        self.sensor_id = self.dl.sensor_id
        self.verbose = verbose
        print("\nExtracting features... (this might take a while)")
        self.n_snips_per_seg, self.len_snip = self.get_n_snips_per_seg()

        self.X_good_train = None
        self.X_good_threshold = None
        self.X_good_test = None
        self.X_bad_threshold = None
        self.X_bad_test = None
        self.X_all = None

        if self.dl.labels_file is not None:
            self.X_good_train, _ = self.process_data(dl.good_train_segments)
            self.X_good_threshold, self.df_good_threshold_locations = self.process_data(dl.good_threshold_segments)
            self.X_good_test, self.df_good_test_locations = self.process_data(dl.good_test_segments)
            self.X_bad_threshold, self.df_bad_threshold_locations = self.process_data(dl.bad_threshold_segments)
            self.X_bad_test, self.df_bad_test_locations = self.process_data(dl.bad_test_segments)
        else:
            all_segments = {
                plate.identifier: self.dl.segment_keys
                for plate in self.dl.list_bpp
            }
            self.X_all, self.df_all_locations = self.process_data(all_segments)

    def get_n_snips_per_seg(self):
        
        n_snips_per_seg = {}
        sampling_rates = set(bpp.sampling_rate for bpp in self.dl.list_bpp)
        assert len(sampling_rates) == 1, "All BPP plates should have the same sampling rate."
        len_snip = round(
            sampling_rates.pop() / 500 / config.SNIPPET_LENGTH_MM
        ) # sample points per snip = sampling_rate / speed (=500mm/s) / snip_len (=1mm)
        # This value is only approximate, what is enforced is n_snips_per_seg

        # Calculate number of snippets per segment
        for seg_key in self.dl.segment_keys:
            # Get all segments with this key
            relevant_segments = {
                (bpp.identifier,k): v for bpp in self.dl.list_bpp for k, v in bpp.segments.items() if k == seg_key
            }
            # Calculate lengths
            lengths = [len(v) for v in relevant_segments.values()]
            # Calculate number of snippets per segment
            if (max(lengths) - min(lengths)) < (len_snip * 0.1):
                # There might be other (better) ways to compute this value to maximize snippet length uniformity (and minimize overlaps or truncation)
                n_snips_per_seg[seg_key] = int(min(lengths) / len_snip)
            else:
                raise ValueError(
                    f"Length difference for segment {seg_key} is too large: {max(lengths)-min(lengths)}.",
                    f"The approximate length of a snippet is {len_snip} samples."
                )
        
        return n_snips_per_seg, len_snip

    def process_data(self, segments_dict):
        
        X = pd.DataFrame()
        df_locations = pd.DataFrame()

        for bpp in self.dl.list_bpp:
            for segment_key in segments_dict.get(bpp.identifier, []):
                # Combine data from the snippets: n_snip x snip_len x len(channels)
                X_time_norm, _, snip_len = combine_plates(
                    [bpp],
                    n_snip_per_seg=self.n_snips_per_seg[segment_key],
                    channels=bpp.data_channels,
                    norm_snip=True,
                    max_loss = max(self.n_snips_per_seg.values()), # To make sure that segments are truncated and snippets do not overlap
                    segments_subset=[segment_key],
                    return_snip_len = True,
                    verbose=self.verbose,
                )
                X_time_unnorm = combine_plates(
                    [bpp],
                    n_snip_per_seg=self.n_snips_per_seg[segment_key],
                    channels=bpp.data_channels,
                    norm_snip=False,
                    max_loss = max(self.n_snips_per_seg.values()), # To make "sure" that segments are truncated and snippets do not overlap
                    segments_subset=[segment_key],
                    verbose=self.verbose,
                )[0]
                X_spectrum = compute_fft(
                    X_time_unnorm,
                    n_channels=len(bpp.data_channels),
                    sampling_rate=bpp.sampling_rate,
                    remove_mean=True,
                )[0]
                X_segment = pd.concat(
                    [
                        get_stats(
                            X_spectrum,
                            channels=[channel + "_FFT" for channel in bpp.data_channels],
                            exclude_absolutes=True,
                            sampling_rate=bpp.sampling_rate,
                            is_fft=True,
                        ),
                        get_stats(
                            X_time_norm,
                            channels=bpp.data_channels,
                            exclude_absolutes=True,
                            sampling_rate=bpp.sampling_rate,
                        ),
                    ],
                    axis=1,
                )
                locations_segment = pd.DataFrame(
                    index = X_segment.index,
                    data = {
                        "plate": bpp.identifier,
                        "segment_type": segment_key[0],
                        "segment_number": segment_key[1],
                        "snippet": [f"{i+1}/{self.n_snips_per_seg[segment_key]}" for i in range(self.n_snips_per_seg[segment_key])],
                        "start_time": [
                            bpp.segments[segment_key]["Time"].iloc[i * snip_len]
                            for i in range(self.n_snips_per_seg[segment_key])
                        ],
                        "end_time": [
                            bpp.segments[segment_key]["Time"].iloc[(i + 1) * snip_len - 1]
                            for i in range(self.n_snips_per_seg[segment_key])
                        ],
                    }
                )
                # Concatenate the data
                if X.empty:
                    X = X_segment
                    df_locations = locations_segment
                else:
                    X = pd.concat([X, X_segment], axis=0, ignore_index=True)
                    df_locations = pd.concat([df_locations, locations_segment], axis=0, ignore_index=True)

        return X, df_locations
    