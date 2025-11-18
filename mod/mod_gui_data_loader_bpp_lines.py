"""TBC"""

import os
import random
import numpy as np
import pandas as pd
import gui_files.config as config
from mod.mod_plate import Plate
import gui_files.config as config

data_folder = os.path.join("data", config.BASE_DATA_FOLDER)

class DataLoaderBPPLines:
    """TBC"""

    def __init__(
        self,
        bpp_type,
        data_dir,
        random_seed=None,
        verbose=True
    ):
        #################################
        ### MODIFY HERE FOR REAL-TIME ###
        #################################

        self.bpp_type = bpp_type
        self.data_dir = data_dir
        self.data_path = os.path.join(data_folder,data_dir)
        self.random_seed = random_seed if random_seed is not None else random.randint(0, 1000)
        self.verbose = verbose

        labels_file = os.path.join(data_folder,config.LABELS_FOLDER,f"{self.data_dir}.xlsx")
        if os.path.exists(labels_file):
            self.labels_file = labels_file
        else:
            self.labels_file = None
            print(f"Warning: labels file {labels_file} not found. Proceeding without labels.\n")

        self.sensor_id = self.get_sensor()

        if self.bpp_type == "small":
            self.prefix_line_segments = ["aL", "hL", "dL", "kL", "bL"]
        elif self.bpp_type == "large":
            self.prefix_line_segments = ["RO", "RI", "RTO", "RTI", "RBO", "RBI", "RSH", "RMH", "RLH", "LO", "LI", "LBO", "LBI", "LTO", "LTI", "LSH", "LMH", "LLH", "VZ", "RZ", "LZ"]
        else:
            raise ValueError("bpp_type must be 'small' or 'large'.")
        
        self.list_bpp = self.load_plates()
        self.subset_plates = [bpp.identifier for bpp in self.list_bpp]
        # self.subset_type = "id"
        self.segment_keys = self.check_segments()
        if self.labels_file is not None:
            self.bad_segments, self.good_segments = self.get_bad_good_segments()
            self.bad_threshold_segments, self.bad_test_segments, self.good_train_segments, self.good_threshold_segments, self.good_test_segments = self.train_test_split()

    def get_sensor(self):
        data_files = os.listdir(self.data_path)

        # Check if all files are .csv
        if all(file.endswith('.csv') for file in data_files) and \
        all(os.path.splitext(file)[0].isdigit() for file in data_files):
            return "E001"

        # Check if files are .bin/.txt pairs
        bin_files = [file for file in data_files if file.endswith('.bin')]
        txt_files = [file for file in data_files if file.endswith('.txt')]

        if len(bin_files) > 0 and len(txt_files) > 0:
            # For .bin files: "data_yyyy-mm-dd_hh-mm-ss_n.bin"
            # For .txt files: "data_yyyy-mm-dd_hh-mm-ss_n_binheader.txt"
            bin_names = {os.path.splitext(f)[0] for f in bin_files}
            txt_names = {os.path.splitext(f)[0].replace("_binheader", "") for f in txt_files}
            if bin_names == txt_names:
                return "E005"

        raise ValueError("Files in data_path do not match expected formats for known sensors.")

    def load_plates(self):
        """TBC"""

        def find_nth_threshold_crossing(array, threshold, n):
            above_threshold = array > threshold
            # Find transitions: 1 = crossing from below, -1 = crossing from above
            transitions = np.diff(above_threshold.astype(int))
            # Indices where array crosses threshold from below (0 to 1 transition)
            crossings_from_below = np.where(transitions == 1)[0] + 1
            # Indices where array crosses threshold from above (1 to 0 transition)
            crossings_from_above = np.where(transitions == -1)[0] + 1
            assert np.all(
                crossings_from_below < crossings_from_above
            ), "Crossings from below should come before crossings from above."
            # Return first n crossings of each type
            return crossings_from_above[n - 1]

        if self.sensor_id == "E001" and self.bpp_type == "small":
            list_bpp = Plate.list_from_lwm_csv(
                weld_type="lines_bpp",
                folderpath=self.data_path,
                data_channels=["Back RR", "TempR", "Plasma R"],
                trigger_channel=None,
            )
            for bpp in list_bpp:
                index = find_nth_threshold_crossing(bpp.dataframe["OutputLL"], 5, 24)
                bpp.dataframe = bpp.dataframe.iloc[: index + 10]
                bpp.split_segments()
                bpp.segments = {k: v for k, v in bpp.segments.items() if k[0] in self.prefix_line_segments}

        # elif self.sensor_id == "E001_60":
        #     list_bpp = Plate.list_from_lwm_csv(
        #         weld_type="bpp",
        #         folderpath=self.data_path,
        #         data_channels=["Back RR", "TempR", "Plasma R"],
        #         trigger_channel=None,
        #         subset_list=self.subset_plates,
        #         subset_type=self.subset_type,
        #     )
        #     for bpp in list_bpp:
        #         bpp.split_segments(
        #             ref_channel="Back RR", remove_median=False, n_steps=30
        #         )
        #         bpp.segments = {k: v for k, v in bpp.segments.items() if k[0] in self.prefix_line_segments}

        elif self.sensor_id == "E005" and self.bpp_type == "small":
            list_bpp = Plate.list_from_empa_bin(
                weld_type="bpp",
                folderpath=self.data_path,
                data_channels=["BR_Laser", "OE_IR", "OE_Vis"],  # , "Mic1", "Mic2"],
                trigger_channel="Trigger",
            )
            for bpp in list_bpp:
                bpp.split_segments()
                bpp.segments = {k: v for k, v in bpp.segments.items() if k[0] in self.prefix_line_segments}

        elif self.sensor_id == "E005" and self.bpp_type == "large":
            list_bpp_1 = Plate.list_from_empa_bin(
                weld_type="long_bpp_1",
                folderpath=self.data_path,
                data_channels=["BR_Laser", "OE_IR", "OE_Vis"],  # , "Mic1", "Mic2"],
                trigger_channel="Trigger",
            )
            list_bpp_2 = Plate.list_from_empa_bin(
                weld_type="long_bpp_2",
                folderpath=self.data_path,
                data_channels=["BR_Laser", "OE_IR", "OE_Vis"],  # , "Mic1", "Mic2"],
                trigger_channel="Trigger",
            )
            assert len(list_bpp_1) == len(list_bpp_2)
            for bpp in list_bpp_1 + list_bpp_2:
                bpp.split_segments()
                bpp.segments = {k: v for k, v in bpp.segments.items() if k[0] in self.prefix_line_segments}
            list_bpp = [
                Plate.combine_long_bpp(part_1, part_2)
                for part_1, part_2 in zip(list_bpp_1, list_bpp_2)
            ]

        else:
            raise ValueError(f"Combination of sensor_id {self.sensor_id} and bpp_type {self.bpp_type} not recognized.")

        return list_bpp
    
    def check_segments(self):
        segments_0 = list(self.list_bpp[0].segments.keys())
        # Check if all other BPPs have the same segments
        for bpp in self.list_bpp[1:]:
            current_segments = bpp.segments.keys()
            if set(current_segments) != set(segments_0):
                raise ValueError(
                    f"BPP {bpp.identifier} has different segments than {self.list_bpp[0].identifier}. "
                    f"Expected: {segments_0}, Got: {current_segments}."
                )
        return segments_0

    def get_bad_good_segments(self):

        bad_segments_df = pd.read_excel(self.labels_file, header=0, index_col=0)
        bad_segments = bad_segments_df['bad_line_segments'].to_dict()
        bad_segments = {str(k): v for k, v in bad_segments.items()}
        for key, value in bad_segments.items():
            if isinstance(value, str):
                plate_segments = [seg.strip() for seg in value.split(",")]
                if self.bpp_type == "small":
                    plate_segments = [(f"{seg[0]}L", seg[-2:]) for seg in plate_segments if f"{seg[0]}L" in self.prefix_line_segments]
                elif self.bpp_type == "large":
                    plate_segments = [tuple(seg.split("_")) for seg in plate_segments if seg.split("_")[0] in self.prefix_line_segments]
                bad_segments[key] = plate_segments
            elif pd.isna(value):
                bad_segments[key] = []
            else:
                raise ValueError(f"Unexpected value type for plate {key}: value {value} of type {type(value)}")
        if self.bpp_type == "large":
            id_conversion = {id_logs: id_plus for (id_logs, id_plus) in zip(bad_segments.keys(), self.subset_plates)}
            bad_segments = {id_conversion[k]: v for k, v in bad_segments.items()}
        good_segments = {
            str(i): [segment_key for segment_key in self.segment_keys if segment_key not in bad_segments.get(str(i),[])]
            for i in self.subset_plates
        }
        return bad_segments, good_segments
    
    def train_test_split(self):

        n_good_segments = sum([len(v) for v in self.good_segments.values()])
        n_bad_segments = sum([len(v) for v in self.bad_segments.values()])
        # print(n_bad_segments, n_good_segments, len(self.list_bpp), len(self.segment_keys))
        assert n_bad_segments + n_good_segments == len(self.list_bpp) * len(self.segment_keys), \
            f"Total number {len(self.list_bpp) * len(self.segment_keys)} of segments does not \
                match the sum of {n_bad_segments} bad and {n_good_segments} good segments."
        
        n_bad_threshold_segments = n_bad_segments // 2
        n_bad_test_segments = n_bad_segments - n_bad_threshold_segments
        n_good_threshold_segments = n_bad_threshold_segments
        n_good_test_segments = n_bad_test_segments
        n_good_train_segments = n_good_segments - n_good_threshold_segments - n_good_test_segments

        def split_segments_dict(segments_dict, len_splits, preselect = None):
            all_segments = [(plate, seg) for plate, segs in segments_dict.items() for seg in segs]
            assert sum(len_splits) == len(all_segments), "Sum of lengths of splits does not match total number of segments."
            if preselect is not None:
                assert set(preselect).issubset(set(all_segments)), "Preselect list contains segments not in the original list."
                len_splits[-1] -= len(preselect)
                assert len_splits[-1] > 0, "Preselect list is too long."
                for item in preselect:
                    all_segments.remove(item)
            random.seed(self.random_seed)
            random.shuffle(all_segments)
            split_segment_dicts = []
            for idx, len_split in enumerate(len_splits):
                split_segment_dicts.append({})
                for plate, seg in all_segments[:len_split]:
                    split_segment_dicts[idx].setdefault(plate, []).append(seg)
                all_segments = all_segments[len_split:]
            if preselect is not None:
                for plate, seg in preselect:
                    split_segment_dicts[-1].setdefault(plate, []).append(seg)
            assert len(all_segments) == 0, "Not all segments were assigned to splits."
            return split_segment_dicts

        bad_threshold_segments, bad_test_segments = split_segments_dict(
            self.bad_segments,
            len_splits=[n_bad_threshold_segments, n_bad_test_segments],
        )
        good_train_segments, good_threshold_segments, good_test_segments = split_segments_dict(
            self.good_segments, len_splits=[n_good_train_segments, n_good_threshold_segments, n_good_test_segments]
        )

        if self.verbose:
            print(f"Number of good train segments: {n_good_train_segments}"
                f"\nNumber of good threshold segments: {n_good_threshold_segments}"
                f"\nNumber of good test segments: {n_good_test_segments}"
                f"\nNumber of bad threshold segments: {n_bad_threshold_segments}"
                f"\nNumber of bad test segments: {n_bad_test_segments}")
        
        return bad_threshold_segments, bad_test_segments, good_train_segments, good_threshold_segments, good_test_segments