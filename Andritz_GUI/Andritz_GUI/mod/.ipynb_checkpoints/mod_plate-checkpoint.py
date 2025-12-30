"""
TBC
"""

import os
import warnings
import datetime
from typing import Type
import pandas as pd
import numpy as np
import sqlite3
from scipy.ndimage import label, binary_dilation
from mod.mod_spcm_bin import read_bin_data
from mod.mod_exceptions import UndefinedException

# General settings
Z_DISTANCE = 500
# Plate-dependent
N_SEGMENTS = {"step_weld": 4, "cont_weld": 1, "triple": 3, "lines_bpp": 20, "bpp": 144, "long_bpp_1": 137, "long_bpp_2": 135, "long_bpp": 272}


class Plate:
    """See __init__()"""

    def __init__(
        self,
        identifier: str,
        date_time: datetime.datetime,
        weld_type: str,
        sensor: str,
        path: str,
        data_channels: list[str],
        trigger_channel: str | None,
        trigger_threshold: float | None,  # half of the trigger peak
        sampling_rate: float,
        duration: float | None,
        config: pd.DataFrame | None,
        dataframe: pd.DataFrame,
        n_segments: int,
        segments: dict[pd.DataFrame] | None = None,
        x_offset: float = 0,
        y_offset: float = 0,
        gap: float = 0,
    ) -> None:
        """_summary_

        Args:
            identifier (str): _description_
            date_time (datetime.datetime): _description_
            weld_type (str): _description_
            sensor (str): _description_
            path (str): _description_
            data_channels (list[str]): _description_
            trigger_channel (str | None): _description_
            trigger_threshold (float | None): For instance, half of the max trigger peak.
            sampling_rate (float): In samples per second.
            duration (float | None): In seconds.
            config (pd.DataFrame | None): _description_
            dataframe (pd.DataFrame): _description_
            n_segments (int): _description_
            segments (dict[pd.DataFrame] | None, optional): _description_. Defaults to None.
            x_offset (float, optional): _description_. Defaults to 0.
            y_offset (float, optional): _description_. Defaults to 0.
            gap (float, optional): _description_. Defaults to 0.
        """

        self.identifier = identifier
        self.date_time = date_time
        self.weld_type = weld_type
        self.sensor = sensor
        self.path = path
        self.data_channels = data_channels
        self.trigger_channel = trigger_channel
        self.trigger_threshold = trigger_threshold
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.config = config
        self.dataframe = dataframe
        self.n_segments = n_segments
        self.segments = segments
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.gap = gap
        print(f"Loaded plate {self.identifier}.")# with weld type {self.weld_type}.")

    @classmethod
    def list_from_empa_bin(
        cls: Type["Plate"],
        weld_type: str,
        folderpath: str,
        data_channels: list[str] | None = [
            "BR_Laser",
            "OE_IR",
            "OE_Vis",
            "Mic1",
            "Mic2",
        ],
        trigger_channel: str | None = "Trigger",
        trigger_threshold: float | None = 5,
        x_offsets: list[float] | None = None,
        y_offsets: list[float] | None = None,
        gaps: list[float] | None = None,
        subset_list: list[int] | None = None,
        subset_type: str | None = None,
    ) -> list["Plate"]:
        """
        TBC
        """

        if subset_list is None and weld_type in ["long_bpp_1", "long_bpp_2"]:
            n_long_plates = len(os.listdir(folderpath))//4
            if weld_type == "long_bpp_1":
                subset_list = 2 * np.array(range(n_long_plates))
            elif weld_type == "long_bpp_2":
                subset_list = 2 * np.array(range(n_long_plates)) + 1
            subset_type = "index"

        filenames, x_offsets, y_offsets, gaps = get_filenames_offsets_gaps(
            folderpath=folderpath,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            gaps=gaps,
            subset_list=subset_list,
            subset_type=subset_type,
        )

        # Get dictionary containing path of data and settings for each filename
        files = {
            filename: {
                "data": os.path.join(folderpath, f"{filename}.bin"),
                "settings": os.path.join(folderpath, f"{filename}_binheader.txt"),
            }
            for filename in filenames
        }

        plates_dict = {}

        for idx, filename in enumerate(filenames):

            date, time = filename.split("_")[1:3]
            date_time = f"{date}-{time}"

            plate_df, read_data_channels, sampling_rate = read_bin_data(
                settings_file=files[filename]["settings"],
                data_file=files[filename]["data"],
                subset_data_channels=data_channels,
                trigger_channel=trigger_channel,
            )

            plates_dict[filename] = cls(
                identifier=filename.split("_")[-1],
                date_time=datetime.datetime.strptime(
                    date_time.strip(), "%Y-%m-%d-%H-%M-%S"
                ),
                weld_type=weld_type,
                sensor="empa",
                path=os.path.join(folderpath, filename),
                data_channels=read_data_channels,
                trigger_channel=trigger_channel,
                trigger_threshold=trigger_threshold,
                sampling_rate=sampling_rate,
                duration=len(plate_df) / sampling_rate,
                config=None,
                dataframe=plate_df,
                n_segments=int(N_SEGMENTS[weld_type]),
                x_offset=x_offsets[idx],
                y_offset=y_offsets[idx],
                gap=gaps[idx],
            )

        plates_list = list(plates_dict.values())

        return plates_list

    @classmethod
    def list_from_lwm_csv(
        cls: Type["Plate"],
        weld_type: str,
        folderpath: str,
        data_channels: list[str] | None = None,
        trigger_channel: str | None = None,
        trigger_threshold: float | None = 6,
        x_offsets: list[float] | None = None,
        y_offsets: list[float] | None = None,
        gaps: list[float] | None = None,
        subset_list: list[int] | None = None,
        subset_type: str | None = None,
    ) -> list["Plate"]:
        """
        TBC
        Works but is not very logical
        """

        if subset_list is None and weld_type in ["long_bpp_1", "long_bpp_2"]:
            n_long_plates = len(os.listdir(folderpath))//4
            if weld_type == "long_bpp_1":
                subset_list = 2 * np.array(range(n_long_plates))
            elif weld_type == "long_bpp_2":
                subset_list = 2 * np.array(range(n_long_plates)) + 1
            subset_type = "index"

        filenames, x_offsets, y_offsets, gaps = get_filenames_offsets_gaps(
            folderpath=folderpath,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            gaps=gaps,
            subset_list=subset_list,
            subset_type=subset_type,
        )

        plates_dict = {}

        for idx, filename in enumerate(filenames):

            filepath = os.path.join(folderpath, f"{filename}.csv")

            # Read config info from first 9 lines of file
            with open(filepath, "r", encoding="utf-8") as file:
                # Reads first 9 lines to list and removes newline characters
                config_lines = [file.readline().strip() for _ in range(9)]
                # Convert the list of config lines into a DataFrame with one column
                config = pd.DataFrame(config_lines, columns=["one_str"])
                # Split along first occurrence of ":" separator
                config = pd.DataFrame(
                    [i.split(":", 1) for i in config.one_str],
                    columns=["Config", "Value"],
                )

            # Read signal data from file
            df = pd.read_csv(
                filepath,
                sep=";",
                header=9,
                index_col=0,
            )

            standard_channels = {
                "Analog": [
                    "T1-Raw",
                    "T1-Raw.1",
                    "T2-Raw",
                    "VIS1-RAW",
                    "VIS2-Raw",
                    "VIS3-Raw",
                    "VIS4-Raw",
                ],  # ['RR', 'T1', 'T2', 'VIS1', 'VIS2', 'VIS3', 'VIS4'],#
                "OutputLL": ["Back RR", "TempR", "Plasma R"],
            }

            available_channels = [
                channel for channel in list(df.columns) if channel != "Time"
            ]

            if "Analog" in available_channels:
                trigger_channel = "Analog"
            elif "OutputLL" in available_channels:
                trigger_channel = "OutputLL"
            if trigger_channel is not None:
                assert set(standard_channels[trigger_channel]).issubset(
                    set(available_channels)
                ), "Some standard channels are not available."
                if data_channels is None:
                    data_channels = standard_channels[trigger_channel]
                else:
                    assert set(data_channels).issubset(
                        set(available_channels)
                        & set(standard_channels[trigger_channel])
                    ), "Some user data channels are not available or not standard."
                df = df[["Time"] + [trigger_channel] + data_channels]

            plates_dict[filename] = cls(
                identifier=filename,
                date_time=datetime.datetime.strptime(
                    config.loc[7, :].Value.strip(), "%m/%d/%Y %H:%M:%S %p"
                ),
                weld_type=weld_type,
                sensor="lwm",
                path=filepath,
                data_channels=data_channels,
                trigger_channel=trigger_channel,
                trigger_threshold=trigger_threshold,
                sampling_rate=float(config.loc[2, :].Value.strip().split(" ")[0])
                * 1000,
                duration=float(config.loc[1, :].Value.strip().split(" ")[0]),
                config=config,
                dataframe=df,
                n_segments=int(N_SEGMENTS[weld_type]),
                x_offset=x_offsets[idx],
                y_offset=y_offsets[idx],
                gap=gaps[idx],
            )

        plates_list = list(plates_dict.values())

        return plates_list

    @classmethod
    def list_from_fdtwo_db(
        cls: Type["Plate"],
        weld_type: str,
        folderpath: str,
        ddc: int,
        user_data_channels: list[str] | None = None,
        user_trigger_channel: str | None = None,
        user_trigger_threshold: float | None = None,
        x_offsets: list[float] | None = None,
        y_offsets: list[float] | None = None,
        gaps: list[float] | None = None,
        subset_list: list[int] | None = None,
        subset_type: str | None = None,
    ) -> list["Plate"]:
        """TBC"""

        if subset_list is None and weld_type in ["long_bpp_1", "long_bpp_2"]:
            n_long_plates = len(os.listdir(folderpath))//4
            if weld_type == "long_bpp_1":
                subset_list = 2 * np.array(range(n_long_plates))
            elif weld_type == "long_bpp_2":
                subset_list = 2 * np.array(range(n_long_plates)) + 1
            subset_type = "index"

        filenames, x_offsets, y_offsets, gaps = get_filenames_offsets_gaps(
            folderpath=folderpath,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            gaps=gaps,
            subset_list=subset_list,
            subset_type=subset_type,
        )

        def get_df_from_db(path: str) -> pd.DataFrame:
            # query1 = "SELECT name FROM sqlite_master WHERE type='table';"
            # with sqlite3.connect(path) as conn:
            #     names = pd.read_sql_query(query1, conn)
            # print(names)
            query = "SELECT * FROM Messwerte;"
            with sqlite3.connect(path) as conn:
                df = pd.read_sql_query(query, conn)
            return df

        def get_time(string: str) -> float:
            string = string.split("_")[1]
            # Parse the string
            hours, minutes = string.split("h")
            minutes, seconds = minutes.split("m")
            seconds = float(seconds)
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + seconds
            return total_seconds

        plates_dict = {}

        for idx, filename in enumerate(filenames):

            filepath = os.path.join(
                folderpath, filename, filename.split("#")[0] + f"ddc{ddc}.db"
            )
            identifier = filename.split("#")[-1]

            # Get df, available data and trigger channels
            temp_trigger_channel = user_trigger_channel
            temp_trigger_threshold = user_trigger_threshold
            data_channels = []
            df = get_df_from_db(filepath)
            # Clean up dataframe
            df["Time"] = df["time"].map(get_time)
            df["Time"] -= df["Time"][0]
            # adc channels are just noise; chan13-15 are either triggers or almost always saturated
            df = df.drop(
                [
                    "id",
                    "time",
                    "packet_nr",
                    "status",
                    "systime4D",
                    "temp",
                    "humidity",
                    "pressure",
                    "MAC",
                    "IP",
                    "adc4",
                    "adc10",
                    "chan13",
                    "chan14",
                    "chan15",
                ],
                axis=1,
            )
            # Keep only active channels
            df = df.loc[:, (df != 0).any()]
            # Check positivity and integrity
            assert (df >= 0).all().all()
            assert (df.drop(columns=["Time"]).dtypes == "int64").all().all()
            # Determine available channels
            available_channels = list(df.drop(columns=["Time"]).columns)

            # Determine what to use as data channels, trigger channel, trigger threshold
            if user_data_channels is None:
                data_channels = [
                    channel
                    for channel in available_channels
                    if channel != temp_trigger_channel
                ]
            else:
                data_channels = [
                    channel
                    for channel in user_data_channels
                    if (
                        channel in available_channels
                        and channel != user_trigger_channel
                    )
                ]
                missing_channels = [
                    channel
                    for channel in user_data_channels
                    if channel not in available_channels
                ]
                if missing_channels:
                    warnings.warn(
                        f"The following user-specified data channels are not available and will be ignored: {missing_channels}",
                        UserWarning,
                    )
            if (
                user_trigger_channel is None
                or user_trigger_channel not in available_channels
            ):
                temp_trigger_channel = None
                temp_trigger_threshold = None
                if user_trigger_channel is not None:
                    warnings.warn(
                        "Selected trigger channel is not available, using None instead",
                        UserWarning,
                    )
            elif temp_trigger_threshold is None:
                temp_trigger_threshold = max(df[temp_trigger_channel]) / 2

            # Clean up df
            if temp_trigger_channel is not None:
                assert temp_trigger_threshold is not None, "Trigger threshold is None"
                trigger_channel = "Trigger"
                trigger_threshold = temp_trigger_threshold
                df = df.rename({temp_trigger_channel: trigger_channel}, axis=1)
                df = df[["Time", trigger_channel] + data_channels]
            else:
                df = df[["Time"] + data_channels]
                trigger_channel = None
                trigger_threshold = None

            # Get duration of signal in seconds
            duration = df.loc[len(df) - 1, "Time"] - df.loc[0, "Time"]
            # Get sampling frequency in Hz (round to multiple of 100)
            sampling_rate = round(len(df) / duration / 100) * 100.0

            plates_dict[identifier] = cls(
                identifier=identifier,
                date_time=datetime.datetime.strptime(
                    filename.split(".")[0].strip(), "%Y-%m-%d_%Hh%Mm%S"
                ),
                weld_type=weld_type,
                sensor="fdtwo",
                path=filepath,
                data_channels=data_channels,
                trigger_channel=trigger_channel,
                trigger_threshold=trigger_threshold,
                sampling_rate=sampling_rate,
                duration=duration,
                config=None,
                dataframe=df,
                n_segments=int(N_SEGMENTS[weld_type]),
                x_offset=x_offsets[idx],
                y_offset=y_offsets[idx],
                gap=gaps[idx],
            )

        plates_list = list(plates_dict.values())

        # Check that all plates have the same data_channels, trigger_channel, trigger_threshold, and sampling_rate
        attributes_to_check = [
            "trigger_channel",
            "trigger_threshold",
            "data_channels",
            "sampling_rate",
        ]
        for attribute in attributes_to_check:
            attribute_set = {
                (
                    frozenset(getattr(plate, attribute))
                    if isinstance(getattr(plate, attribute), list)
                    else getattr(plate, attribute)
                )
                for plate in plates_list
            }
            if len(attribute_set) > 1:
                warnings.warn(
                    f"Not all plates have the same {attribute}: {attribute_set}.",
                    UserWarning,
                )
                if attribute == "trigger_channel":
                    print("No common trigger found, defaulting to None")
                    for plate in plates_list:
                        plate.trigger_channel = None
                        plate.trigger_threshold = None
                        if "Trigger" in df.columns:
                            plate.dataframe = plate.dataframe.drop(columns=["Trigger"])
                elif attribute == "data_channels":
                    channels_intersection = sorted(
                        list(set.intersection(*(set(t) for t in attribute_set)))
                    )
                    assert (
                        channels_intersection
                    ), "No data channel is shared among all plates"
                    print(
                        f"Using only the {len(data_channels)} data channels shared among all plates: {channels_intersection}"
                    )
                    trigger_column = (
                        [plates_list[0].trigger_channel]
                        if plates_list[0].trigger_channel is not None
                        else []
                    )
                    for plate in plates_list:
                        plate.dataframe = plate.dataframe[
                            ["Time", *trigger_column, *channels_intersection]
                        ]
                        plate.data_channels = channels_intersection

        return plates_list
    
    @classmethod
    def combine_long_bpp(
        cls: Type["Plate"],
        part_1: "Plate",
        part_2: "Plate",
    ) -> "Plate":
        """
        TBC
        """
        complete_plate = cls(
            identifier=f"{part_1.identifier}+{part_2.identifier}",
            date_time=part_1.date_time,
            weld_type="long_bpp",
            sensor=part_1.sensor,
            path=part_1.path,
            data_channels=part_1.data_channels,
            trigger_channel=part_1.trigger_channel,
            trigger_threshold=part_1.trigger_threshold,
            sampling_rate=part_1.sampling_rate,
            duration=part_1.duration + part_2.duration,
            config=None,
            dataframe=pd.DataFrame(),
            n_segments=int(N_SEGMENTS["long_bpp"]),
            x_offset=0,
            y_offset=0,
            gap=0,
        )
        complete_plate.segments = part_1.segments | part_2.segments

        return complete_plate

    def plot_channels(
        self,
        separate_channels: bool = False,
        by_segment: bool = True,
        plot_every: int = 1,
        include_trigger: bool = False,
        axis=None,
        figsize=None,
        subset_segment_keys: list | None = None,
    ) -> None:
        # TO DO: just use standard kwargs as in any plot
        """
        TBC
        """
        plot_kwargs = {
            "x": "Time",
            "xlabel": "Time [s]",
            "ylabel": "Amplitude [V]",
            "figsize": (10, 5) if figsize is None else figsize,
            "ax": axis,
        }
        if by_segment and self.segments is not None:
            if subset_segment_keys is not None:
                subset_segments_dict = {
                    k: v for k, v in self.segments.items() if k in subset_segment_keys
                }
            else:
                subset_segments_dict = self.segments
            for idx, segment in subset_segments_dict.items():
                segment[::plot_every].plot(
                    y=(
                        self.data_channels + [self.trigger_channel]
                        if include_trigger
                        else self.data_channels
                    ),
                    title=f"segment {idx}/{self.n_segments} "
                    f"from plate {self.identifier} of type {self.weld_type}",
                    **plot_kwargs,
                )
                if separate_channels:
                    for channel in (
                        self.data_channels + [self.trigger_channel]
                        if include_trigger
                        else self.data_channels
                    ):
                        segment[::plot_every].plot(
                            y=channel,
                            title=f"channel {channel} of segment {idx}/{self.n_segments} "
                            f"from plate {self.identifier} of type {self.weld_type}",
                            **plot_kwargs,
                        )
        else:
            self.dataframe[::plot_every].plot(
                y=(
                    self.data_channels + [self.trigger_channel]
                    if include_trigger
                    else self.data_channels
                ),
                title=f"{self.weld_type} {self.identifier}",
                **plot_kwargs,
            )
            if separate_channels:
                for channel in (
                    self.data_channels + [self.trigger_channel]
                    if include_trigger
                    else self.data_channels
                ):
                    self.dataframe[::plot_every].plot(
                        y=channel,
                        title=f"channel {channel} from {self.weld_type} {self.identifier}",
                        **plot_kwargs,
                    )

    def split_segments(
        self,
        ref_channel: str | None = None,
        n_steps: int = 10,
        remove_at_boundary_seconds: tuple[float] = (0.0001,0.0001),  # corresponds to 50um of weld on each side
        remove_median: bool = True,
    ) -> None:
        """
        TBC
        """

        if self.segments is not None:
            raise ValueError(
                f"Plate {self.identifier} of type {self.weld_type} was "
                f"already split into {len(self.segments)} segments."
            )
        
        if self.dataframe.isna().any().any():
            warnings.warn(
                f"Plate {self.identifier} of type {self.weld_type} has NaN values in the dataframe."
            )
            self.dataframe = self.dataframe.dropna(axis=0, how="any", inplace=False, ignore_index=False)

        def found_split_margins(
            channel: str, n_steps: int, threshold: int | None = None, remove_median: bool = True
        ):

            iterative_thr = (
                1
                / n_steps
                * (np.max(self.dataframe[channel]) - int(remove_median)*np.median(self.dataframe[channel]))
            )
            # Empirical value. Works only for signals with:
            # * a long period of noise pre- and post- signal; (or a ground level at 0)
            # * a good enough separation in amplitude between noise and signal;
            # * consistently high signal when active (not just spike and then low)

            if threshold is None:
                threshold = int(remove_median)*np.median(self.dataframe[channel]) + iterative_thr

            start_indices = np.zeros(self.n_segments + 1)
            iteration = 1

            while len(start_indices) != self.n_segments and iteration < n_steps:

                # Determine start and end of segments
                # Using NumPy for faster comparison
                above_threshold = self.dataframe[channel].values > threshold

                # Using np.where to find the indices of transitions (faster than .diff())
                # Start indices are inside the interval, stop indices are outside
                start_indices = (
                    np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
                )
                stop_indices = (
                    np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1
                )
                # print(len(start_indices), len(stop_indices))

                # print(iterative_thr, threshold, channel, n_steps, len(start_indices), iteration)

                # Make sure that the first and last segments are also considered
                if self.dataframe[channel].values[0] > threshold:
                    start_indices = np.insert(start_indices, 0, 0)
                if self.dataframe[channel].values[-1] > threshold:
                    stop_indices = np.append(stop_indices, len(self.dataframe))
                # print(len(start_indices), len(stop_indices))

                # Make sure that there are no false positives (quick bursts)
                idx = 0
                while idx < len(start_indices):
                    if stop_indices[idx] - start_indices[idx] < max(2,self.sampling_rate/20000):  # arbitrary value
                        start_indices = np.delete(start_indices, idx)
                        stop_indices = np.delete(stop_indices, idx)
                    else:
                        idx += 1

                threshold += iterative_thr
                iteration += 1
                # print(len(start_indices), len(stop_indices), n_steps, threshold)

            # The BPP program on LWM has 4 extra trigger bursts at the start
            if self.weld_type in ("bpp", "lines_bpp") and len(start_indices) == self.n_segments+4:
                start_indices = start_indices[4:]
                stop_indices = stop_indices[4:]

            # The long BPP program has 2 extra trigger bursts at the start
            elif self.weld_type == "long_bpp_1" and len(start_indices) == self.n_segments+2:
                start_indices = start_indices[2:]
                stop_indices = stop_indices[2:]

            # print(len(start_indices), len(stop_indices), n_steps, threshold)

            # Ensure that there is the right amount of start and stop positions
            if (
                len(start_indices) != self.n_segments
                or len(stop_indices) != self.n_segments
            ):
                # warnings.warn(
                #     f"Could not get transition points with channel {channel}: "
                #     f"({self.n_segments},{self.n_segments}) expected,"
                #     f" got ({len(start_indices)},{len(stop_indices)}).",
                #     UserWarning,
                # )
                return False

            # Ensure that the length difference between the segments is at most 10%
            segment_lengths = [
                stop_indices[i] - start_indices[i] for i in range(self.n_segments)
            ]
            if self.weld_type in ("step_weld", "cont_weld", "triple") and \
            max(segment_lengths) - min(segment_lengths) >= 0.1 * min(segment_lengths):
                return False

            return start_indices, stop_indices

        margins = tuple()
        if ref_channel is not None:
            margins = found_split_margins(
                channel=ref_channel,
                n_steps=n_steps,
                remove_median=remove_median,
            )
        elif self.trigger_channel is not None:
            margins = found_split_margins(
                channel=self.trigger_channel,
                n_steps=2,
                threshold=self.trigger_threshold,
                remove_median=remove_median,
            )
        else:
            # warnings.warn(
            #     "No reference channel or trigger channel was provided for "
            #     "splitting the segments; trying all channels in sequence",
            #     UserWarning,
            # )
            sorted_channels = (
                self.dataframe[self.data_channels]
                .max()
                .sort_values(ascending=False)
                .index
            )
            for channel in sorted_channels:
                if not margins:
                    margins = found_split_margins(
                        channel=channel,
                        n_steps=n_steps,
                        remove_median=remove_median,
                    )
                else:
                    break
        if not margins:
            raise UndefinedException("Could not split signal!")

        # Split the DataFrame into segments
        start_indices, stop_indices = margins
        remove_at_boundary_samples = (int(remove_at_boundary_seconds[0] * self.sampling_rate), int(remove_at_boundary_seconds[1] * self.sampling_rate))
        if self.weld_type in ("step_weld", "cont_weld", "triple"):
            segment_keys = tuple(i + 1 for i in range(self.n_segments))
        elif self.weld_type == "lines_bpp":
            segment_keys = (
                ("aL", "01"),
                ("aL", "02"),
                ("aL", "03"),
                ("aL", "04"),
                ("hL", "03"),
                ("hL", "04"),
                ("hL", "01"),
                ("hL", "02"),
                ("dL", "03"),
                ("dL", "04"),
                ("dL", "01"),
                ("dL", "02"),
                ("kL", "03"),
                ("kL", "04"),
                ("kL", "01"),
                ("kL", "02"),
                ("bL", "03"),
                ("bL", "04"),
                ("bL", "01"),
                ("bL", "02"),
            )
        elif self.weld_type == "bpp":
            segment_keys = (
                ("aL", "01"),
                ("aL", "02"),
                ("aL", "03"),
                ("aL", "04"),
                ("hL", "03"),
                ("hL", "04"),
                ("hL", "01"),
                ("hL", "02"),
                ("dL", "03"),
                ("dL", "04"),
                ("dL", "01"),
                ("dL", "02"),
                ("kL", "03"),
                ("kL", "04"),
                ("kL", "01"),
                ("kL", "02"),
                ("bL", "03"),
                ("bL", "04"),
                ("bL", "01"),
                ("bL", "02"),
                ("hC", "24"),
                ("hC", "23"),
                ("hC", "22"),
                ("hC", "21"),
                ("hC", "20"),
                ("hC", "19"),
                ("hC", "18"),
                ("hC", "17"),
                ("hC", "16"),
                ("hC", "15"),
                ("hC", "14"),
                ("hC", "13"),
                ("hC", "12"),
                ("hC", "11"),
                ("dC", "24"),
                ("dC", "23"),
                ("dC", "22"),
                ("dC", "21"),
                ("dC", "20"),
                ("dC", "19"),
                ("dC", "18"),
                ("dC", "17"),
                ("dC", "16"),
                ("dC", "15"),
                ("dC", "14"),
                ("dC", "13"),
                ("dC", "12"),
                ("dC", "11"),
                ("kC", "22"),
                ("kC", "21"),
                ("kC", "20"),
                ("kC", "19"),
                ("kC", "18"),
                ("kC", "17"),
                ("kC", "16"),
                ("kC", "15"),
                ("kC", "14"),
                ("kC", "13"),
                ("kC", "12"),
                ("kC", "11"),
                ("bC", "22"),
                ("bC", "21"),
                ("bC", "20"),
                ("bC", "19"),
                ("bC", "18"),
                ("bC", "17"),
                ("bC", "16"),
                ("bC", "15"),
                ("bC", "14"),
                ("bC", "13"),
                ("bC", "12"),
                ("bC", "11"),
                ("iC", "18"),
                ("iC", "17"),
                ("iC", "16"),
                ("iC", "15"),
                ("iC", "14"),
                ("iC", "13"),
                ("iC", "12"),
                ("iC", "11"),
                ("iC", "24"),
                ("iC", "23"),
                ("iC", "22"),
                ("iC", "21"),
                ("iC", "20"),
                ("iC", "19"),
                ("cC", "21"),
                ("cC", "20"),
                ("cC", "19"),
                ("cC", "18"),
                ("cC", "17"),
                ("cC", "16"),
                ("cC", "15"),
                ("cC", "14"),
                ("cC", "13"),
                ("cC", "12"),
                ("cC", "11"),
                ("cC", "24"),
                ("cC", "23"),
                ("cC", "22"),
                ("eB", "38"),
                ("eB", "33"),
                ("eB", "28"),
                ("eB", "23"),
                ("eB", "18"),
                ("eB", "13"),
                ("gB", "38"),
                ("gB", "33"),
                ("gB", "28"),
                ("gB", "23"),
                ("gB", "18"),
                ("gB", "13"),
                ("fS", "78"),
                ("fS", "74"),
                ("fS", "69"),
                ("fS", "65"),
                ("fS", "60"),
                ("fS", "56"),
                ("fS", "51"),
                ("fS", "52"),
                ("fS", "61"),
                ("fS", "70"),
                ("fS", "79"),
                ("fS", "75"),
                ("fS", "66"),
                ("fS", "57"),
                ("fS", "53"),
                ("fS", "62"),
                ("fS", "71"),
                ("fS", "80"),
                ("fS", "76"),
                ("fS", "67"),
                ("fS", "58"),
                ("fS", "54"),
                ("fS", "63"),
                ("fS", "72"),
                ("fS", "81"),
                ("fS", "82"),
                ("fS", "77"),
                ("fS", "73"),
                ("fS", "68"),
                ("fS", "64"),
                ("fS", "59"),
                ("fS", "55"),
            )
        elif self.weld_type == "long_bpp_1":
            segment_keys = (
                ('RTO', '9'),
                ('RSH', '5'),
                ('RMH', '3'),
                ('RI', '1'),
                ('RO', '1'),
                ('RO', '2'),
                ('RZ', '2'),
                ('RI', '2'),
                ('RBI', '7'),
                ('RLH', '7'),
                ('RLH', '6'),
                ('RO', '4'),
                ('RZ', '4'),
                ('RO', '5'),
                ('RBO', '9'),
                ('RBO', '8'),
                ('RBO', '7'),
                ('RBI', '6'),
                ('RLH', '9'),
                ('RLH', '2'),
                ('RLH', '3'),
                ('RLH', '4'),
                ('RLH', '5'),
                ('RMH', '4'),
                ('RMH', '5'),
                ('RMD', '6'),
                ('RMD', '5'),
                ('RMD', '4'),
                ('RMD', '3'),
                ('RMD', '2'),
                ('RMD', '1'),
                ('RMH', '1'),
                ('RMH', '2'),
                ('RSH', '6'),
                ('RSH', '7'),
                ('RSH', '2'),
                ('RSH', '3'),
                ('RSH', '4'),
                ('RTI', '7'),
                ('RZ', '1'),
                ('RO', '3'),
                ('RZ', '3'),
                ('RLH', '8'),
                ('RBI', '5'),
                ('RBI', '4'),
                ('RBI', '3'),
                ('RBO', '3'),
                ('RBO', '4'),
                ('RBO', '5'),
                ('RBO', '6'),
                ('RLH', '1'),
                ('S5', '19'),
                ('S5', '18'),
                ('S5', '17'),
                ('S5', '16'),
                ('S5', '15'),
                ('S5', '14'),
                ('S5', '13'),
                ('S5', '12'),
                ('S5', '11'),
                ('S5', '10'),
                ('S5', '09'),
                ('S5', '08'),
                ('S5', '07'),
                ('S5', '06'),
                ('S5', '05'),
                ('S5', '04'),
                ('S5', '03'),
                ('S5', '02'),
                ('S5', '01'),
                ('RSH', '1'),
                ('RTO', '6'),
                ('RTO', '5'),
                ('RTI', '5'),
                ('RTI', '6'),
                ('RTO', '8'),
                ('RTO', '7'),
                ('RSD', '1'),
                ('RSD', '2'),
                ('RSD', '3'),
                ('RSD', '4'),
                ('RTI', '4'),
                ('RTI', '3'),
                ('RTI', '2'),
                ('RTI', '1'),
                ('RTO', '2'),
                ('RTO', '3'),
                ('RTO', '4'),
                ('VZ', '1'),
                ('S4', '01'),
                ('S4', '02'),
                ('S4', '03'),
                ('S4', '04'),
                ('S4', '05'),
                ('S4', '06'),
                ('S4', '07'),
                ('S4', '08'),
                ('S4', '09'),
                ('S4', '10'),
                ('S4', '11'),
                ('S4', '12'),
                ('S4', '13'),
                ('S4', '14'),
                ('S4', '15'),
                ('S4', '16'),
                ('S4', '17'),
                ('S4', '18'),
                ('S4', '19'),
                ('RBI', '2'),
                ('RBI', '1'),
                ('RBO', '1'),
                ('RBO', '2'),
                ('VZ', '2'),
                ('RLD', '14'),
                ('RLD', '13'),
                ('RLD', '12'),
                ('RLD', '11'),
                ('RLD', '10'),
                ('RLD', '09'),
                ('RLD', '08'),
                ('RLD', '07'),
                ('RLD', '06'),
                ('RLD', '05'),
                ('RLD', '04'),
                ('RLD', '03'),
                ('RLD', '02'),
                ('RLD', '01'),
                ('RTO', '1'),
                ('S3', '11'),
                ('S3', '12'),
                ('S3', '13'),
                ('S3', '14'),
                ('S3', '15'),
                ('S3', '16'),
                ('S3', '17'),
                ('S3', '18'),
                ('S3', '19')
            )
        elif self.weld_type == "long_bpp_2":
            segment_keys = (
                ('LBI', '8'),
                ('LBI', '7'),
                ('LBO', '8'),
                ('LBO', '9'),
                ('LTI', '8'),
                ('LTI', '7'),
                ('LTI', '6'),
                ('LTO', '7'),
                ('LTO', '8'),
                ('LTO', '9'),
                ('S3', '01'),
                ('S3', '02'),
                ('S3', '03'),
                ('S3', '04'),
                ('S3', '05'),
                ('S3', '06'),
                ('S3', '07'),
                ('S3', '08'),
                ('S3', '09'),
                ('S3', '10'),
                ('LBI', '6'),
                ('LBO', '6'),
                ('LBO', '7'),
                ('S2', '19'),
                ('S2', '18'),
                ('S2', '17'),
                ('S2', '16'),
                ('S2', '15'),
                ('S2', '14'),
                ('S2', '13'),
                ('S2', '12'),
                ('S2', '11'),
                ('S2', '10'),
                ('S2', '09'),
                ('S2', '08'),
                ('S2', '07'),
                ('S2', '06'),
                ('S2', '05'),
                ('S2', '04'),
                ('S2', '03'),
                ('S2', '02'),
                ('S2', '01'),
                ('LTI', '5'),
                ('LTI', '4'),
                ('LTO', '4'),
                ('LTO', '5'),
                ('LTO', '6'),
                ('S1', '01'),
                ('S1', '02'),
                ('S1', '03'),
                ('S1', '04'),
                ('S1', '05'),
                ('S1', '06'),
                ('S1', '07'),
                ('S1', '08'),
                ('S1', '09'),
                ('S1', '10'),
                ('S1', '11'),
                ('S1', '12'),
                ('S1', '13'),
                ('S1', '14'),
                ('S1', '15'),
                ('S1', '16'),
                ('S1', '17'),
                ('S1', '18'),
                ('S1', '19'),
                ('LBI', '4'),
                ('LBI', '5'),
                ('LBO', '5'),
                ('LBO', '4'),
                ('LBI', '3'),
                ('LBI', '2'),
                ('LBO', '2'),
                ('LBO', '3'),
                ('LMH', '2'),
                ('LSH', '4'),
                ('LSH', '3'),
                ('LSH', '2'),
                ('LLH', '5'),
                ('LLH', '4'),
                ('LLH', '3'),
                ('LLH', '2'),
                ('LLH', '9'),
                ('LLH', '8'),
                ('LLH', '7'),
                ('LLH', '6'),
                ('LSH', '1'),
                ('LSH', '6'),
                ('LSH', '5'),
                ('LMH', '4'),
                ('LMH', '3'),
                ('LBI', '1'),
                ('LI', '2'),
                ('LI', '1'),
                ('LTI', '1'),
                ('LTI', '2'),
                ('LTI', '3'),
                ('LLH', '1'),
                ('LTO', '3'),
                ('LTO', '2'),
                ('LLD', '01'),
                ('LLD', '02'),
                ('LLD', '03'),
                ('LLD', '04'),
                ('LLD', '05'),
                ('LLD', '06'),
                ('LLD', '07'),
                ('LLD', '08'),
                ('LLD', '09'),
                ('LLD', '10'),
                ('LLD', '11'),
                ('LLD', '12'),
                ('LLD', '13'),
                ('LLD', '14'),
                ('LSD', '1'),
                ('LSD', '2'),
                ('LSD', '3'),
                ('LSD', '4'),
                ('LMH', '5'),
                ('LMH', '1'),
                ('LMD', '6'),
                ('LMD', '5'),
                ('LMD', '4'),
                ('LMD', '3'),
                ('LMD', '2'),
                ('LMD', '1'),
                ('LO', '2'),
                ('LO', '1'),
                ('LZ', '1'),
                ('LTO', '1'),
                ('LZ', '2'),
                ('LZ', '3'),
                ('LO', '3'),
                ('LBO', '1'),
                ('LZ', '4')
            )
        # lengths = np.array(stop_indices) - np.array(start_indices)
        # print(max(lengths), min(lengths), remove_at_boundary_samples)
        self.segments = {
            segment_keys[i]: self.dataframe.iloc[
                start_indices[i] + remove_at_boundary_samples[0] : stop_indices[i] - remove_at_boundary_samples[1]
            ].reset_index(
                # Make sure that the index of the segments gets reset, otherwise
                # there could be problems with other functions (setting labels etc)
                inplace=False,
                drop=True,
            )
            for i in range(self.n_segments)
        }
        print(
            f"Split {self.n_segments} segments"
            # f" of lengths: {[len(segment) for segment in self.segments.values()]}"
            f" for plate {self.identifier}."# of type {self.weld_type}."
        )
        # print(f"Needed {iteration-1} iterations.")

        return

    def add_location_and_angle(self) -> None:
        """
        TBC
        """
        if self.segments is None:
            self.split_segments()
        if self.weld_type == "step_weld":
            limits = [(16.5, 6.5), (-6.5, -16.5), (39.5, 29.5), (-29.5, -39.5)]
        elif self.weld_type == "cont_weld":
            limits = [(39.5, -39.5)]
        else:
            raise NameError(f"Method not implemented for weld_type {self.weld_type}.")
        for idx, segment in self.segments.items():
            self.segments[idx].loc[:, "X-position"] = (
                np.linspace(
                    start=limits[idx - 1][0],
                    stop=limits[idx - 1][1],
                    num=len(segment),
                )
                + self.x_offset
            )
            print(
                f"Added X-position for segment {idx}/{self.n_segments} "
                f"of plate {self.identifier} of type {self.weld_type}."
            )
            self.segments[idx].loc[:, "X-angle"] = (
                np.arctan(segment["X-position"] / Z_DISTANCE) / np.pi * 180
            )
            print(
                f"Added X-angle for segment {idx}/{self.n_segments} "
                f"of plate {self.identifier} of type {self.weld_type}."
            )

    def add_labels(self, df_labels: pd.DataFrame) -> None:
        """
        WORKS ONLY FOR A SPECIFIC LABELLING SCHEME!
        """
        # Make sure that the plate has already been split into segments
        if self.segments is None:
            self.split_segments()

        # Consider only defects of the current plate
        df = df_labels[df_labels["Plate"] == self.identifier]

        # Define length in um of a segment
        seg_length_um = {"step_weld": 10000, "cont_weld": 79000}[self.weld_type]

        # Iterate over segments
        for n, segment in self.segments.items():

            # Initialize "Defect" column with empty list
            defects = pd.Series([[] for _ in segment.index])

            # Consider only defects of the current segment
            # Separate defects that affect a point from defects that affect an interval
            point_df = df[df["Segment"] == str(n)].dropna(axis=0, subset=["Point"])
            long_df = df[df["Segment"] == str(n)].dropna(
                axis=0, subset=["Start", "End"], how="any"
            )

            # Iterate over rows and assign defects that affect a point
            for row in point_df.itertuples(index=False):
                # Defects at the end of the segment
                if row.Point == "Ende":
                    defects.iloc[-1] += [row.Defect]
                else:
                    defects.iloc[
                        int(len(segment) * float(row.Point) // seg_length_um)
                    ] += [row.Defect]

            # Iterate over rows and assign defects that affect an interval
            for row in long_df.itertuples(index=False):
                dfct_start = int(len(segment) * float(row.Start) // seg_length_um)
                dfct_end = int(len(segment) * float(row.End) // seg_length_um)
                defects.iloc[dfct_start:dfct_end] += pd.Series(
                    [[row.Defect] for _ in range(dfct_start, dfct_end)]
                ).values

            self.segments[n]["Defects"] = defects.map(set).values


def combine_plates(
    plates: list[Plate],
    n_snip_per_seg: int,
    channels: (
        list[str] | None
    ) = None,  # channels and/or columns in segments' dataframes
    norm_snip: bool = True,
    max_loss: int = 50,
    segments_subset: list | None = None,
    return_snip_len: bool = False,
    verbose=True,
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    """
    In any case, we always get n_snip_per_seg snippets of uniform length snip_len (determined by the algorithm).
    If the max sample loss is smaller than max_loss, the segments are truncated (at the end) and the snippets are non-overlapping.
    Otherwise, the last two snippets of each segment overlap.
    """

    if segments_subset is None:
        segments_dict = {plate: plate.segments for plate in plates}
    else:
        segments_dict = {
            plate: {segment_key: plate.segments[segment_key]}
            for plate in plates
            for segment_key in segments_subset
        }

    if channels is None:
        # Take channels that occur in all plates
        channels = list(
            set(plates[0].data_channels).intersection(
                *map(set, [plate.data_channels for plate in plates[1:]])
            )
        )
    if (channels is None) or (not channels):
        raise ValueError("Channels list is empty.")

    if "Defects" in channels:
        channels.remove("Defects")

    # Truncate or overlap to have constant length across segments
    max_seg_len = max(
        len(segment) for plate in plates for segment in segments_dict[plate].values()
    )
    min_seg_len = min(
        len(segment) for plate in plates for segment in segments_dict[plate].values()
    )

    if verbose:
        print(f"Maximal segment length: {max_seg_len} samples.")
        print(f"Minimal segment length: {min_seg_len} samples.")
        print(f"Extracting {n_snip_per_seg} snippets from each segment of each plate:")

    uni_seg_len = min_seg_len - min_seg_len % n_snip_per_seg
    snip_len = uni_seg_len // n_snip_per_seg

    # Labels
    y = None

    if max_seg_len - uni_seg_len <= max_loss:

        # Cut into n_snip_per_seg non-overlapping segments
        # Discard what is left at the end

        if verbose:
            print(
                f">> Truncating segments to have uniform length of {uni_seg_len} samples."
            )
            print(f">> Length of snippets: {snip_len} samples.")
            print(
                f">> Maximal sample loss per segment: {max_seg_len - uni_seg_len} samples."
            )

        X = np.concatenate(
            [
                # Have: uni_seg_len x len(channels)
                # Want: n_snip_per_seg x snip_len x len(channels)
                np.array(segment[channels].iloc[:uni_seg_len]).reshape(
                    (n_snip_per_seg, snip_len, len(channels))
                )
                for plate in plates
                for segment in segments_dict[plate].values()
            ],
            axis=0,
        )
        # Got: (n_plates * n_segments * n_snip_per_seg) x snip_len x len(channels)

        # Determine labels
        if all(
            "Defects" in segment.columns
            for plate in plates
            for segment in segments_dict[plate].values()
        ):

            # Concatenate labels of snippets
            y = np.concatenate(
                [
                    np.concatenate(
                        [
                            np.apply_along_axis(
                                # Make union of defects sets, "located" at each point of the snippet
                                # Consider also the discarded timesteps for the label of the last snippet of each segment
                                lambda row: set().union(*row),
                                axis=1,
                                arr=np.array(
                                    segment["Defects"].iloc[: uni_seg_len - snip_len]
                                ).reshape(n_snip_per_seg - 1, snip_len),
                            ),
                            [
                                set().union(
                                    *segment["Defects"].iloc[uni_seg_len - snip_len :]
                                )
                            ],
                        ],
                        axis=0,
                    )
                    for plate in plates
                    for segment in segments_dict[plate].values()
                ],
                axis=0,
            )
            print("Determined labels of snippets.")

    else:

        # Cut into (n_snip_per_seg-1) non-overlapping segments
        # Add one further (overlapping) segment at the end

        uni_seg_len = min_seg_len - min_seg_len % (n_snip_per_seg - 1)
        snip_len = uni_seg_len // (n_snip_per_seg - 1)
        max_overlap = snip_len - (min_seg_len - uni_seg_len)

        if verbose:
            print(f">> Length of snippets: {snip_len} samples.")
            print(
                f">> The last two snippets of each segment overlap by at most: {max_overlap} samples."
            )

        X = np.concatenate(
            [
                np.concatenate(
                    [
                        # Have: uni_seg_len x len(channels)
                        # Want: (n_snip_per_seg-1) x snip_len x len(channels)
                        np.array(segment[channels].iloc[:uni_seg_len]).reshape(
                            (n_snip_per_seg - 1, snip_len, len(channels))
                        ),
                        np.array(segment[channels].iloc[-snip_len:]).reshape(
                            (1, snip_len, len(channels))
                        ),
                    ],
                    axis=0,
                )
                for plate in plates
                for segment in segments_dict[plate].values()
            ],
            axis=0,
        )
        # Got: (n_plates * n_segments * n_snip_per_seg) x snip_len x len(channels)

        # Determine labels
        if all(
            "Defects" in segment.columns
            for plate in plates
            for segment in segments_dict[plate].values()
        ):

            # Concatenate labels of snippets
            y = np.concatenate(
                [
                    np.apply_along_axis(
                        # Make union of defects sets, "located" at each point of the snippet
                        lambda row: set().union(*row),
                        axis=1,
                        arr=np.concatenate(
                            [
                                np.array(segment["Defects"].iloc[:uni_seg_len]).reshape(
                                    (n_snip_per_seg - 1, snip_len)
                                ),
                                np.array(segment["Defects"].iloc[-snip_len:]).reshape(
                                    (1, snip_len)
                                ),
                            ],
                            axis=0,
                        ),
                    )
                    for plate in plates
                    for segment in segments_dict[plate].values()
                ],
                axis=0,
            )
            if verbose:
                print("Determined labels of snippets.")

    # z-score normalization by snippet and by channel
    if norm_snip:
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    # Need to invert features and timesteps for classifiers, so then we have:
    # snippets x features x timesteps
    X = np.swapaxes(X, 1, 2)

    # X = X.squeeze()

    if return_snip_len:
        # Return also the length of the snippets
        return X, y, snip_len

    return X, y


def enumerate_defects(labels_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str]]:
    """TBC"""

    # Factorize the defects column
    codes, unique_values = pd.factorize(labels_df["Defect"])

    # Increment codes by 1 (to start from 1 instead of 0)
    labels_df["Defect"] = codes + 1

    # Create the dictionary {code+1: string}
    mapping_dict = {code + 1: value for code, value in enumerate(unique_values)}

    return (labels_df, mapping_dict)


def get_filenames_offsets_gaps(
    folderpath: str,
    x_offsets: list[float] | None = None,
    y_offsets: list[float] | None = None,
    gaps: list[float] | None = None,
    subset_list: list[int] | None = None,
    subset_type: str | None = None,
) -> tuple[list]:
    """
    TBC
    """

    # Determine filenames
    filenames = [
        name.split(".")[0] if name.split(".")[-1] in ("csv", "bin") else name
        for name in os.listdir(folderpath)
        if (
            os.path.isfile(os.path.join(folderpath, name))
            and name.split(".")[-1] in ("csv", "bin")
        )
        or (
            os.path.isdir(os.path.join(folderpath, name))
            and name.split("_")[-1][0] == "#"
        )
    ]
    filenames.sort()

    # Subset mode: load only a subset of the plates
    if subset_list is None:
        subset_idx = list(range(len(filenames)))
    elif subset_type is None or subset_type == "ID":
        # Find indices of the corresponding identifiers
        file_identifiers = [name.split("_")[-1].strip("#") for name in filenames]
        assert len(subset_list) == len(
            set(subset_list)
        ), "There are multiple instances of the same plate identifier in the subset list."
        assert len(file_identifiers) == len(
            set(file_identifiers)
        ), "There are multiple instances of the same plate identifier in the folder."
        assert all(
            item in file_identifiers for item in list(map(str, subset_list))
        ), "Some subset plate identifiers are not valid."
        subset_idx = [
            file_identifiers.index(identifier)
            for identifier in list(map(str, subset_list))
        ]
    elif subset_type == "index":
        subset_idx = subset_list.copy()
    else:
        raise ValueError("Subset type inexistent, only 'ID' or 'index' are allowed")
    filenames, x_offsets, y_offsets, gaps = [
        (
            [var[idx] for idx in subset_idx]
            if var is not None
            else len(subset_idx) * [0.0]
        )
        for var in [filenames, x_offsets, y_offsets, gaps]
    ]

    return filenames, x_offsets, y_offsets, gaps


def get_segments_bounds(
    series: pd.Series,
    threshold: float,
    num_seg: int,
    min_seg_len: int = 100,
    max_gap_len: int = 4,
) -> None | dict[pd.Series]:
    """TBC"""
    # Threshold the series
    mask_above_thr = series > threshold
    # Close gaps shorter than max_gap_len
    smoothed_mask = binary_dilation(mask_above_thr, iterations=max_gap_len // 2)
    # Label contiguous regions
    labels_array, num_seg_found = label(smoothed_mask)
    # Extract only segments longer than min_seg_len
    segments_bounds = []
    for segment_id in range(1, num_seg_found + 1):
        segment_range = np.where(labels_array == segment_id)[0]
        if len(segment_range) >= min_seg_len:
            segments_bounds.append((segment_range[0], segment_range[-1]))
    # Return the segment indices if the amount of segments is correct
    return segments_bounds if len(segments_bounds) == num_seg else None
