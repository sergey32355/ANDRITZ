"""TBC"""

import numpy as np
import pandas as pd


def get_channel_sorting_dict(channel_order_hex, n_channels):
    """Determine channel sorting dictionary, in case order of channels is alternating"""

    # Default
    channel_sorting_dict = {i: i for i in range(n_channels)}

    # Rearrange channel numbers if the channels have been saved alternatingly in the .bin file
    if bool(int(channel_order_hex, 16) & int("0x00020000", 16)):

        # Deal with uneven number of channels
        even_n_channels = n_channels if n_channels % 2 == 0 else n_channels + 1

        # Determine how to shift each channel
        list_of_lower_shifted_channels = list(range(even_n_channels // 2))
        list_of_upper_shifted_channels = [
            entry + (even_n_channels // 2) for entry in list_of_lower_shifted_channels
        ]

        # Concatenate channels in alternating order
        channel_placements_list = list(
            sum(
                zip(list_of_lower_shifted_channels, list_of_upper_shifted_channels),
                (),
            )
        )

        # Trim fictitious channels if n_channels is odd
        if n_channels % 2 == 1:
            channel_placements_list = channel_placements_list[:-1]

        # Determine channel sorting dictionary
        channel_sorting_dict = {
            i: channel_placements_list.index(i) for i in range(n_channels)
        }

    return channel_sorting_dict


def read_bin_data(
    settings_file: str,
    data_file: str,
    subset_data_channels: list[str] | None = None,
    trigger_channel: str = "Trigger",
) -> tuple[pd.DataFrame, list[str], float]:
    """TBC"""

    with open(settings_file, "r", encoding="utf-8") as settings:

        # Inverted list as stack
        settings_lines = settings.readlines()[::-1]

    sampling_rate = None
    adc_resolution = None
    lenh = None
    lenl = None
    max_adc_value = None
    channel_order_hex = None
    orig_max_ranges = []
    channels = []

    # Read settings
    while settings_lines:

        current_line = settings_lines.pop()

        if "[Ch" in current_line:
            channels.append(settings_lines.pop().split("=")[-1].strip())

        elif "OrigMaxRange" in current_line:
            orig_max_ranges.append(float(current_line.split("=")[-1].strip()))

        elif "Samplerate" in current_line:
            sampling_rate = float(current_line.split("=")[-1].strip())

        elif "Resolution" in current_line:
            adc_resolution = float(current_line.split("=")[-1].strip())

        elif "LenH" in current_line and lenh is None:
            lenh = int(current_line.split("=")[-1].strip())

        elif "LenL" in current_line and lenl is None:
            lenl = int(current_line.split("=")[-1].strip())

        elif "MaxADCValue" in current_line:
            max_adc_value = float(current_line.split("=")[-1].strip())

        elif "RawDataFormat" in current_line:
            channel_order_hex = hex(int(current_line.split("=")[-1].strip()))

    # Determine data length
    data_length: int = -1
    data_length = (lenh << 32) | (data_length & 0xFFFFFFFF)
    data_length = (data_length & 0xFFFFFFFF00000000) | (lenl)

    data_storage: np.ndarray = np.zeros(data_length * len(channels))

    with open(data_file, "rb") as binary_data_file:
        data_storage = np.fromfile(
            binary_data_file,
            dtype=(np.int16 if adc_resolution > 8 else np.int8),
            count=(data_length * len(channels)),
        )

    data_storage = data_storage.astype(np.float32).reshape(
        (
            data_length,
            len(channels),
        )
    )

    # Rearrange channel numbers if the channels have been saved alternatingly in the .bin file
    channel_sorting_dict = get_channel_sorting_dict(channel_order_hex, len(channels))
    data_storage = data_storage[:, list(channel_sorting_dict.values())]

    # Conversion from mV to V; and rescaling by channel
    data_storage = data_storage / 1000 / max_adc_value * np.array(orig_max_ranges)

    df = pd.DataFrame(data_storage, columns=channels)

    # Remove useless channels
    if subset_data_channels is not None:
        channels = subset_data_channels.copy()
        if trigger_channel is not None:
            channels += [trigger_channel]
        df = df[channels]
    if trigger_channel is not None:
        data_channels = [ch for ch in channels if ch != trigger_channel]
    else:
        data_channels = channels.copy()

    df.insert(
        loc=0,
        column="Time",
        value=np.linspace(0, data_length / sampling_rate, data_length),
    )

    return df, data_channels, sampling_rate

if __name__ == "__main__":
    df, data_channels, sampling_rate = read_bin_data(
        "xxx_binheader.txt",
        "xxx.bin"
    )
    print(df.head(), data_channels, sampling_rate)