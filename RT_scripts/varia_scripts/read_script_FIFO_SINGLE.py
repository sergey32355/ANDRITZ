import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

import spcm
from spcm import units
from sklearn.ensemble import RandomForestClassifier

card: spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(
    card_type=spcm.SPCM_TYPE_AI
) as card:  # if you want to open the first card of a specific type

    # single FIFO
    card.card_mode(spcm.SPC_REC_FIFO_SINGLE)

    # card.timeout(10 * units.s)                                # if you want to set a timeout

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(1 * units.MHz)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels[0].amp(1 * units.V)

    # define the data buffer
    num_samples = 100 * units.MiS
    notify_samples = 100 * units.KiS

    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(100 * units.MiS)
    data_transfer.allocate_buffer(num_samples)
    # data_transfer.pre_trigger(10 * units.KiS)
    # data_transfer.post_trigger(10 * units.KiS)
    data_transfer.notify_samples(notify_samples)
    data_transfer.start_buffer_transfer()
    # data_transfer.verbose(True)

    # start the card
    card.start(spcm.M2CMD_DATA_STARTDMA | spcm.M2CMD_CARD_ENABLETRIGGER)

    # load
    with open("model.pkl", "rb") as f:
        random_forest = pickle.load(f)

    def classify(data):
        if len(data) == 0:
            return
        signal = np.concatenate(data)
        features = np.array(
            [
                [
                    np.mean(signal),
                    np.std(signal),
                    skew(signal),
                    kurtosis(signal),
                    np.min(signal),
                    np.max(signal),
                    np.median(signal),
                    np.max(signal) - np.min(signal),
                    np.percentile(signal, 25),
                    np.percentile(signal, 50),
                    np.percentile(signal, 75),
                ]
            ]
        )

        prediction = random_forest.predict(features)
        print(prediction)

    try:
        print("started")
        i = 0

        data_to_classify = []
        started = False

        for data_block in data_transfer:

            channel = channels[0]
            unit_data_V = channel.convert_data(data_block[channel], units.V)
            # data["mic"].append(np.array(unit_data_V))
            data_max = np.max(unit_data_V).magnitude
            # print(f'{data_max:.4f}')

            if data_max > 0.1:
                if not started:
                    print("ok")
                started = True

            if started:
                new_data = np.array(unit_data_V.magnitude)
                data_to_classify.append(new_data)

                if i == 30:  # run for 3s
                    classify(data_to_classify)
                    data_to_classify = []
                    started = False
                    i = 0
                    print("restarting")

                i += 1

    except KeyboardInterrupt:
        pass
    #     with open(filename, "wb") as file:
    #         # Write the data to the file after the loop
    #         pickle.dump(data, file)
    #         # Ensure that the data is written to disk and does not just stay in the buffer
    #         file.flush()
    #         os.fsync(file.fileno())
