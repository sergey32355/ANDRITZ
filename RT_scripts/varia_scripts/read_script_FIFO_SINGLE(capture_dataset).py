import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import spcm
from spcm import units

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
    clock.sample_rate(100 * units.kHz)

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

    # Optionally pass the filename as argument through the command line
    filename = f"{sys.argv[1]}" if len(sys.argv) > 1 else "test.pkl"

    data = []
    started = False
    block_counter = 0

    try:
        print("ready")

        data_to_classify = []

        for data_block in data_transfer:

            channel = channels[0]
            unit_data_V = channel.convert_data(data_block[channel], units.V)
            data_max = np.max(unit_data_V).magnitude

            if data_max > 0.1:
                if not started:
                    print("started")
                started = True

            if started:
                new_data = np.array(unit_data_V.magnitude)
                data.append(new_data)

                if block_counter == 30:  # run for 3s
                    print("done")
                    with open(filename, "wb") as file:
                        pickle.dump(np.concatenate(data), file)
                        file.flush()
                        os.fsync(file.fileno())
                        exit()

                block_counter += 1

    except KeyboardInterrupt:
        with open(filename, "wb") as file:
            # Write the data to the file after the loop
            pickle.dump(data, file)
            # Ensure that the data is written to disk and does not just stay in the buffer
            file.flush()
            os.fsync(file.fileno())
