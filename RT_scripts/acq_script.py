import spcm
from spcm import units
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

card : spcm.Card

# if you want to open the first card of a specific type
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:
    
    card.card_mode(spcm.SPC_REC_STD_SINGLE)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1 | spcm.CHANNEL2 | spcm.CHANNEL3)
    channels.amp(1000 * units.mV)
    channels.offset(0 * units.mV)
    channels.termination(1)

    # setup channel trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)
    trigger.and_mask(spcm.SPC_TMASK_NONE)
    trigger.ch_or_mask0(channels[0].ch_mask())
    trigger.ch_mode(channels[0], spcm.SPC_TM_HIGH)
    trigger.ch_level0(channels[0], 500 * units.mV, return_unit=units.mV)

    # setup clock
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(20 * units.kHz)

    for repeat in range(3):

        # define the data buffer
        data_transfer = spcm.DataTransfer(card)
        data_transfer.duration(5*units.s, post_trigger_duration=4.5 * units.s)
        
        # start card and wait until recording is finished
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)

        # start DMA transfer and wait until the data is transferred
        data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)

        # acquire and save data and metadata
        data = {}
        metadata = {
            'sample_rate': clock.sample_rate(),
            # 'duration': data_transfer.duration(return_unit=units.s),
            # 'post_trigger_duration': data_transfer.post_trigger_duration(return_unit=units.s),
            # 'trigger_level': trigger.ch_level0(channels[0], return_unit=units.mV),
            # 'trigger_mode': trigger.ch_mode(channels[0]),
            # 'channel_termination': channels.termination(0),
            # 'channel_offset': channels.offset(0, return_unit=units.mV),
            # 'channel_amplitude': channels.amp(0, return_unit=units.mV)
        }

        for i in range(4):
            channel_data = np.array(channels[i].convert_data(data_transfer.buffer[channels[i], :], units.V))
            data[f'channel {i}'] = channel_data
        
        with open(f"data/data_{repeat}.pkl", "wb") as file:
            pickle.dump(data, file)
            file.flush()
            os.fsync(file.fileno())

        with open(f"data/metadata_{repeat}.pkl", "wb") as file:
            pickle.dump(metadata, file)
            file.flush()
            os.fsync(file.fileno())
