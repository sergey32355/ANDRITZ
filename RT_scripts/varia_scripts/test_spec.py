import time
import spcm
from spcm import units
import numpy as np

proc_settings={
    "amplitude": 100,
    "sampling_rate": 1,
    "trigger_level": 50,
    "pre_trigger_duration": 10,
    "post_trigger_duration": 1000,
    "trig_chan_num": 0,
    "delay_between_measurements": 1000,
    "n_measurements": 2,
}

def Spectrum_card_tracking(self):

    """This function tracks the spectrum card and acquires data in real time. It is designed to work with the Spectrum M2P series of data acquisition cards.
    The function sets up the card, configures the channels, triggers, and clock, and then continuously waits for triggers to acquire data.
    When data is acquired, it processes the signals and can optionally display them in real time."""

    #check for settings
    AMPLITUDE=float(proc_settings.get("ampl_per_channel"))
    SAMPLING_RATE=float(proc_settings.get("sampling_rate"))
    TRIGGER_LEVEL=float(proc_settings.get("trigger_level"))
    PRETRIG_DURATION=float(proc_settings.get("pre_trigger_duration"))
    POSTTRIG_DURATION=float(proc_settings.get("post_trigger_duration"))
    TRIG_CHAN_NUM=int(proc_settings.get("trig_chan_num"))
    N_MEASUREMENTS=int(proc_settings.get("n_measurements"))
    # DELAY_MEASUREMENTS=int(proc_settings.get("delay_between_measurements"))
    CHAN_NAMES=["chan_0","chan_1","chan_2","chan_3","chan_4","chan_5","chan_6","chan_7"]     

    signals=[]
    for i in range(0,8):
        signals.append(np.array([]))

    card:spcm.Card
    with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:
        
        #https://github.com/SpectrumInstrumentation/spcm/blob/master/src/examples/01_acquisition/01_acq_single.py
        #https://github.com/SpectrumInstrumentation/spcm/blob/master/src/examples/01_acquisition/02_acq_single_2ch.py

        # do a simple standard setup
        card.card_mode(spcm.SPC_REC_STD_SINGLE)       # single trigger standard mode
        # card.timeout(5 * units.s)                     # timeout 5 s
                
        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_NONE)       # trigger set to none #software
        trigger.and_mask(spcm.SPC_TMASK_NONE)      # no AND mask

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)            # clock mode internal PLL
        clock.sample_rate(SAMPLING_RATE * units.MHz, return_unit=units.MHz)

        channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1 | spcm.CHANNEL2 | spcm.CHANNEL3 | spcm.CHANNEL4 | spcm.CHANNEL5 | spcm.CHANNEL6 | spcm.CHANNEL7) # enable channel 0 and 1
        channels.amp(AMPLITUDE * units.mV) # for all channels            
        channels.offset(0 * units.mV) # set for both channels
        channels.termination(1) # set for both channels

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)            # clock mode internal PLL
        clock.sample_rate(SAMPLING_RATE * units.MHz, return_unit=units.MHz)

        # Channel triggering
        trigger.ch_or_mask0(channels[TRIG_CHAN_NUM].ch_mask())
        trigger.ch_mode(channels[TRIG_CHAN_NUM], spcm.SPC_TM_POS)
        trigger.ch_level0(channels[TRIG_CHAN_NUM], float(TRIGGER_LEVEL) * units.mV, return_unit=units.mV) # trigger level - float(TRIGGER_LEVEL)
                    
        data_transfer = spcm.DataTransfer(card)
        data_transfer.duration((PRETRIG_DURATION+POSTTRIG_DURATION)*units.ms, post_trigger_duration=POSTTRIG_DURATION*units.ms)

        RT_Frame_Counter=1

        card.start(spcm.M2CMD_DATA_STARTDMA | spcm.M2CMD_CARD_ENABLETRIGGER)
        #(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
           
        # except:
        #     if(EXIT_RT_FLAG==True):
        #         print("Data acquisition is terminating...")
        #         return
        #     print("waiting...")
        #     continue 

        print("Data aquired...")
        data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)     
                                                                        
        sign_empty=False

        for i in range(0,8):
            signals[i]=channels[i].convert_data(data_transfer.buffer[channels[i], :], units.V)
            signals[i]=np.asarray(signals[i])
            if(len(signals[i])==0): sign_empty=True
                
        if(sign_empty):continue
        
        print("")
        print("********************************************************************")
        print("***********************MEASUREMENT "+str(RT_Frame_Counter)+"********************")        
        RT_Frame_Counter=RT_Frame_Counter+1

        time.sleep(DELAY_MEASUREMENTS / 1000.0 ) #ms to s                

    card.stop() # Stops the current run of the card. If the card is not running this command has no effect.
    card.reset() # A software and hardware reset is done for the board. All settings are set to the default values. The data in the board’s on-board memory will be no longer valid. Any output signals like trigger or clock output will be disabled.
    card.close() # Closes the connection to the card using a handle