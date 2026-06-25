import spcm
from spcm import units
import jax
import numpy as np
import matplotlib.pyplot as plt
import time 
import jax.numpy as jnp
import random
#from multiprocess import Process, Queue, Value, Array, Manager
from multiprocessing import Process, Value
import traceback
from jaxdl.rl.networks.actor_nets import sample_actions
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

def run_output_card(power_val):
    with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as output_card:

        output_card.card_mode(spcm.SPC_REP_FIFO_SINGLE)
        output_card.set_i(spcm.SPC_DATA_OUTBUFSIZE, 1024)

        # setup all channels
        out_channels = spcm.Channels(output_card, card_enable=spcm.CHANNEL0)
        out_channels.enable(True)
        out_channels.output_load(1 * units.Mohm)
        out_channels.amp(5.5 * units.V)

        clock = spcm.Clock(output_card)
        sample_rate = clock.sample_rate(40 * units.MHz, return_unit=units.MHz)
        clock.clock_output(0)

        # setup the trigger mode
        out_trigger = spcm.Trigger(output_card)
        out_trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

        # define the data buffer
        notify_samples = 32 * units.KiS
        num_samples = notify_samples * 4

        out_data_transfer = spcm.DataTransfer(output_card)
        out_data_transfer.memory_size(num_samples)
        out_data_transfer.allocate_buffer(num_samples)
        out_data_transfer.loops(0)  # loop continuously
        out_data_transfer.notify_samples(notify_samples)
        
        out_data_transfer.buffer[:] = power_val.value
        out_data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, direction=spcm.SPCM_DIR_PCTOCARD)
        out_data_transfer.avail_card_len(num_samples)

        print("... data has been transferred to board memory")
        
        output_card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
        
        for out_data_block in out_data_transfer:
            out_data_block[:] = power_val.value

@jax.jit
def agent_policy_fn(rng, actor_net, obs):
    rng, action = sample_actions(rng, actor_net, obs, temperature=0.1)
    return rng, action

def random_policy_fn(rng, actor_net, obs):
    rng, key = jax.random.split(rng)
    action = jax.random.uniform(key, (1, 1), minval=-1, maxval=1)
    return rng, action


def update_running_means(reflection_data, emission_data, in_data_block, count):
    reflection_data = reflection_data + (np.mean(in_data_block[0] / 32767.) - reflection_data) / count
    emission_data = emission_data + (np.mean(in_data_block[1] / 32767.) - emission_data) / count
    return reflection_data, emission_data

class SpectrumRunner():

    def __init__(self, input_card):
        self.power = Value('d', 32000)

        output_card_runner = Process(target=run_output_card, args=(self.power, ))
        output_card_runner.start()

        self.input_card = input_card
        self.input_card.card_mode(spcm.SPC_REC_FIFO_GATE)
        self.input_card.set_i(spcm.SPC_DATA_OUTBUFSIZE, 1024)

        in_channels = spcm.Channels(self.input_card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL2)
        in_channels.termination(1)
        in_channels.amp(5 * units.V)
        self.in_channels = in_channels

        in_trigger = spcm.Trigger(self.input_card)
        in_trigger.ext0_mode(spcm.SPC_TM_POS)
        in_trigger.or_mask(spcm.SPC_TMASK_EXT0)
        in_trigger.ext0_level0(500 * units.mV)

        in_notify_samples = 16 * units.KiS
        #in_notify_samples = 64 * units.KiS
        in_num_samples = 4 * in_notify_samples

        clock = spcm.Clock(self.input_card)
        clock.mode(spcm.SPC_CM_INTPLL)
        sample_rate = clock.sample_rate(40 * units.MHz, return_unit=units.MHz)

        clock.clock_output(False)

        in_data_transfer = spcm.DataTransfer(self.input_card)
        in_data_transfer.memory_size(in_num_samples)
        in_data_transfer.allocate_buffer(in_num_samples)
        in_data_transfer.notify_samples(in_notify_samples)
    
        in_data_transfer.start_buffer_transfer()

        self.in_data_transfer = in_data_transfer
        self.input_card.start(spcm.M2CMD_DATA_STARTDMA | spcm.M2CMD_CARD_ENABLETRIGGER)
        self.card_started = False
        print('card started at', time.time())

        print("Setup ready")

    def run_episode(self, rng, actor_net, random_episode=False):
        obses = [np.array([0.0, 0.0]).reshape(1, 2)]
        termination_step = 4638
        
        # warmup
        
        random_count = 43
        random_reflection_data = 0.
        random_emission_data = 0.
        random_in_data_block = np.ones((2, 16384))
        for j in range(10):
            rng, power = agent_policy_fn(rng, actor_net, obses[0])
            power.block_until_ready()

            reflection_data, emission_data = update_running_means(random_reflection_data, random_emission_data, random_in_data_block, random_count)
            #reflection_data.block_until_ready()

        rng, power = agent_policy_fn(rng, actor_net, obses[0])
        self.power.value = float((power.item() + 1) * 16383.5)

        powers = [power]
        step = 0


        print('waiting for trigger')
        prev_time = 0
        
        reflection_data = 0.0
        emission_data = 0.0
        current_times = []
        count = 0

        for in_data_block in self.in_data_transfer:
            if prev_time == 0:
                prev_time = time.time()

            current_time = time.time()
            current_times.append(current_time)

            if step >= termination_step:
                break

            count += 1
            reflection_data, emission_data = update_running_means(reflection_data, emission_data, in_data_block, count)

            # Change action every 24 steps, corresponds to roughly 9.8ms at 16 KiS notify size
            if (step % 24 == 0) and (step > 1):
                inputs = np.array([reflection_data, emission_data]).reshape(1, 2)

                if random_episode:
                    rng, power = random_policy_fn(rng, actor_net, inputs)
                else:
                    rng, power = agent_policy_fn(rng, actor_net, inputs)

                self.power.value = int((power.item() + 1) * 16383)

                reflection_data = 0.
                emission_data = 0.
                count = 1

                obses.append(inputs)
                powers.append(power)
            
            step += 1
        
        return rng, jnp.stack(powers).reshape(-1, 1), jnp.stack(obses)
    
