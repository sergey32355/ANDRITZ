import atexit

import numpy as np
import spcm
import spcm_core

# Cards opened by open_spectrum_cards(); closed by close_spectrum_cards().
_open_cards = []
# spcm.Channels of each open card, used by read_spectrum_data() for volt conversion.
_card_channels = {}
# spcm.DataTransfer of each open card; its DMA buffer is allocated once and reused.
_card_transfers = {}


def list_spectrum_cards(max_cards=16):
    """Find the Spectrum cards installed in this PC.

    Tries to open /dev/spcm0 ... /dev/spcm{max_cards-1}. Each card that opens is
    queried for its name and serial number and closed straight away; no settings
    are changed. A card already opened by another program (e.g. SBench) cannot be
    opened and is therefore not listed.

    Returns a list of dicts with keys 'index', 'device', 'product_name' and
    'serial_number'. The number of cards found is len() of that list.
    """
    cards = []
    for index in range(max_cards):
        device = f"/dev/spcm{index}"
        # Probe with the low-level driver first: a failed spcm.Card open leaves an
        # object whose destructor raises when it tries to stop the invalid handle.
        handle = spcm_core.spcm_hOpen(spcm_core.create_string_buffer(device.encode()))
        if not handle:
            continue
        spcm_core.spcm_vClose(handle)
        with spcm.Card(device) as card:
            cards.append(
                {
                    "index": index,
                    "device": device,
                    "product_name": card.product_name(),
                    "serial_number": card.sn(),
                }
            )
    return cards


def open_spectrum_cards(
    card_info,
    sampling_rate=10_000_000,
    trigger_channel=-1,
    pre_trigger_samples=1024,
    post_trigger_samples=7168,
    input_range_mV=10_000,
    trigger_level_mV=0,
):
    """Open one card from list_spectrum_cards() and configure all its channels.

    `card_info` is one entry of that list, e.g. open_spectrum_cards(found[0]).
    The card is configured but not started. It stays open until
    close_spectrum_cards() is called or Python exits, whichever comes first.

    sampling_rate        : samples per second, same for all channels.
    trigger_channel      : input channel (0, 1, ...) to trigger on, rising edge.
                           -1 = software trigger: recording starts right away.
    pre_trigger_samples  : samples per channel recorded before the trigger.
    post_trigger_samples : samples per channel recorded after the trigger.
    input_range_mV       : input range of all channels, +/- this value in mV.
    trigger_level_mV     : trigger level on trigger_channel, in mV.

    Returns the opened spcm.Card object.
    """
    card = spcm.Card(card_info["device"])
    card.open()
    _open_cards.append(card)

    channels = spcm.Channels(card, card_enable=(1 << card.num_channels()) - 1)
    channels.amp(input_range_mV)

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(int(sampling_rate))

    card.card_mode(spcm.SPC_REC_STD_SINGLE)
    trigger = spcm.Trigger(card)
    if trigger_channel == -1:
        trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
        trigger.ch_or_mask0(0)
    else:
        ch = channels[trigger_channel]
        trigger.or_mask(spcm.SPC_TMASK_NONE)
        trigger.ch_or_mask0(ch.ch_mask())
        trigger.ch_mode(ch, spcm.SPC_TM_POS)
        trigger.ch_level0(ch, trigger_level_mV * spcm.units.mV)

    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(pre_trigger_samples + post_trigger_samples)
    data_transfer.post_trigger(post_trigger_samples)
    data_transfer.allocate_buffer(pre_trigger_samples + post_trigger_samples)

    _card_channels[card] = channels
    _card_transfers[card] = data_transfer
    return card


def read_spectrum_data(card, timeout_s=10):
    """Record one acquisition on a card set up by open_spectrum_cards().

    Starts the card, waits until pre + post trigger samples are recorded on all
    channels and copies them to the PC. With trigger_channel = -1 the recording
    starts right away; otherwise it waits for the trigger. If no trigger arrives
    within timeout_s seconds, the card is stopped and spcm.SpcmTimeout is raised.

    Returns a float array of shape (N_channels, M) in volts, row i = channel i.
    """
    # Make sure the previous recording and DMA are finished before restarting.
    card.stop(spcm.M2CMD_DATA_STOPDMA)
    card.timeout(int(timeout_s * 1000))
    data_transfer = _card_transfers[card]
    # Record into card memory first; a timeout here means no trigger arrived.
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
    except spcm.SpcmTimeout:
        card.stop()
        raise
    # Then copy the finished recording from card memory to the PC.
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)

    channels = _card_channels[card]
    raw = data_transfer.buffer
    return np.array(
        [channels[i].convert_data(raw[i], return_unit=spcm.units.V) for i in range(len(channels))]
    )


def close_spectrum_cards(card=None):
    """Close a card opened by open_spectrum_cards(). Safe to call twice.

    card : the spcm.Card returned by open_spectrum_cards(). If None, every card
           opened by open_spectrum_cards() is closed (this is what runs at exit).
    """
    if card is None:
        while _open_cards:
            c = _open_cards.pop()
            _card_channels.pop(c, None)
            _card_transfers.pop(c, None)
            c.close()
    elif card in _open_cards:
        _open_cards.remove(card)
        _card_channels.pop(card, None)
        _card_transfers.pop(card, None)
        card.close()


atexit.register(close_spectrum_cards)
