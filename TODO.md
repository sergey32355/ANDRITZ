* Cannot add new_bpp to dropdown??
* "Browse" does not work for RT folder
* Saving does not work

## RT:
* Goal: wait for the trigger, then record it for a fixed amount of time, and repeat for as long as needed. Input: how long to record
* If the waiting time gets exceeded, or if RT is canceled, the card should be re-set-up properly, not completely disconnected. -> different: never time out if active, and exit if stopped (but save or not?)
* But we also need to generate the .bin and .txt files !?! Or something similar
* And also need an option for recording without trigger (although: can use other channels?), e.g. M2CMD_CARD_FORCETRIGGER or just start immediately and end after given time or when stopped
* data_transfer.time_data() could be useful, since it is aligned with the first trigger (and has negative time for pre-trigger)

* Preparare pipeline Sergey on github
* Fix data processing for GUI (different file format) -> already fine?
* Need to do config file with fixed stuff (channels, amplitudes etc), besides the input from the GUI
* what is return unit?

* Should separate thread for acquisition from thread for processing

* Make an executable with a single config file for the hardware stuff (not from GUI) (and the GUI displays a link to it)
* 