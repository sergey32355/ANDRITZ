import numpy as np
import pandas as pd
import os
from os import walk
import sys
from time import sleep
import traceback
import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

SEGMENTS_MISSED_SEGMENT_NAME="noname_"

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

def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

class SPlate:
    def __init__(self,plate_name=""):
        self.name=plate_name
        self.raw_signals=[]
        self.time=[]
        self.chans_names=[]
        self.segments_names=[]                                     #names of the segments
        self.sigments_sign=[]                                     #signals for each signal
        self.sigments_start_t=[]     #we store the labelling results in segments for supervised classwification, format - [start el,end el,label]
        self.sigments_labels=[]
        self.delta_t=0
        self.snipp_l=[]                                                     #snippets length (i.e. number of sampling points)

    def get_data_from_df(self,df,time_col_name="Time"):
        jk = list(df.columns.values)        
        jk.remove(time_col_name)        
        self.time=df[time_col_name].to_numpy()    
        self.delta_t = self.time[1]-self.time[0]
        self.chans_names=jk
        self.raw_signals=[]        
        for mm in jk:
            self.raw_signals.append(df[mm].to_numpy())            
    
    def plate_info(self):
        print("")
        print("***********************************")
        print("Plate name: " + self.name)
        print("Channels num.: " + str(len(self.chans_names)))
        print("Channels names.: " + str(self.chans_names))        
        print("Raw sign. num.: " + str(len(self.raw_signals)))
        print("Time: start -  " + str(min(self.time))+" , end - "+str(max(self.time))+" , step -"+str(float(self.time[1])-float(self.time[0])))
        print("Segments num: "+ str(len(self.sigments_sign)))
        print("Segments names: "+ str(self.segments_names))              
                
    def get_segments(self,ref_chan_name="",threshold="automatic"):
        
        indx_trig = self.chans_names.index(ref_chan_name) 
        ref_sign=self.raw_signals[indx_trig]
        self.sigments_sign=[]        
        ref_threshold=-1
        if isinstance(threshold,str):
            unique = set()            
            for x in ref_sign:
                unique.add(x)
            un=list(unique)
            ref_threshold=min(un)+(max(un)-min(un))/2            
        else:
            ref_threshold=threshold

        raising_edge=[]
        falling_edge=[]
        for x in range(0,len(ref_sign)):
            if(x!=0):
                if(ref_sign[x-1]<=ref_threshold) and (ref_sign[x]>ref_threshold):
                    raising_edge.append(x)
                if(ref_sign[x-1]>ref_threshold) and (ref_sign[x]<=ref_threshold):
                    falling_edge.append(x)

        segm_extr=[]                                
        for l in range(0,len(raising_edge)):
            segm_extr.append([])
            for k in range(0,len(self.chans_names)):
                if(l<len(falling_edge)):
                    segment=self.raw_signals[k][raising_edge[l]:falling_edge[l]]
                else:
                    segment=self.raw_signals[k][raising_edge[l]:-1]                
                segm_extr[l].append(segment) 
            self.sigments_labels.append([])
            
        self.sigments_start_t.append(self.time[raising_edge[l]])        
        self.sigments_sign=segm_extr
        

def OpenDataFromFolder(PATH="",
                       SEGMENTATION_REF_CHAN_NAME="Trigger",
                       SEGMENTATION_THRESHOLD="automatic",
                       SEGMENTATION_SEGMENTS_NAMES_LIST=[]
                      ):
    
    arr = next(os.walk(PATH))[2]
    arr_txt=[]
    arr_bin=[]
    for k in arr:
        filename, file_extension = os.path.splitext(k)
        if(file_extension==".txt"):
            arr_txt.append(k)
            for l in arr:
                filename_1, file_extension_1 = os.path.splitext(l)
                if((l!=k) and (file_extension_1==".bin") and (filename_1 in filename)):
                    arr_bin.append(l)
                    break
    plates=[]             
    for l in range(0,len(arr_txt)):
        path_txt=PATH+"\\"+arr_txt[l]
        path_bin=PATH+"\\"+arr_bin[l]    
        df,d_chan,samp_r= read_bin_data(path_txt,path_bin)
        cur_plate=SPlate(plate_name=arr_bin[l])
        cur_plate.segments_names = SEGMENTATION_SEGMENTS_NAMES_LIST
        cur_plate.get_data_from_df(df)
        cur_plate.get_segments(ref_chan_name=SEGMENTATION_REF_CHAN_NAME,threshold=SEGMENTATION_THRESHOLD)
        if(cur_plate.segments_names == []):
            for q in range(0,len(cur_plate.sigments_sign)):
                cur_plate.segments_names.append(SEGMENTS_MISSED_SEGMENT_NAME+str(q))
        elif(len(cur_plate.segments_names)<len(cur_plate.sigments_sign)):
            num_to_add=len(cur_plate.sigments_sign) - len(cur_plate.segments_names)
            for mm in range(0,num_to_add):
                cur_plate.segments_names.append(SEGMENTS_MISSED_SEGMENT_NAME+str(mm))
                
        plates.append(cur_plate)
        print_progress_bar(l+1, len(arr_txt), "Opening files")
        
    return plates

#find plates in list
def FindPlateInArray(plates = [],plate_name="",chan_name="",segm_name=""):                        
            if(len(plates) == 0):
                print("Plates list is empty")
                return -1,-1,-1
            #find plate in list            
            indx_plate=-1
            for k in range(0,len(plates)):
                if(plates[k].name==plate_name):
                    indx_plate=k
                    break            
            #find channel in list
            indx_chan=-1
            if(chan_name!="all"):
                for k in range(0,len(plates[indx_plate].chans_names)):
                    if(plates[indx_plate].chans_names[k]==chan_name):
                        indx_chan=k
                        break            
            #find the segment
            indx_segment=-1
            for k in range(0,len(plates[indx_plate].segments_names)):
                    if(plates[indx_plate].segments_names[k]==segm_name):
                        indx_segment=k
                        break                        
            return indx_plate, indx_chan, indx_segment

#devide signals into snippets
def SplitIntoSnips(plates=[],snip_size=50,plate_name="",chan_name="",segment_name=""):
    feat=[]
    labs=[]
    indx_plate,indx_chan, indx_segment = FindPlateInArray(plates=plates.copy(),plate_name=plate_name,chan_name=chan_name,segm_name=segment_name)
    
    for hj in range(0,len(plates[indx_plate].sigments_labels[indx_segment])):                            
                    label=plates[indx_plate].sigments_labels[indx_segment][hj][2]
                    first_el=plates[indx_plate].sigments_labels[indx_segment][hj][0]
                    last_el=plates[indx_plate].sigments_labels[indx_segment][hj][1]
                    snips_count=int(np.round((last_el-first_el)/snip_size))
                    st=0
                    for k in range(0,snips_count):
                        labs.append(label)
                        if(indx_chan==-1):
                            feat_tmp=np.empty
                            for b in range(0,len(plates[indx_plate].sigments_sign[indx_segment])):
                                feat_tmp_1=np.asarray(plates[indx_plate].sigments_sign[indx_segment][b][st:st+snip_size])                                  
                                feat_tmp=np.concatenate((feat_tmp,feat_tmp_1),axis=None)
                        else:
                            feat_tmp=np.asarray(plates[indx_plate].sigments_sign[indx_segment][indx_chan][st:st+snip_size])
                        st=st+snip_size
                        feat.append(feat_tmp)
    return feat, labs

#show signals and labelling in the figure
def ShowSignalInFigure(fig, plates=[],colors_code=[],indx_plate=0,indx_segment=0,indx_chan=0):
    #%matplotlib qt
    if(indx_chan!=-1):#this is the case of not all channels are selected     
        #if(plt.fignum_exists(fig_id)==False):
        #    fig=plt.figure(fig_id)                    
        fig.clf()        
        plt.plot(plates[indx_plate].sigments_sign[indx_segment][indx_chan])
        if(plates[indx_plate].sigments_labels[indx_segment]!=[]):#if labels are assigned then show those                
            for k in plates[indx_plate].sigments_labels[indx_segment]:                
                start=k[0]
                end=k[1]
                c_num=k[2]
                plt.axvspan(start, end, facecolor=colors_code[c_num], alpha=0.5)       
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.show()
    else:
        #if(plt.fignum_exists(fig_id)==False):
        #    fig=plt.figure(fig_id)        
        fig.clf()
        for kp in range(0,len(plates[indx_plate].chans_names)):                    
           plt.plot(plates[indx_plate].sigments_sign[indx_segment][kp])
        if(plates[indx_plate].sigments_labels[indx_segment]!=[]):#if labels are assigned then show those
            print(len(plates[indx_plate].sigments_labels[indx_segment]))            
            for k in plates[indx_plate].sigments_labels[indx_segment]:                
                start=k[0]
                end=k[1]
                c_num=k[2]
                plt.axvspan(start, end, facecolor=colors_code[c_num], alpha=0.5)
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.show()        