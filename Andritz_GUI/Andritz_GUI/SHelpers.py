from turtle import width
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
import time
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import logging
import threading
import sys
import trace
import json
import warnings

#additional libs
import librosa
import csv
#classifiers
from xgboost import XGBClassifier

import torch
import torchaudio
from torch import nn
import torchaudio.functional as F
import torchaudio.transforms as T
#this is needed for Autoencoder_1
from torch.utils.data import DataLoader
import pickle as pl
import torch.nn as nn
import torch.nn.functional as F
from barbar import Bar
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms, utils

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from   PySide6.QtWidgets import QMainWindow


SEGMENTS_MISSED_SEGMENT_NAME="noname_"
LABEL_DEFAULT_NAME="Label_"

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

def ShowResultsInFigure_AllSegmentsInRow(signals,labels,chans_indx,snip_size,colorscheme,fig,fig_ax,orig_label_tags):    
    """
    if plt.fignum_exists(fig_id)==False:
        fig=plt.figure(fig_id)        
    else:
        fig=plt.figure(fig_id)           
    fig.clf() 
    ax= fig.add_subplot(111)
    """   
    fig_ax.clear()
    
    unique_l=[]
    colors_l=[]
    line_v=[]
    ymin=[]    
    ymax=[]
    ymin_tmp=[]
    ymax_tmp=[]
    legend_tags=[]
    
    borders=[]
    borders.append(0)
    
    for kl in range(0,len(signals)):
        #ymin.append(np.min(signals[kl]))
        #ymax.append(np.max(signals[kl]))        
        start_=borders[kl]
        l_shp=np.shape(np.asarray(signals[kl]))
        if(len(l_shp)==2): l_=l_shp[1]
        if(len(l_shp)==1): l_=l_shp[0]
        end_=start_+l_
        borders.append(end_)
        x_=np.arange(start_,end_)

        ymin_tmp=[]
        ymax_tmp=[]
        for nnb in range(0,len(chans_indx)):
            ymin_tmp.append(np.min(signals[kl][nnb]))
            ymax_tmp.append(np.max(signals[kl][nnb]))
            fig_ax.plot(x_,signals[kl][nnb],color="blue")     
        ymin.append(np.min(np.asarray(ymin_tmp)))
        ymax.append(np.max(np.asarray(ymax_tmp)))

        st_=start_
        en_=st_+snip_size
        
        #fill the backgrounds
        for l in range(0,len(labels[kl])):            
            cur_color=colorscheme[int(labels[kl][l])]
            line_=fig_ax.axvspan(st_, en_, alpha=0.1, color=cur_color)
            st_=en_
            en_=en_+snip_size   
            if(int(labels[kl][l]) not in unique_l):
                unique_l.append(int(labels[kl][l]))
                colors_l.append(cur_color)
                line_v.append(line_)                            
                
    for l in range(0,len(unique_l)):
       if(orig_label_tags is not None):
           legend_tags.append(str(orig_label_tags[l]))  
       else:
           legend_tags.append(str(unique_l[l]))  

    for jj in range(0,len(borders)):
        if(jj<len(borders) and jj<len(ymin) and jj<len(ymax)):
            fig_ax.vlines(borders[jj], ymin[jj], ymax[jj], colors="red")
            
    fig_ax.legend(line_v,legend_tags)
    fig.canvas.draw()
    fig.canvas.flush_events()
    #fig_ax.update()
    #plt.show()

#************************************************************************************************************************************
#************************************************************************************************************************************
#****************************************************SHOW SPECTRGRAMS****************************************************************
#https://docs.pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

#************************************************************************************************************************************
#************************************************************************************************************************************
#****************************************************PLATE***************************************************************************

class SPlate:

    def __init__(self,plate_name=""):
        self.name=plate_name
        self.raw_signals=[]
        self.time=[]
        self.chans_names=[]
        self.segments_names=[]                                     #names of the segments
        self.sigments_sign=[]                                     #signals for each signal
        self.sigments_start_t=[]    #start time of each segment
        self.sigments_labels=[]     #we store the labelling results in segments for supervised classwification, format - [start el,end el,label]
        self.delta_t=0
        self.snipp_l=[]                                                     #snippets length (i.e. number of sampling points)
        self.sr=[] 

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

    def get_segments_PRECITEC_TORCH(self):
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        indx = self.chans_names.index("Area")               
        
        cnt_area=0
        cnt_segm=0
        start=[]        
        trigger=torch.tensor(self.raw_signals[indx]).to(device)
        
        for k in range(0,len(self.raw_signals[indx])):            
            if(k==0):start.append(k)
            if(trigger[k]!=trigger[k-1]):
                start.append(k-1)
                start.append(k)
            start.append(len(self.raw_signals[indx]))
                         
        segments=[]
        raw_sign_list_torch=[]
        for l in range(0,len(self.raw_signals)):
            segments.append([])
            signs=torch.tensor(self.raw_signals).to(device)
            raw_sign_list_torch.append(signs)
            
        for l in range(0,len(self.raw_signals)):
            tmp_sign=[]
            for p in range(0,len(start),2):
                if(device=="cuda"):
                    tmp_sign.append((raw_sign_list_torch[l][start[p]:start[p+1]]).cpu().numpy())
                if(device=="cpu"):
                    tmp_sign.append((raw_sign_list_torch[l][start[p]:start[p+1]]).numpy())
            segments[l].append(tmp_sign)
        return segments
    
    def get_segments_PRECITEC(self):
        #device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        indx = self.chans_names.index("Area")               
        #unique=GetUniqueElements_List(self.raw_signals[indx])        
        #for k in range(0,len(unique)): segments.append([])
        cnt_area=0
        cnt_segm=0
        start=[]        
        for k in range(0,len(self.raw_signals[indx])):
            if(k==0):start.append(0)
            else:
                if(self.raw_signals[indx][k]!=self.raw_signals[indx][k-1]):
                    start.append(k)
        start.append(len(self.raw_signals[indx]))
                         
        segments=[]
        tmp_sign=[]

        for p in range(0,len(start)-1):        
            segments.append([])
            tmp_sign=[]
            #print(len(self.raw_signals))
            for lks in range(0,len(self.raw_signals)):    
                if((start[p+1]-start[p])>0):
                    #tmp_sign.append(self.raw_signals[lks][start[p]:start[p+1]])
                    segments[p].append(self.raw_signals[lks][start[p]:start[p+1]])
            #print(len(tmp_sign))
        self.sigments_sign=[]
        self.sigments_sign=segments        
        self.sigments_labels=[]
        if(len(segments)!=0):
            for ip in range(0,len(segments)):self.sigments_labels.append([])
    
    def get_segments(self,ref_chan_name="",threshold="automatic"):
        indx_trig=-1
        if(isinstance(ref_chan_name,str)):
            indx_trig = self.chans_names.index(ref_chan_name) 
        if(isinstance(ref_chan_name,int)):
            indx_trig=ref_chan_name
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

    #assign the lebel to selected segment pattern
    def AssignLabelToSegmentPattern(self,segment_indx=0,label=LABEL_DEFAULT_NAME,start_el=0,end_el=2):
        if(segment_indx>len(self.sigments_sign)):
            print("")
            print("Assigning label to segment pattern failed. Segment index is greater then the number of segments of the plate.")
            return
        if(label=="" or label==" "):
            print("Assigning label to segment pattern failed. Label tag is empty - fill and repeat operation.")
            return            
        try:            
            self.sigments_labels[segment_indx].append(list([start_el,end_el,label]))
        except:
            print("")
            print("Assigning label to segment pattern failed. Check the segments signals length or content.")
    
    #assign a label to entire segment
    def AssignLabelToEntireSegment(self,segment_indx=0,label=LABEL_DEFAULT_NAME):
        try:
            start_el = 0
            shp=np.shape(np.asarray(self.sigments_sign[segment_indx]))
            if(len(shp)==1):
                end_el=int(shp[0])
            else:
                end_el=int(shp[1])
            label_=LABEL_DEFAULT_NAME
            self.AssignLabelToSegmentPattern(segment_indx=segment_indx,
                                             label=label,
                                             start_el=start_el,
                                             end_el=end_el)
        except:
            print("")
            print("Assigning entire segment to label failed. Check segments signals content and repeat.")
            try:  print("Segment "+str(segment_indx)+" shape is "+str(np.shape(np.asarray(self.sigments_sign[segment_indx][1]))))
            except: print("Unable to establish the shape of the segment signals (probably those are inconsistent.)")
            return

    #all segments of this plate are assigned the given lebel
    def AssignLabelToAllSegments(self,label=LABEL_DEFAULT_NAME):
        if(len(self.sigments_sign)==0):
            print("")
            print("Assigning all segments to lable failed. This segment signals are empty or content is corrupted. Check and repeat later.")
        for l in range(0,len(self.sigments_sign)):
            self.AssignLabelToEntireSegment(segment_indx=l,label=label)            

    def GetUniqueLabelsList(self):
        l_sgm=len(self.sigments_sign)
        labels_list=[]
        for l in range(0,l_sgm):
            #self.sigments_labels[segment_indx].append(list([start_el,end_el,label]))
            cur_l = len(self.sigments_labels[l])
            for pp in range(0,cur_l):
                labels_list.append(self.sigments_labels[l][pp][2])
        #unique_l=list(set(labels_list))
        unique_l=GetUniqueElements_List(labels_list)
        return unique_l

def GetUniqueElements_List(inputList):
    list_out=[]
    for k in range(0,len(inputList)):
        if(inputList[k] not in list_out):
            list_out.append(inputList[k])
    return list_out
        

def OpenDataFromFolder(PATH="",
                       SEGMENTATION_REF_CHAN_NAME="Trigger",
                       SEGMENTATION_THRESHOLD="automatic",
                       SEGMENTATION_SEGMENTS_NAMES_LIST=[],
                       ONLY_SINGLE_FILE=False,
                       SINGLE_FILE_PATH_BIN="",
                       SINGLE_FILE_PATH_TXT="",
                       SINGLE_FILE_PATH_CSV="",
                      ):

    #print(SEGMENTATION_SEGMENTS_NAMES_LIST)            
    arr_txt=[]#spectrum files
    arr_bin=[]#spectreum files
    arr_csv=[]
    if(ONLY_SINGLE_FILE==False):        
        arr = next(os.walk(PATH))[2]
        for k in arr:
            filename, file_extension = os.path.splitext(k)
            if(file_extension==".txt"):
                k_1=PATH+"\\"+k
                arr_txt.append(k_1)
                for l in arr:
                    filename_1, file_extension_1 = os.path.splitext(l)
                    if((l!=k) and (file_extension_1==".bin") and (filename_1 in filename)):
                        l_1=PATH+"\\"+l
                        arr_bin.append(l_1)
                        break
            if(file_extension==".csv"):
                arr_csv.append(k)                
    else:        
        arr_txt.append(SINGLE_FILE_PATH_TXT)
        arr_bin.append(SINGLE_FILE_PATH_BIN)
        arr_csv.append(SINGLE_FILE_PATH_CSV)
        
    plates=[]             
    #we open the classical binary format from SPECTRUM cards
    if(len(arr_txt)!=0 and len(arr_bin)!=0):
        for l in range(0,len(arr_txt)):
            #if(ONLY_SINGLE_FILE==False):
            #    path_txt=ONLY_SINGLE_FILE+"\\"+arr_txt[l]
            #    path_bin=ONLY_SINGLE_FILE+"\\"+arr_bin[l]    
            #else:
            path_txt=arr_txt[l]
            path_bin=arr_bin[l]
            
            df,d_chan,samp_r= read_bin_data(path_txt,path_bin)
            cur_plate=SPlate(plate_name=arr_bin[l])
            cur_plate.sr=samp_r
            cur_plate.segments_names = SEGMENTATION_SEGMENTS_NAMES_LIST
            cur_plate.get_data_from_df(df)
            cur_plate.get_segments(ref_chan_name=SEGMENTATION_REF_CHAN_NAME,threshold=SEGMENTATION_THRESHOLD)

            if(cur_plate.segments_names == []):
                for q in range(0,len(cur_plate.sigments_sign)):
                    cur_plate.segments_names.append(SEGMENTS_MISSED_SEGMENT_NAME+str(q))
            elif(len(cur_plate.segments_names)<len(cur_plate.sigments_sign)):
                num_to_add=len(cur_plate.sigments_sign) - len(cur_plate.segments_names)
                for mm in range(0,num_to_add):
                    cur_plate.segments_names.append(SEGMENTS_MISSED_SEGMENT_NAME+str(len(cur_plate.sigments_sign)+mm))
            elif(len(cur_plate.segments_names)>len(cur_plate.sigments_sign)):
                num_to_remove = len(cur_plate.segments_names) - len(cur_plate.sigments_sign)
                for k in range(0,num_to_remove):
                    del cur_plate.segments_names[-1]
                    
            plates.append(cur_plate)
            print("")
            print_progress_bar(l+1, len(arr_txt), "File opened (SPECTRUM *.bin)")    

    #csv format
    
    if(len(arr_csv)!=0) and (len(arr_csv)==1 and arr_csv[0]!=""):        
        cnt=0
        files_cnt=0
        for l in arr_csv:
            path_csv=PATH+"\\"+l
            #precitec format - multicolumn csv
            #try:
            with open(path_csv, mode ='r') as file:
                cur_plate=SPlate(plate_name=l)
                csvFile = csv.reader(file)
                cnt=0
                
                time=[]
                signals=[]
                
                for line in csvFile:                    
                    cnt+=1
                    if(cnt==3):                        
                        spl_str=line[0].split(":")                        
                        sr=-1
                        try: sr=int(spl_str[1].split(" ")[1])
                        except: pass
                        if(spl_str[1].split(" ")[2]=="kHz"):
                            sr=sr*1000
                        if(spl_str[1].split(" ")[2]=="MHz"):
                            sr=sr*1000000
                        cur_plate.sr=sr
                    if(cnt==10):
                        #headers
                        str_columns=line[0].split(";")
                        cur_plate.chans_names=[]
                        for lp in range(2,len(str_columns)):
                            signals.append([])
                            cur_plate.chans_names.append(str_columns[lp])                        
                    if(cnt>10):
                        str_line=line[0].split(";")
                        sgn_count=0
                        for gh in range(0,len(str_line)):
                            if(gh==0):pass
                            elif(gh==1): time.append(float(str_line[gh]))
                            else:
                                signals[sgn_count].append(float(str_line[gh]))
                                sgn_count+=1
               
                cur_plate.time=[]
                cur_plate.time=time
                cur_plate.raw_signals=[]
                cur_plate.raw_signals=signals
                cur_plate.get_segments_PRECITEC()#cur_plate.get_segments_PRECITEC()
                cur_plate.segments_names = SEGMENTATION_SEGMENTS_NAMES_LIST
                
                if(len(cur_plate.sigments_sign)!=0):
                    
                    segm_num=len(cur_plate.sigments_sign)#we assume the number of segments in all channels is the same (as originated from trigger channel
                    #print(segm_num)
                    if(cur_plate.segments_names == []):
                        for q in range(0,segm_num):
                            cur_plate.segments_names.append(SEGMENTS_MISSED_SEGMENT_NAME+str(q))
                    elif(len(cur_plate.segments_names)<len(cur_plate.sigments_sign)):
                        num_to_add=segm_num - len(cur_plate.segments_names)
                        for mm in range(0,num_to_add):
                            cur_plate.segments_names.append(SEGMENTS_MISSED_SEGMENT_NAME+str(len(cur_plate.sigments_sign)+mm))
                    elif(len(cur_plate.segments_names)>len(cur_plate.sigments_sign)):
                        num_to_remove = len(cur_plate.segments_names) - segm_num
                        for k in range(0,num_to_remove):
                            del cur_plate.segments_names[-1]
                    
                plates.append(cur_plate)
            
            #except: pass
            files_cnt+=1
            print("")  
            print_progress_bar(files_cnt, len(arr_csv), "File opened (Precitec *.csv)")
    print("")            
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


#https://www.pythonguis.com/tutorials/pyside-plotting-matplotlib/
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        #ADD SUBPLOTS - DO IT HERE
        #this solution have to be tested for stability in multithreading
        # self.axes1 = self.fig.add_subplot(121) # add in the same row or use (211) to place them one under the other
        super().__init__(self.fig)
        
#https://www.pythonguis.com/tutorials/creating-multiple-windows/
class ChartWindow(QMainWindow):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self,chart_name="Window",width=5,height=4,dpi=100):
        super().__init__()
        #layout = QVBoxLayout()
        #self.label = QLabel(window_name)
        #layout.addWidget(self.label)
        #self.setLayout(layout)
        self.chart_name=chart_name
        self.Canvas=MplCanvas(self, width=width, height=height, dpi=dpi)
        self.Canvas.axes.set_title(self.chart_name)
        #Canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40])  
        self.setCentralWidget(self.Canvas)
        #self.show()

#show all segments with labels assigned by user (for the given plate)
def ShowAllSingleSegmentsWithLabels(fig_id, 
                                    plate,
                                    colors_code=None,
                                    indx_chan=0,
                                    aplpha=0.1,
                                    show_labels=True, #this is for ground truth labels
                                    points_num_limit_check=False,
                                    points_num_limit=3000,
                                    show_proc_labels=False, #this is for processed labels
                                    proc_labels_snip_size=100,
                                    proc_labels_color_scheme="only_anom",
                                    proc_labels_show_segm_borders=True,
                                    proc_labels=[]
                                    ):
    
    warnings.filterwarnings( "ignore")

    chan_num=0    
    if isinstance(indx_chan, list)==True:    chan_num=indx_chan
    if isinstance(indx_chan, np.ndarray)==True:chan_num=list(indx_chan)        
    if isinstance(indx_chan, int)==True:     chan_num=[indx_chan]    
    if indx_chan==-1:     chan_num=list(np.linspace(0,len(plate.chans_names),len(plate.chans_names)))

    unique_labels=plate.GetUniqueLabelsList()
    #print(unique_labels)
    indx_color=np.linspace(0,len(unique_labels),len(unique_labels))    

    sectors=[]
    labels_tags=[]
   
    border=[]
    border.append(0)
        
    if(isinstance(fig_id,str)): 
        fig=plt.figure(fig_id)     
        fig.clf()
        fig_ax= fig.add_subplot(111)
    if(isinstance(fig_id,plt.Figure)):         
        fig= fig_id   
        ax_list = fig.get_axes()
        fig_ax=ax_list[0]
        fig_ax.clear()
    if(isinstance(fig_id,ChartWindow)):        
        fig= fig_id.Canvas.fig
        fig_ax= fig_id.Canvas.axes
        fig_ax.clear()
    #fig.clf()
    #fig_ax= fig.add_subplot(111)
    

    #make full signals first
    full_sign=[]
    sample_stamp=[]    
    for kks in range(0,len(chan_num)):    
        full_sign.append([])       
        sample_stamp.append([])
        sample_stamp[-1].append(0)
        for segment in plate.sigments_sign:               
            if(len(full_sign[-1])==0):full_sign[-1]=np.asarray(segment[chan_num[kks]])                                
            else:
                sfg = np.asarray(full_sign[-1])            
                sfg1=np.concatenate((sfg,segment[chan_num[kks]]),axis=None)
                full_sign[-1]=np.empty
                full_sign[-1]=np.asarray(sfg1)            
                sample_stamp[-1].append(len(sfg1)+sample_stamp[-1][len(sample_stamp[-1])-1])            

    #sparse signal if needed
    step_size=1    
    if(points_num_limit_check==True):
        for m in range(0,len(full_sign)):
            shp_=np.shape(full_sign[m])
            l_segm_=-1
            if(len(shp_)==1): l_segm_=shp_[0]
            else:            l_segm_=shp_[1]
            if l_segm_>points_num_limit:
                step_size=int(l_segm_/points_num_limit)
                reduced_signal=[]
                points_steps=[]
                for kk in range(0,l_segm_,step_size):
                    reduced_signal.append(full_sign[m][kk])
                    points_steps.append(kk)                     
                fig_ax.plot(points_steps,reduced_signal)
            else:                 
                fig_ax.plot(full_sign[m])
    else:                
        for m in range(0,len(full_sign)):
            fig_ax.plot(full_sign[m])


    if(show_labels==True):
        for kkj in range(0,len(chan_num)):
            cnt=0            
            for k in range(0,len(plate.sigments_labels)):                
                if (k!=0): cnt=cnt+len(plate.sigments_sign[k-1][kkj])
                for j in range(0,len(plate.sigments_labels[k])):     
                    curs_start=int((plate.sigments_labels[k][j][0]+cnt))
                    curs_end=int((plate.sigments_labels[k][j][1]+cnt))
                    label=plate.sigments_labels[k][j][2]                
                    c_ind=unique_labels.index(label)
                    if(c_ind>len(colors_code)-1): c_ind=len(colors_code)-1
                    sector_=plt.axvspan(curs_start, curs_end, facecolor=colors_code[c_ind], alpha=aplpha)    
                    if(label not in labels_tags):
                       labels_tags.append(label)
                       sectors.append(sector_)
                           
    if(show_proc_labels==True) and (proc_labels is not None) and (len(proc_labels)!=0):
                
        
        min_v=np.min(np.asarray(full_sign),axis=None)
        max_v=np.max(np.asarray(full_sign),axis=None)

        show_scheme=""
        if(isinstance(proc_labels_color_scheme,int)):
            if(proc_labels_color_scheme==0):show_scheme="all_grades"
            else: show_scheme="only_anom" 
        else: show_scheme=proc_labels_color_scheme

        #if(proc_labels_color_scheme=="all"):                   
        cnt = 0        
        chan_n=chan_num[0]
        for p in range(0,len(proc_labels)):
            if(p>0): 
                cnt=cnt+len(plate.sigments_sign[p][chan_n])
                if (proc_labels_show_segm_borders==True): 
                    fig_ax.vlines(cnt,ymin=min_v,ymax=max_v,colors="black",linestyles="solid")
            
            for k in range(0,len(proc_labels[p])):

                st_ = k * proc_labels_snip_size + cnt
                en_ = st_ + proc_labels_snip_size

                cur_l = int(proc_labels[p][k])
                cur_color=colors_code[cur_l % len(colors_code)]
                if(show_scheme=="all_grades"):    
                    fig_ax.axvspan(st_, en_, alpha=0.1, color=cur_color) 
                if(show_scheme=="only_anom"):    
                    if(cur_l!=0): fig_ax.axvspan(st_, en_, alpha=0.1, color=cur_color) 

                st_= en_
    """
            shp_=np.shape(plate.sigments_sign[indx_segment][chan_num[i]])
            l_segm_=-1
            if(len(shp_)==1): l_segm_=shp_[0]
            else:            l_segm_=shp_[1]
            if l_segm_>points_num_limit:
                step_size=int(l_segm_/points_num_limit)
                reduced_signal=[]
                points_steps=[]
                for kk in range(0,l_segm_,step_size):
                    reduced_signal.append(plate.sigments_sign[indx_segment][chan_num[i]][kk])
                    points_steps.append(kk)
                plt.plot(points_steps,reduced_signal)
            else:
                plt.plot(plate.sigments_sign[indx_segment][chan_num[i]])
            """
    """
    for kks in range(0,len(plate.sigments_sign)):        
        shp=np.shape(plate.sigments_sign[kks])
        l_segm=-1
        if(len(shp)==1): l_segm=shp[0]
        else:            l_segm=shp[1]
        if len(shp)==0: continue
        
        start=border[kks]
        end=start+l_segm       
        x_axis=np.linspace(start,end,l_segm)    
        border.append(end)
        #print("XXXXXXXXXXXXXX")
        #print(border[kks])
        #print(l_segm)        
        
        for i in range(0,len(chan_num)):           
            y_min=np.min(plate.sigments_sign[kks][chan_num[i]])
            y_max=np.max(plate.sigments_sign[kks][chan_num[i]])            
            
            plt.plot(x_axis,plate.sigments_sign[kks][chan_num[i]],color="blue")

            if(plate.sigments_labels[kks]!=[]) and (show_labels==True):
                for k in plate.sigments_labels[kks]:     
                    curs_start=k[0]+start
                    curs_end=k[1]+start
                    label=k[2]                
                    c_ind=unique_labels.index(label)
                    sector_=plt.axvspan(curs_start, curs_end, facecolor=colors_code[c_ind], alpha=aplpha)    
                    if(label not in labels_tags):
                        labels_tags.append(label)
                        sectors.append(sector_)                
        plt.vlines(border[kks], y_min,y_max,colors="red", linestyles='solid')
    """

    fig.legend(sectors,labels_tags)
    fig.canvas.draw_idle() #draw()
    fig.canvas.flush_events()
    fig.show()    
    #plt.show()
    
    #print(border)
#use example
#ShowAllSingleSegmentsWithLabels("ssdfsfsdf", PLATES_ARRAY[0],colors_code=cc_dd,indx_chan=0,aplpha=0.1)

#show signals and labelling in the figure
def ShowSingleSegmentWithLabels(fig_id, plate,
                                indx_segment=0,
                                colors_code=None,
                                indx_chan=0,
                                show_labels=True,
                                aplpha=0.1,
                                points_num_limit_check=False,
                                points_num_limit=3000
                                ):
    chan_num=0    
    if isinstance(indx_chan, list)==True:    chan_num=indx_chan
    if isinstance(indx_chan, np.ndarray)==True:chan_num=list(indx_chan)        
    if isinstance(indx_chan, int)==True:     chan_num=[indx_chan]    
    if indx_chan==-1:     chan_num=list(np.linspace(0,len(plate.chans_names),len(plate.chans_names)))
    
    unique_labels=plate.GetUniqueLabelsList()    
    indx_color=np.arange(0,len(unique_labels),dtype=int)#,len(unique_labels)) #CHECK FUNCTIONALITY
    print(indx_color)
    print(unique_labels)
    sectors=[]
    labels_tags=[]

    #%matplotlib qt
    if plt.fignum_exists(fig_id)==True:
        fig=plt.figure(fig_id) #fig=Figure(fig_id)
    else:
        fig=plt.figure(fig_id) 
    fig.clf()  
    
    for i in range(0,len(chan_num)):
        if(points_num_limit_check==False):
            plt.plot(plate.sigments_sign[indx_segment][chan_num[i]])
        else:#show in sparse to be faster
            shp_=np.shape(plate.sigments_sign[indx_segment][chan_num[i]])
            l_segm_=-1
            if(len(shp_)==1): l_segm_=shp_[0]
            else:            l_segm_=shp_[1]
            if l_segm_>points_num_limit:
                step_size=int(l_segm_/points_num_limit)
                reduced_signal=[]
                points_steps=[]
                for kk in range(0,l_segm_,step_size):
                    reduced_signal.append(plate.sigments_sign[indx_segment][chan_num[i]][kk])
                    points_steps.append(kk)
                plt.plot(points_steps,reduced_signal)
            else:
                plt.plot(plate.sigments_sign[indx_segment][chan_num[i]])

        if(plate.sigments_labels[indx_segment]!=[]) and (show_labels==True):
            for k in plate.sigments_labels[indx_segment]:     
                start=k[0]
                end=k[1]
                label=k[2]   
                index_label=unique_labels.index(label)                
                c_ind=indx_color[index_label]
                sector_=plt.axvspan(start, end, facecolor=colors_code[c_ind], alpha=aplpha)    
                if(label not in labels_tags):
                    labels_tags.append(label)
                    sectors.append(sector_)
    fig.legend(sectors,labels_tags)
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show() 
    
    """
    #%matplotlib qt
    if(indx_chan!=-1):#this is the case when all channels are selected     
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
        """

#classifiers

def XGBoostClassifRun(X_train=[], X_test=[], y_train=[], y_test=[]):
                bst=None
                xgb_labs_back=[]
                try:
                    #now adjsut the labels -  the reason is that xgbboost wants the labels start from 0, i.e. 0,1,2....
                    unique_labels=list(set(y_train))    #list(set(labs))
                    #now adjsut the labels -  the reason is that xgbboost wants the labels start from 0, i.e. 0,1,2....
                    xgb_unique_labs=[]
                    for l in range(0,len(unique_labels)):
                        xgb_unique_labs.append(l)
                    xgb_lab_train=y_train.copy()
                    xgb_lab_test=y_test.copy()
                    print(unique_labels)                
                    for i in range(0,len(xgb_lab_train)):
                        indx=unique_labels.index(xgb_lab_train[i])
                        xgb_lab_train[i]=int(xgb_unique_labs[indx])
                    for i in range(0,len(xgb_lab_test)):
                        indx1=unique_labels.index(xgb_lab_test[i])
                        xgb_lab_test[i]=int(xgb_unique_labs[indx1])                    
                    #print("xgb_test_labs:"+str(xgb_lab_test))            
                    #print("unique_labels:"+str(unique_labels))                              
                    bst = XGBClassifier(n_estimators=8000, max_depth=12, learning_rate=0.01, objective='binary:logistic')
                    bst.fit(X_train, xgb_lab_train)
                    preds = bst.predict(X_test)
                    #show in figure  
                    xgb_labs_back=preds.copy()
                    for i in range(0,len(xgb_lab_test)):
                        indx2=xgb_unique_labs.index(xgb_labs_back[i])
                        xgb_labs_back[i]=int(unique_labels[indx2])        
                except Exception as ex:
                    print(str(ex))                    
                return bst, xgb_labs_back,unique_labels,xgb_unique_labs

def IsolTreesTraining(feat=None,labels=None):
    if(feat is None) or (len(feat)==0): 
        print("")
        print("features are emopty - training aborted...")
        return
    
    unique_labs=list(set(labels))
    if(len(unique_labs)==0):
        print("No labeled data is defined. One label is needed for training...")
        pass
    #https://www.eyer.ai/blog/anomaly-detection-in-time-series-data-python-a-starter-guide/            
    x_sf=[]
    ref_l=labels[0]
    for m in range(0,len(labels)):
        if(ref_l==labels[m]):
            cur_feat=(np.asarray(feat)[m,:])
            x_sf.append(cur_feat)
    shp_training_set=np.shape(np.asarray(x_sf))
    print("")
    print("Isolation trees dataset inlcudes lab. - "+str(ref_l))
    print("Trainset shape - "+str(shp_training_set))
    print("Isolation trees training...")
    clf = IsolationForest(n_estimators=8000, contamination=0.01)
    start_t = time.time()
    clf.fit(np.asarray(x_sf))
    end_t = time.time()
    print("Training is done in "+str(end_t-start_t)+ str(" s"))
    start_t = time.time()
    y_pred = clf.predict(x_sf)
    end_t = time.time()
    print("Pridiction time: "+str(end_t-start_t)+" s for " +str(shp_training_set[0])+" features")
    return clf
#IsolTreesTraining(feat=CLASSIF_FEAT,labels=CLASSIF_LABS)

def DBSCANTraining(feat=None,labels=None):
    if(feat is None) or (len(feat)==0): 
        print("")
        print("features are empty - training aborted...")
        return None
    
    unique_labs=list(set(labels))
    if(len(unique_labs)==0):
        print("No labeled data is defined. One label is needed for training...")
        pass
    #https://www.eyer.ai/blog/anomaly-detection-in-time-series-data-python-a-starter-guide/            
    x_sf=[]
    ref_l=labels[0]
    for m in range(0,len(labels)):
        if(ref_l==labels[m]):
            cur_feat=(np.asarray(feat)[m,:])
            x_sf.append(cur_feat)
    shp_training_set=np.shape(np.asarray(x_sf))
    print("")
    print("DBSCAN dataset inlcude lab. - "+str(ref_l))
    print("Trainset shape - "+str(shp_training_set))
    print("DBSCAN training...")
    clf = DBSCAN(eps=0.1, min_samples=41)
    start_t = time.time()
    clf.fit(np.asarray(x_sf))
    end_t = time.time()
    print("Training is done in "+str(end_t-start_t)+ str(" s"))
    start_t = time.time()
    y_pred = clf.fit_predict(x_sf)
    end_t = time.time()
    print("Pridiction time: "+str(end_t-start_t)+" s for " +str(shp_training_set[0])+" features")
    return clf
#clg=DBSCANTraining(feat=CLASSIF_FEAT,labels=CLASSIF_LABS) 

def OneCLassSVMTraining(feat=None,labels=None):

    if(feat is None) or (len(feat)==0): 
        print("")
        print("features are empty - training aborted...")
        return None  

    unique_labs=list(set(labels))
    if(len(unique_labs)==0):
        print("No labeled data is defined. One label is needed for training...")
        pass

    x_sf=[]
    ref_l=labels[0]
    for m in range(0,len(labels)):
        if(ref_l==labels[m]):
            cur_feat=(np.asarray(feat)[m,:])
            x_sf.append(cur_feat)
    shp_training_set=np.shape(np.asarray(x_sf))
    print("")
    print("SVM dataset inlcude lab. - "+str(ref_l))
    print("Trainset shape - "+str(shp_training_set))
    print("DBSCAN training...")
    start_t = time.time()
    clf = OneClassSVM(nu=0.1, kernel="rbf",gamma=0.2).fit(feat)     
    end_t = time.time()
    print("Training is done in "+str(end_t-start_t)+ str(" s"))
    start_t = time.time()
    y_pred = clf.fit_predict(x_sf)
    end_t = time.time()
    print("Pridiction time: "+str(end_t-start_t)+" s for " +str(shp_training_set[0])+" features")
    return clf


#*****************************************************************************************
#******************************PREPROCESSING**********************************************
#*****************************************************************************************

    #------------------obtain spectrograms-------------------------
    #------------------as a preprocessing step---------------------
    #--------------------------------------------------------------
    
class DataPreproc:
    def __init__ (self):
        #SplitEntireSignalIntoSnippets
        self.sign_1=torch.empty
        self.sign_2=torch.empty
        self.sign_3=torch.empty
        self.sign_4=torch.empty      
        self.preproc_1=torch.empty
        self.proceed=False
        #SplitLabPlateSegmentIntoSnips
        self.sign_5=torch.empty   
        self.sign_6=torch.empty
        self.np_signal1=np.empty
        self.np_signal2=np.empty
        #preprocessing
        self.preproc=torch.empty                
        self.preproc_in=torch.empty     
        self.preproc_flat=torch.empty
        self.preproc_final=torch.empty
        #SplitLabPlateAllSegmentIntoSnips
        self.np_signal3=np.empty        
        self.sign_7=torch.empty         
        self.segm_snips_list=[]
        #SplitAllLabPlateSegmentIntoSnips        
        self.sign_8=torch.empty 
        self.sign_9=torch.empty 
        self.segm_labels_list=[]
        self.segm_sign_list=[]
        #SplitAllLabPlateOfAllSegmentsIntoSnips
        self.np_sign_4=torch.empty 
        self.segm_labels_list1=[]
        self.segm_sign_list1=[]       
        self.segm_labels_list2=[]
        self.segm_sign_list2=[]       
        
    #this splits the entire number of segments pf the plate
    def SplitAllPlateSegmentsIntoSnippets(self,plate, 
                                          channs_indx=[], 
                                          snip_size=5,
                                          torch_tensor=True, 
                                          preproc_type="None",         
                                          proc_time=False,
                                         ):
                
        segm_num=len(plate.sigments_sign)
        
        for t in range(0,segm_num):
            self.np_signal3=np.empty
            self.np_signal3=plate.sigments_sign[t]
            self.sign_7=self.SplitEntireSignalIntoSnippets(signal=self.np_signal3, 
                                                      channs_indx=channs_indx, 
                                                      snip_size=snip_size,
                                                      torch_tensor=True, 
                                                      preproc_type=preproc_type,
                                                      proc_time=False)            
            if(torch_tensor==False):
                feat = self.sign_7.cpu().numpy().copy()
                self.segm_snips_list.append(feat)
            else:
                self.segm_snips_list.append(self.sign_7.clone())
        return self.segm_snips_list.copy()

#pass through all plates and 
    def SplitAllLabPlateOfAllSegmentsIntoSnips(self,plates,
                                               snip_size=5,
                                               channs_indx=0,
                                               torch_tensor=False, 
                                               preproc_type="None",                                                
                                              ):
        self.segm_labels_list2=[]
        self.segm_sign_list2=[]       
        
        for pl in plates:
            self.segm_labels_list1=[]
            self.segm_sign_list1=[]       
            self.segm_sign_list1,self.segm_labels_list1=self.SplitAllLabPlateSegmentsIntoSnips(pl,
                                                                                               snip_size=snip_size,
                                                                                               channs_indx=channs_indx,
                                                                                               torch_tensor=torch_tensor, 
                                                                                               preproc_type=preproc_type,  
                                                                                               )
            self.segm_sign_list1,self.segm_labels_list1=self.Helper_FlatListOfLabeledFeat(self.segm_sign_list1,
                                                                                          self.segm_labels_list1
                                                                                         )
            for k in range(0,len(self.segm_sign_list1)):
                self.segm_labels_list2.append(self.segm_labels_list1[k])
                self.segm_sign_list2.append(self.segm_sign_list1[k])      

        return self.segm_sign_list2.copy(),self.segm_labels_list2.copy()

    #this splits all labeled segments into snippets for the given plate
    def SplitAllLabPlateSegmentsIntoSnips(self,plate,
                                         snip_size=5,
                                         channs_indx=-1,
                                         torch_tensor=False, 
                                         preproc_type="None",
                                        ):

        self.segm_labels_list=[]
        self.segm_sign_list=[]
        segm_num=len(plate.sigments_sign)
        
        for t in range(0,segm_num):           
            self.sign_8=torch.empty 
            self.sign_9=np.empty            
            self.sign_8,self.sign_9=self.SplitLabPlateSegmentIntoSnips(plate,
                                                                       snip_size=snip_size,
                                                                       segm_index=t,
                                                                       channs_indx=channs_indx,
                                                                       torch_tensor=True, 
                                                                       preproc_type=preproc_type,      
                                                                      )
            
            if(self.sign_9 is None or self.sign_8 is None):
                if(torch_tensor==True):
                    self.segm_sign_list.append(torch.empty)
                    self.segm_labels_list.append(np.empty)
                else:
                    self.segm_sign_list.append(np.empty)
                    self.segm_labels_list.append(np.empty)
            else:
                if(torch_tensor==True):
                    self.segm_sign_list.append(self.sign_8.clone())
                    self.segm_labels_list.append(np.asarray(self.sign_9.clone()))                    
                else:                    
                    self.segm_sign_list.append(self.sign_8.cpu().numpy().copy())
                    self.segm_labels_list.append(self.sign_9.copy())#.cpu().numpy().copy())
        return self.segm_sign_list.copy(),self.segm_labels_list.copy()
            
    #split obnly labeled data within a segment into snyppets
    def SplitLabPlateSegmentIntoSnips(self,plate,
                                           snip_size=5,
                                           segm_index=0,
                                           channs_indx=-1,
                                           torch_tensor=False, 
                                           preproc_type="None",
                                     ):

        self.np_signal1=np.empty
        self.np_signal2=np.empty
        self.sign_6=torch.empty
        
        lab=[]        
        labeled_data_l=len(plate.sigments_labels[segm_index])    
        self.np_signal1 = np.asarray(plate.sigments_sign[segm_index])        
        #service labels_names
        #lab_tmp=[]
        for hj in range(0,labeled_data_l):             
            #read labeled data first
            label=plate.sigments_labels[segm_index][hj][2]
            first_el=plate.sigments_labels[segm_index][hj][0]
            last_el=plate.sigments_labels[segm_index][hj][1]
            self.np_signal2=self.np_signal1[:,first_el:last_el].copy()         
            self.sign_5=torch.empty   
            #if(label not in lab_tmp): lab_tmp.append(label)
            #print(preproc_type)
            #print(self.np_signal2.shape)
            self.sign_5= self.SplitEntireSignalIntoSnippets(  signal=self.np_signal2, 
                                                              channs_indx=channs_indx, 
                                                              snip_size=snip_size,
                                                              torch_tensor=True, 
                                                              preproc_type=preproc_type,
                                                              proc_time=False
                                                            )       
            
            if(self.sign_5 is not None):
                if(self.sign_6==torch.empty):  self.sign_6=self.sign_5.clone()
                else: self.sign_6 = torch.cat([self.sign_6,self.sign_5],dim=0)     
                for k in range(0,self.sign_5.shape[0]):
                    lab.append(label)
            else: pass     
                
        if(torch_tensor==True):
                    if((self.sign_6!=torch.empty) and (self.sign_6 is not None)):
                        #le = preprocessing.LabelEncoder()
                        #lab_tags = le.fit_transform(lab)
                        #new_lab=[]
                        #tag_tmp=np.linspace(0,len(lab_tmp),len(lab_tmp))
                        #for xx in range(0,len(lab)):
                        #    indx=lab_tmp.index(lab[xx])
                        #    tag=tag_tmp[indx]
                        #    new_lab.append(tag)
                            
                        #torch_labels_1=torch.empty
                        #torch_labels_1=torch.tensor(np.asarray(new_lab))
                        #if(torch.cuda.is_available()): torch_labels_1.cuda()                    
                        feat=self.sign_6.clone()
                        return feat, np.asarray(lab.copy())#torch_labels_1
                    else: return None,None
        else:    
            if(self.sign_6==torch.empty):
                return None,None
            else:
                feat=self.sign_6.cpu().numpy().copy()
                return feat, np.asarray(lab.copy())               
        
    #split into snippets the given signal
    def SplitEntireSignalIntoSnippets(self,signal=None, 
                                      channs_indx=[], 
                                      snip_size=5,
                                      torch_tensor=True, 
                                      preproc_type="None",
                                      proc_time=False
                                     ):
        self.sign_1=torch.empty
        self.sign_2=torch.empty
        self.sign_3=torch.empty
        self.sign_4=torch.empty        

        if(proc_time==True):
            start_t = time.time()
        self.sign_1=torch.tensor(np.asarray(signal))
        if(torch.cuda.is_available()): self.sign_1.cuda()
        sign_shape=self.sign_1.shape   
        if(len(sign_shape)>1):
            num_snipps=np.round(sign_shape[1]/snip_size)
        if(len(sign_shape)<=1):
            num_snipps=np.round(sign_shape[0]/snip_size)
            
        for k in range(0,sign_shape[0]):            
            self.proceed = False
            if(type(channs_indx) is not list):
                if(channs_indx == -1): self.proceed=True  
                else: 
                    if(k==channs_indx): self.proceed=True  
            else:
                if(k in channs_indx): self.proceed=True  
                
            if(self.proceed==True):                
                self.sign_2=torch.split(self.sign_1[k],snip_size,dim=0)        
                self.sign_2 = list(self.sign_2)
                while(True): 
                    if(len(self.sign_2)<num_snipps):break                                                      
                    del self.sign_2[-1]
                self.sign_3=torch.stack(self.sign_2, dim=0) 
                #preprocessing
                if(preproc_type=="None" or  preproc_type=="" or preproc_type==" "): pass
                else: self.sign_3=self.DataPreprocessing(self.sign_3,preproc_type=preproc_type)
                #assign further
                if(self.sign_4 == torch.empty): self.sign_4 = self.sign_3 
                else: self.sign_4 = torch.cat([self.sign_4,self.sign_3],dim=1)#stack((self.sign_4,self.sign_3),dim=1) 
        if(proc_time==True):
            end_t = time.time()
            print("Feat. extr. time(s): "+str(end_t - start_t))
        if(torch_tensor==True):
            return self.sign_4.clone()
        else:
            if(self.sign_4.is_cuda):
                return self.sign_4.cpu().numpy().copy()      
            else: return self.sign_4.numpy().copy()      
                

    #------------------obtain spectrograms-------------------------
    #------------------as a preprocessing step---------------------
    #--------------------------------------------------------------
    def DataPreprocessing(self,signal,preproc_type="torch_MEL"):
        if(torch.cuda.is_available()):
            if(signal.is_cuda==False):
                signal.cuda()
                
        self.preproc=torch.empty
        self.preproc_final=torch.empty

        shp=signal.shape
        
        if(preproc_type=="torch_MEL"):
            #parameters
            n_fft = 128         #256       #Toni_MEL_nfft
            win_length = None
            hop_length = 2024   #2024  #Toni_MEL_Samp_hop_length
            n_mels = 512 #512   #self.Toni_MEL_Samp_n_mels
            sample_rate = 1000000 #40000 #self.Toni_MEL_samp_rate
            #MEL decomposition            
            for l in range(0,shp[0]):
                self.preproc_in=torch.empty
                self.preproc_in=signal[l,:].reshape((1, -1))                
                #https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
                MEL_transoform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                                n_fft=n_fft,
                                                                                win_length=win_length,
                                                                                hop_length=hop_length,
                                                                                center=True,
                                                                                pad_mode="reflect",
                                                                                power=2.0,
                                                                                norm="slaney",
                                                                                n_mels=n_mels,
                                                                                mel_scale="htk",).to(torch.double)   
                
                self.preproc = MEL_transoform(self.preproc_in)                
                self.preproc_flat = self.preproc.reshape(-1)    
                self.preproc_flat=self.preproc_flat.unsqueeze(0)                
                if(self.preproc_final==torch.empty): self.preproc_final = self.preproc_flat
                else:   self.preproc_final=torch.cat([self.preproc_final,self.preproc_flat],dim=0)                    
            return self.preproc_final.clone()
        #fft
        if(preproc_type=="torch_FFT"):
            #https://www.kaggle.com/code/yassinealouini/signal-processing-theory-and-practice-with-pytorch
            for l in range(0,shp[0]):
                self.preproc_in=torch.empty
                self.preproc_in=signal[l,:].reshape((1, -1))                
                self.preproc=torch.fft.fft(self.preproc_in)
                shp_proc=self.preproc.shape
                cut_half=int(shp_proc[1]/2)
                self.preproc=self.preproc[:,0:]
                if(self.preproc_final==torch.empty): self.preproc_final = self.preproc
                else:   self.preproc_final=torch.cat([self.preproc_final,self.preproc],dim=0)                       
            return self.preproc_final.clone()
        #mfcc spectrogram
        if(preproc_type=="torch_MFCC"):
            #parameters
            n_mfcc=13
            sample_rate = 1000000 #40000 
            
            #MEL decomposition            
            for l in range(0,shp[0]):
                self.preproc_in=torch.empty
                self.preproc_in=signal[l,:].reshape((1, -1))                
                #https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
                MFCC_transoform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                            n_mfcc=n_mfcc,
                                                            melkwargs={"n_fft": 50, "hop_length": 50, "n_mels": 23, "center": False},
                                                           ).to(torch.double)   
                
                self.preproc = MFCC_transoform(self.preproc_in)                
                self.preproc_flat = self.preproc.reshape(-1)    
                self.preproc_flat=self.preproc_flat.unsqueeze(0)                
                if(self.preproc_final==torch.empty): self.preproc_final = self.preproc_flat
                else:   self.preproc_final=torch.cat([self.preproc_final,self.preproc_flat],dim=0)                    
            return self.preproc_final .clone()

        #widow fft spectrogram
        if(preproc_type=="torch_WFFT"):
            #parameters
            n_fft=50
            win_len=None
            hop_len=None
            power=2.0
            
            #MEL decomposition            
            for l in range(0,shp[0]):
                self.preproc_in=torch.empty
                self.preproc_in=signal[l,:].reshape((1, -1))                
                #https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
                WFFT_transoform = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                                    win_length=win_len,
                                                                    hop_length=hop_len,
                                                                    center=True,
                                                                    pad_mode="reflect",
                                                                    power=power,
                                                                    ).to(torch.double)              
                self.preproc = WFFT_transoform(self.preproc_in)                
                self.preproc_flat = self.preproc.reshape(-1)    
                self.preproc_flat=self.preproc_flat.unsqueeze(0)                
                if(self.preproc_final==torch.empty): self.preproc_final = self.preproc_flat
                else:   self.preproc_final=torch.cat([self.preproc_final,self.preproc_flat],dim=0)                    
            return self.preproc_final .clone()
            
        #Griffin-Lim transform
        if(preproc_type=="torch_GrifLim"):
            #parameters
            n_fft=64 #should be order of two
                        
            #MEL decomposition            
            for l in range(0,shp[0]):
                self.preproc_in=torch.empty
                self.preproc_in=signal[l,:].reshape((1, -1))       
                #https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
                GL_transoform = torchaudio.transforms.GriffinLim(n_fft=n_fft).to(torch.double)              
                self.preproc = GL_transoform(self.preproc_in)                
                self.preproc_flat = self.preproc.reshape(-1)    
                self.preproc_flat=self.preproc_flat.unsqueeze(0)                
                if(self.preproc_final==torch.empty): self.preproc_final = self.preproc_flat
                else:   self.preproc_final=torch.cat([self.preproc_final,self.preproc_flat],dim=0)                    
            return self.preproc_final.clone()

        #Griffin-Lim transform
        if(preproc_type=="torch_LFCC"):
            #parameters            
            sample_rate = 1000000
            n_lfcc=13
                        
            #MEL decomposition            
            for l in range(0,shp[0]):
                self.preproc_in=torch.empty
                self.preproc_in=signal[l,:].reshape((1, -1))       
                #https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
                LFCC_transoform = torchaudio.transforms.LFCC(sample_rate=sample_rate,
                                                             n_lfcc=n_lfcc,
                                                             speckwargs={"n_fft": 40, "hop_length": 30, "center": False},
                                                            ).to(torch.double)              
                self.preproc = LFCC_transoform(self.preproc_in)                
                self.preproc_flat = self.preproc.reshape(-1)    
                self.preproc_flat=self.preproc_flat.unsqueeze(0)                
                if(self.preproc_final==torch.empty): self.preproc_final = self.preproc_flat
                else:   self.preproc_final=torch.cat([self.preproc_final,self.preproc_flat],dim=0)                    
            return self.preproc_final.clone()

        #Spec centroid
        if(preproc_type=="torch_SpecCentr"):
            #parameters            
            sample_rate = 1000000
            n_fft = 50
                        
            #MEL decomposition            
            for l in range(0,shp[0]):
                self.preproc_in=torch.empty
                self.preproc_in=signal[l,:].reshape((1, -1))       
                #https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
                SC_transoform = torchaudio.transforms.SpectralCentroid(sample_rate=sample_rate,
                                                                       n_fft=n_fft,
                                                                      ).to(torch.double)              
                self.preproc = SC_transoform(self.preproc_in)                
                self.preproc_flat = self.preproc.reshape(-1)    
                self.preproc_flat=self.preproc_flat.unsqueeze(0)                
                if(self.preproc_final==torch.empty): self.preproc_final = self.preproc_flat
                else:   self.preproc_final=torch.cat([self.preproc_final,self.preproc_flat],dim=0)                    
            return self.preproc_final.clone()

    #helper functions
    def Helper_FlatListOfLabeledFeat(self,feat,labels):
        feat_new=[]
        labs=[]
        h_l=len(feat)
        for i in range(0,h_l):      
            if(feat[i] is not None): 
                shp = np.shape(feat[i])       
                if(len(shp)>=1):
                    for j in range(0,shp[0]):
                        feat_new.append(feat[i][j][:])
                        labs.append(labels[i][j])
        return feat_new,labs
            
#dp=DataPreproc()
#fd= dp.SplitEntireSignalIntoSnippets(signal=PLATES_ARRAY[0].sigments_sign[0],channs_indx=0,torch_tensor=False,snip_size=1000)
#fd,labs= dp.SplitLabPlateSegmentIntoSnips(PLATES_ARRAY[0],snip_size=5,segm_index=0,channs_indx=[0])

#extract channels from the string
def ExtractChannelsFromString(chans_string,separator=","):
            #chans_string=self.proc_settings.get("chans_list_user")
            if(chans_string==""):
                print("Check the channels list and repeat. AT present the list is empty.")
                return None
            channels_to_use=[]
            x_str = chans_string.split(separator) #",")
            try:                
                for l in range(0,len(x_str)):
                    chan=int(x_str[l])
                    channels_to_use.append(chan)
            except:
                print("The user channels input is incorrect.Check values and repeat...")
                return None
            return channels_to_use

#******************************************************************************************************
#******************************************************************************************************

#read settings before execuring processing
def ReadSettings(window):
    settings={}
    
    #snippet     
    snip_size=int(window.ui.classification_snippet_size_text.text())
    settings["snippet_size"] = snip_size    
    #preprocessing
    classification_preproc_dropdown=window.ui.classification_preproc_dropdown.currentText()
    settings["classification_preproc_dropdown"] = classification_preproc_dropdown    
    #channels
    chans_to_use=window.ui.classification_channels_choice_drop_down.currentIndex()
    settings["classification_channels_choice_drop_down"] = chans_to_use

    classification_user_channels_text_box=window.ui.classification_user_channels_text_box.text()
    settings["classification_user_channels_text_box"] = classification_user_channels_text_box

    chan_from_settings=window.ui.Channel_segment_plot.currentIndex()
    settings["chan_from_settings"] = chan_from_settings

    #algorithm
    algorithm=window.ui.classificationclassifier_dropdown.currentText()
    settings["algorithm"] = algorithm
    #how to process the data
    proc_plates_segments=window.ui.classification_preproc_dropdown_2.currentText()
    settings["plate_segm_process"] = proc_plates_segments
    #real time source
    real_time_source=window.ui.real_time_source_dropdown_2.currentText()
    settings["real_time_source"] = real_time_source
    real_time_folder=window.ui.real_time_folder_text.text()
    settings["real_time_folder_text"] = real_time_folder
    
    #real time spectrum
    trigger_level=window.ui.REAL_T_TRigger_Level_textbox.text()
    settings["trigger_level"] = trigger_level
    sampling_rate=window.ui.REAL_T_smp_rate_textbox_2.text()
    settings["sampling_rate"] = sampling_rate
    pre_trigger_duration=window.ui.REAL_T_PRE_TRigger_durat_textbox_3.text()
    settings["pre_trigger_duration"] = pre_trigger_duration
    post_trigger_duration=window.ui.REAL_Post_trig_durat_textbox_4.text()
    settings["post_trigger_duration"] = post_trigger_duration
    ampl_per_channel=window.ui.REAL_T_amp_chan_textbox_3.text()
    settings["ampl_per_channel"] = ampl_per_channel
    trig_chan_num=window.ui.REAL_T_trigger_channel_drop_box.currentText()
    settings["trig_chan_num"] = trig_chan_num
    show_info=window.ui.RealT_show_info_checkbox.isChecked()
    settings["show_info"] = show_info    
    RealT_show_processed_signals_checkbox_3=window.ui.RealT_show_processed_signals_checkbox_3.isChecked()
    settings["RealT_show_processed_signals_checkbox_3"] = RealT_show_processed_signals_checkbox_3
    only_single_shot=window.ui.RealT_show_only_single_shot_checkbox_4.isChecked()
    settings["only_single_shot"] = only_single_shot

       
    #SPECTROGRAMS SHOW
    spectrogrym_type = window.ui.classification_preproc_dropdown_4.currentText()
    settings["spectrogrym_type"] = spectrogrym_type
    #MEL SPECTROGRAMS
    spectrogrym_MEL_nfft = int(window.ui.Settings_MEL_num_ffts.text())
    settings["spectrogrym_MEL_nfft"] = spectrogrym_MEL_nfft
    
    #CLASIFIERS AND ANOMALY DETECTORS
    autoencoder_torch_thershold=window.ui.Autoencoder_torch_threshold_factor.text()
    settings["autoencoder_torch_thershold"] = autoencoder_torch_thershold
    autoencoder_torch_epochs_num=window.ui.Autoencoder_torch_epochs_num.text()
    settings["autoencoder_torch_epochs_num"] = autoencoder_torch_epochs_num
    autoencoder_torch_learn_rate=window.ui.Autoencoder_torch_learn_rate.text()
    settings["autoencoder_torch_learn_rate"] = autoencoder_torch_learn_rate
    autoencoder_bin_number=window.ui.Autoencoderbins_text_input.text()
    settings["autoencoder_bin_num"] = autoencoder_bin_number
    #resnet specific settings
    Autoencoder_ResNet_weight_decay_input_text=window.ui.Autoencoder_ResNet_weight_decay_input_text.text()
    settings["Autoencoder_ResNet_weight_decay_input_text"] = Autoencoder_ResNet_weight_decay_input_text

    #first page - MAIN
    MAIN_signals_folder=window.ui.load_data_path.text()
    settings["MAIN_signals_folder"]=MAIN_signals_folder
    MAIN_plate_layout_dropbox_selected_item=window.ui.load_data_plate_type_dropdown.currentIndex()
    settings["MAIN_plate_layout_dropbox_selected_item"]=MAIN_plate_layout_dropbox_selected_item
    MAIN_real_time_source_dropdown_2=window.ui.real_time_source_dropdown_2.currentIndex()
    settings["MAIN_real_time_source_dropdown_2"] = MAIN_real_time_source_dropdown_2
    MAIN_classification_snippet_size_text=str(window.ui.classification_snippet_size_text.text())
    settings["MAIN_classification_snippet_size_text"]=MAIN_classification_snippet_size_text
    MAIN_classification_preproc_dropdown=window.ui.classification_preproc_dropdown.currentIndex()
    settings["MAIN_classification_preproc_dropdown"]=MAIN_classification_preproc_dropdown

    #settings - Visualization page
    SETTINGS_VIZUALIZATION_load_default_GUI=window.ui.GUI_load_default_on_start.isChecked()
    settings["SETTINGS_VIZUALIZATION_load_default_GUI"]=SETTINGS_VIZUALIZATION_load_default_GUI

    Color_list_drop_down_=window.ui.Color_list_drop_down_.currentIndex()
    settings["Color_list_drop_down_"]=Color_list_drop_down_

    Settings_Segmentation_colors_number_textbox=window.ui.Settings_Segmentation_colors_number_textbox.text()
    settings["Settings_Segmentation_colors_number_textbox"]=Settings_Segmentation_colors_number_textbox

    Show_results_color_scheme_drop_down_1=window.ui.Show_results_color_scheme_drop_down_1.currentIndex()
    settings["Show_results_color_scheme_drop_down_1"]=Show_results_color_scheme_drop_down_1

    Show_results_color_scheme_drop_down_text=window.ui.Show_results_color_scheme_drop_down_1.currentText()
    settings["Show_results_color_scheme_drop_down_text"]=Show_results_color_scheme_drop_down_text

    GUI_show_results_points_number_limit_textbox=window.ui.GUI_show_results_points_number_limit_textbox.text()
    settings["GUI_show_results_points_number_limit_textbox"]=GUI_show_results_points_number_limit_textbox

    GUI_show_results_points_number_limit_checkbox= bool(window.ui.GUI_show_results_points_number_limit_checkbox.isChecked())#if to check points limit limit
    settings["GUI_show_results_points_number_limit_checkbox"] = GUI_show_results_points_number_limit_checkbox

    return settings

#*********************************************************************************************************************
#*********************************************************************************************************************
#***************************************GUI***************************************************************************

#check on start if to load defaults:
def CheckOnStartToLoadGUI(window,path):
    my_set={}
    my_set = json.load(open( path ))
    to_load = bool(my_set["SETTINGS_VIZUALIZATION_load_default_GUI"])
    if(to_load==True):
        LoadInterfaceFromFile(window,path)

#load interface 
def LoadInterfaceFromFile(window,path):
    #my_set = set(open(path).read().split())
    my_set={}
    my_set = json.load( open( path ) )

    #globals

    #first page - MAIN   
    window.ui.load_data_path.setText(str(my_set["MAIN_signals_folder"]))
    window.ui.load_data_plate_type_dropdown.setCurrentIndex(int(my_set["MAIN_plate_layout_dropbox_selected_item"]))
    window.ui.real_time_source_dropdown_2.setCurrentIndex(int(my_set["MAIN_real_time_source_dropdown_2"]))
    window.ui.classification_snippet_size_text.setText(str(my_set["MAIN_classification_snippet_size_text"]))
    window.ui.classification_preproc_dropdown.setCurrentIndex(int(my_set["MAIN_classification_preproc_dropdown"]))
    window.ui.classification_user_channels_text_box.setText(str(my_set["classification_user_channels_text_box"]))
    window.ui.classification_channels_choice_drop_down.setCurrentIndex(int(my_set["classification_channels_choice_drop_down"]))
    
    #settings - Visualization page
    window.ui.GUI_load_default_on_start.setChecked(bool(my_set["SETTINGS_VIZUALIZATION_load_default_GUI"]))    
    window.ui.Color_list_drop_down_.setCurrentIndex(int(my_set["Color_list_drop_down_"])) 
    window.ui.Settings_Segmentation_colors_number_textbox.setText(str(my_set["Settings_Segmentation_colors_number_textbox"])) 
    window.ui.Show_results_color_scheme_drop_down_1.setCurrentIndex(int(my_set["Show_results_color_scheme_drop_down_1"])) 
    window.ui.GUI_show_results_points_number_limit_textbox.setText(str(my_set["GUI_show_results_points_number_limit_textbox"])) 
    window.ui.GUI_show_results_points_number_limit_checkbox.setChecked(bool(my_set["GUI_show_results_points_number_limit_checkbox"]))
    window.ui.RealT_show_processed_signals_checkbox_3.setChecked(bool(my_set["RealT_show_processed_signals_checkbox_3"]))
    

#save interface
def SaveInterfaceIntoFile(window,path):
    gui=ReadSettings(window)
    #https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
    json.dump( gui, open( path, 'w' ) )

    """
    gui={}
    #settings - Visualization page
    SETTINGS_VIZUALIZATION_load_default_GUI=window.ui.GUI_load_default_on_start.isChecked()
    gui["SETTINGS_VIZUALIZATION_load_default_GUI"]=SETTINGS_VIZUALIZATION_load_default_GUI

    Color_list_drop_down_=window.ui.Color_list_drop_down_.currentIndex()
    gui["Color_list_drop_down_"]=Color_list_drop_down_

    Settings_Segmentation_colors_number_textbox=window.ui.Settings_Segmentation_colors_number_textbox.text()
    gui["Settings_Segmentation_colors_number_textbox"]=Settings_Segmentation_colors_number_textbox

    Show_results_color_scheme_drop_down_1=window.ui.Show_results_color_scheme_drop_down_1.currentIndex()
    gui["Show_results_color_scheme_drop_down_1"]=Show_results_color_scheme_drop_down_1

    GUI_show_results_points_number_limit_textbox=window.ui.GUI_show_results_points_number_limit_textbox.text()
    gui["GUI_show_results_points_number_limit_textbox"]=GUI_show_results_points_number_limit_textbox

    GUI_show_results_points_number_limit_checkbox= bool(window.ui.GUI_show_results_points_number_limit_checkbox.isChecked())#if to check points limit limit
    gui["GUI_show_results_points_number_limit_checkbox"] = GUI_show_results_points_number_limit_checkbox
    
    #first page - MAIN
    MAIN_signals_folder=window.ui.load_data_path.text()
    gui["MAIN_signals_folder"]=MAIN_signals_folder
    MAIN_plate_layout_dropbox_selected_item=window.ui.load_data_plate_type_dropdown.currentIndex()
    gui["MAIN_plate_layout_dropbox_selected_item"]=MAIN_plate_layout_dropbox_selected_item
    MAIN_real_time_source_dropdown_2=window.ui.real_time_source_dropdown_2.currentIndex()
    gui["MAIN_real_time_source_dropdown_2"] = MAIN_real_time_source_dropdown_2
    MAIN_classification_snippet_size_text=str(window.ui.classification_snippet_size_text.text())
    gui["MAIN_classification_snippet_size_text"]=MAIN_classification_snippet_size_text
    MAIN_classification_preproc_dropdown=window.ui.classification_preproc_dropdown.currentIndex()
    gui["MAIN_classification_preproc_dropdown"]=MAIN_classification_preproc_dropdown
            
    #https://blog.finxter.com/5-best-ways-to-write-a-set-to-a-file-in-python/
    
    
    with open(path, 'w') as file:
            for gui_elements in gui:
                file.write(f"{gui_elements}\n")
    """

#setings for display and graphics
def ReadGraphSettings(window):

    settings={}
    #labeling data - show
    classif_show_type= window.ui.classification_plot_choice_dropdown_3.currentText()
    settings["classification_show_labeled_data_type"] = classif_show_type
    #color list how to create it
    classif_color_list_type= window.ui.Color_list_drop_down_.currentText()
    settings["classif_color_list_type"] = classif_color_list_type
    #colors number
    classif_color_list_number= window.ui.Settings_Segmentation_colors_number_textbox.text()
    settings["classif_color_list_number"] = classif_color_list_number
    #how to show the results
    GUI_show_results_points_number_limit_checkbox= bool(window.ui.GUI_show_results_points_number_limit_checkbox.isChecked())#if to check points limit limit
    settings["GUI_show_results_points_number_limit_checkbox"] = GUI_show_results_points_number_limit_checkbox

    GUI_show_results_points_number_limit_textbox= window.ui.GUI_show_results_points_number_limit_textbox.text()#points limit
    settings["GUI_show_results_points_number_limit_textbox"] = GUI_show_results_points_number_limit_textbox
            
    return settings

#color list
def ColorsListGen(type_list,col_num):
    #https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    colors_list=[]
    if(type_list=="Grades of red"):
        colors_list=['#228B22','#FF0000','#8B0000']
    if(type_list=="Random colors"):
        colors_list=dc.get_colors(col_num)
    return colors_list
    
#get bipolar plate segments list
def getSegmentNames(plate_type):
    tmp_tuple=None
    if(plate_type=="bpp"):        tmp_tuple = bpp_layout  
    if(plate_type=="long_bpp_1"): tmp_tuple = long_bpp1_layout  
    if(plate_type=="long_bpp_2"): tmp_tuple = long_bpp2_layout  
    if(tmp_tuple is not None):
        list_names=[]
        for i in range(0,len(tmp_tuple)):
            list_names.append(str(tmp_tuple[i][0]+"_"+tmp_tuple[i][1]))       
        return list_names
    else:
        return []

#****************************************************************************************
#*******************************threading************************************************
#****************************************************************************************
class thread_with_trace(threading.Thread):
  #https://www.geeksforgeeks.org/python/python-different-ways-to-kill-a-thread/
  def __init__(self, *args, **keywords):
    threading.Thread.__init__(self, *args, **keywords)
    self.killed = False

  def start(self):
    self.__run_backup = self.run
    self.run = self.__run      
    threading.Thread.start(self)

  def __run(self):
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

  def globaltrace(self, frame, event, arg):
    if event == 'call':
      return self.localtrace
    else:
      return None

  def localtrace(self, frame, event, arg):
    if self.killed:
      if event == 'line':
        raise SystemExit()
    return self.localtrace

  def kill(self):
    self.killed = True

def func():
  while True:
    print('thread running')
    

#****************************************************************************************
#*******************************bipolar plates layout************************************
#****************************************************************************************

bpp_layout = (
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

long_bpp1_layout = (
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

long_bpp2_layout = (
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

#*************************************************************************************
#****************************MODELS***************************************************
#*************************************************************************************
class Autoencoder(nn.Module):
    #https://www.geeksforgeeks.org/deep-learning/how-to-use-pytorch-for-anomaly-detection/
    def __init__(self,window_size=100):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 100),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 2000),
            nn.ReLU(),
            nn.Linear(2000, 5000),
            nn.ReLU(),
            nn.Linear(5000, window_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def TrainTorchAutoencoder(feat=None,labels=None,num_epochs=100,lr=1e-7):
    if(feat is None) or (len(feat)==0): 
        print("")
        print("features are empty - training aborted...")
        return None

    unique_labs=list(set(labels))
    if(len(unique_labs)==0):
        print("No labeled data is defined. One label is needed for training...")
        pass

    torch.use_deterministic_algorithms(False)
    #https://www.eyer.ai/blog/anomaly-detection-in-time-series-data-python-a-starter-guide/            
    x_sf=[]
    ref_l=labels[0]
    for m in range(0,len(labels)):
        if(ref_l==labels[m]):
            cur_feat=(np.asarray(feat)[m,:])
            x_sf.append(cur_feat)
    shp_training_set=np.shape(np.asarray(x_sf))
    if(len(shp_training_set)==1):d_size=shp_training_set[0]
    else: d_size=shp_training_set[1]
    print("")
    print("Torch autoencoder dataset inlcude lab. - "+str(ref_l))
    print("Trainset shape - "+str(shp_training_set))
    print("Torch autoencoder training...")
    #initialize
    model = Autoencoder(window_size=d_size)
    if torch.cuda.is_available(): model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sequences = torch.tensor(np.asarray(x_sf), dtype=torch.float32)
    if torch.cuda.is_available(): sequences = sequences.cuda()
    
    start_t = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(sequences)
        loss = criterion(output, sequences)
        loss.backward()
        optimizer.step()    
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    end_t = time.time()
    
    print("Training is done in "+str(end_t-start_t)+ str(" s"))
    
    start_t = time.time()
    y_pred = model(sequences)
    end_t = time.time()
    print("Pridiction time: "+str(end_t-start_t)+" s for " +str(shp_training_set[0])+" features")
    return model  

def PredictTorchAutoencoder(model=None, feat=None,thresh_fact=2,show_hist=False,bins_num=50,fig_ax=None):
    shp=np.shape(np.asarray(feat))
    sequences = torch.tensor(np.asarray(feat), dtype=torch.float32)
    if torch.cuda.is_available(): sequences = sequences.cuda()    
    with torch.no_grad():
        predictions = model(sequences)
        losses = torch.mean((predictions - sequences)**2, dim=1)
        if(show_hist==True):
            fig_ax.hist(losses.cpu().numpy(), bins=bins_num)
            fig_ax.set_xlabel("Loss")
            fig_ax.set_ylabel("Frequency")            
            """
            plt.hist(losses.cpu().numpy(), bins=bins_num)
            plt.xlabel("Loss")
            plt.ylabel("Frequency")
            """
            plt.show()
        # Threshold for defining an anomaly        
        threshold =losses.mean() + thresh_fact* losses.std()#losses.mean() + 2 * losses.std()
        #print(f"Anomaly threshold: {threshold.item()}")
        # Detecting anomalies
        anomalies = losses > threshold
        if torch.cuda.is_available(): anomaly_positions = np.where(anomalies.cpu().numpy())[0]
        else:                         anomaly_positions = np.where(anomalies.numpy())[0]
        labels=np.zeros(shp[0])
        for i in range(0,len(anomaly_positions)):
            labels[anomaly_positions[i]]=1
        #print(f"Anomalies found at positions: {np.where(anomalies.cpu().numpy())[0]}")        
        return list(labels)

#****************************************************************************************
#****************************************************************************************
#   AUTOENCODER 2
class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l =  torch.linalg.cholesky(a)#torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

class DAGMM(nn.Module):
    def __init__(self, input_size=1000, n_gmm=2, z_dim=30):
        """Network for DAGMM (KDDCup99)"""
        super(DAGMM, self).__init__()
        #Encoder network
        print("input_size="+str(input_size))
        self.fc0 = nn.Linear(input_size, 100)
        self.fc1 = nn.Linear(100, 80)
        self.fc2 = nn.Linear(80, 70)
        self.fc3 = nn.Linear(70, 50)
        self.fc4 = nn.Linear(50, z_dim)

        #Decoder network
        self.fc5 = nn.Linear(z_dim, 50)
        self.fc6 = nn.Linear(50, 70)
        self.fc7 = nn.Linear(70, 80)
        self.fc8 = nn.Linear(80, 100)
        self.fc9 = nn.Linear(100, input_size)
        self.fc9_1 = nn.Linear(input_size, input_size)

        #Estimation network
        self.fc10 = nn.Linear(z_dim+2, 10)
        self.fc11 = nn.Linear(10, n_gmm)

    def encode(self, x):        
        h = torch.tanh(self.fc0(x))
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, x):
        h = torch.tanh(self.fc5(x))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        h = torch.tanh(self.fc8(h))
        h = torch.tanh(self.fc9(h))
        return self.fc9_1(h)
    
    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc10(z)), 0.5)
        return F.softmax(self.fc11(h), dim=1)
    
    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x):
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma

class TrainerDAGMM:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader = data
        self.device = device
        self.loss_list=[]

    def train(self):
        """Training the DAGMM model"""
        self.model = DAGMM(input_size=self.args.input_size, 
                           n_gmm=self.args.n_gmm, 
                           z_dim=self.args.latent_dim
                          ).to(self.device)#self.args.n_gmm, self.args.latent_dim).to(self.device)
        
        self.model.apply(weights_init_normal)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.device, self.args.n_gmm)
        
        self.loss_list=[]
        
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x,y in Bar(self.train_loader):                    
                x = x.float().to(self.device)
                optimizer.zero_grad()                
                _, x_hat, z, gamma = self.model(x)
                loss = self.compute.forward(x, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()                
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
            self.loss_list.append(total_loss)

def SeparateFeat_Norm_Anom(CLASSIF_FEAT,CLASSIF_LABS):
    unique_labels=GetUniqueElements_List(CLASSIF_LABS)
    feat_norm=[]
    labs_norm=[]
    feat_anomaly=[]
    labs_anomaly=[]
    label_0=CLASSIF_LABS[0]
    for k in range(0,len(CLASSIF_LABS)):
        if(CLASSIF_LABS[k]==label_0):
            feat_norm.append(np.asarray(CLASSIF_FEAT[k][:]))
            labs_norm.append(0)
        else:
            feat_anomaly.append(np.asarray(CLASSIF_FEAT[k][:]))
            labs_anomaly.append(1)
    return feat_norm,labs_norm,feat_anomaly,labs_anomaly

def Train_Autoencoder_2(CLASSIF_FEAT,CLASSIF_LABS,
                  num_of_epochs=200,
                  patience=50,
                  lr=1e-6,
                  lr_milestones=[50],
                  batch_size=10,
                  latent_dim=1,
                  n_gmm=5,
                  lambda_energy=0.01,
                  lambda_cov=0.0005,                  
                 ):
    
    unique_labels=GetUniqueElements_List(CLASSIF_LABS)
    if(len(unique_labels)<=1):
        print("for this algorithms two labels are needed")
    """
    feat_norm=[]
    labs_norm=[]
    feat_anomaly=[]
    labs_anomaly=[]
    label_0=CLASSIF_LABS[0]
    for k in range(0,len(CLASSIF_LABS)):
        if(CLASSIF_LABS[k]==label_0):
            feat_norm.append(np.asarray(CLASSIF_FEAT[k][:]))
            labs_norm.append(0)
        else:
            feat_anomaly.append(np.asarray(CLASSIF_FEAT[k][:]))
            labs_anomaly.append(1)
    """
    feat_norm,labs_norm,feat_anomaly,labs_anomaly= SeparateFeat_Norm_Anom(CLASSIF_FEAT,CLASSIF_LABS)

    class Args:
        def __init__(self):
            self.num_epochs=num_of_epochs#200
            self.patience=patience#50
            self.lr=lr#1e-6
            self.lr_milestones=lr_milestones#[50]
            self.batch_size=batch_size#10
            self.latent_dim=latent_dim#1
            self.n_gmm=n_gmm#5
            self.lambda_energy=lambda_energy#0.01
            self.lambda_cov=lambda_cov#0.0005
            self.input_size=np.shape(np.asarray(feat_norm))[1]

    torch.use_deterministic_algorithms(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args=Args()    
    #datal_train,datal_test = get_KDDCup99(args,feat_norm,labs_norm,feat_anomaly,labs_anomaly)
    norm_x = torch.Tensor(np.asarray(feat_norm)).to(device) 
    norm_y = torch.Tensor(np.asarray(labs_norm)).to(device) 
    norm_dataset = torch.utils.data.TensorDataset(norm_x,norm_y)
    norm_dataloader = torch.utils.data.DataLoader(norm_dataset)
    dagmm = TrainerDAGMM(args, norm_dataloader, device)
    dagmm.train()
    loss_training=dagmm.loss_list

    train_phi = 0    #gamma_sum / N_samples
    train_mu = 0     #mu_sum / gamma_sum.unsqueeze(-1)
    train_cov = 0    #cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
    
    with torch.no_grad():
        
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        
        n_gmm=args.n_gmm
        model= dagmm.model
        compute = ComputeLoss(model, None, None, device, n_gmm)
        
        for x, _ in norm_dataloader:
            _, _, z, gamma = model(x)
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)
            
            batch_gamma_sum = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            N_samples += x.size(0)
        
        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        dagmm=None
        
        return model,train_phi,train_mu,train_cov,args,loss_training

def model_run(model,args,anomaly_dataloader,train_phi,train_mu,train_cov):
    energy_test = []
    n_gmm=args.n_gmm
    #model= dagmm.model
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute = ComputeLoss(model, None, None, device, n_gmm)
    for x, _ in anomaly_dataloader:
            x = x.float().to(device)
    
            _, _, z, gamma = model(x)
            sample_energy, cov_diag  = compute.compute_energy(z, gamma, train_phi,
                                                              train_mu, train_cov,
                                                              sample_mean=False)
            if(str(device) == 'cuda'):            
                energy_test.append(sample_energy.detach().cpu())
            else:
                 energy_test.append(sample_energy)
    shp=len(energy_test)
    energy=[]
    for k in range(0,shp):
        #for l in range(0,energy_test[l]):
        energy.append(energy_test[k].numpy())
    energy=np.asarray(energy)
    print(cov_diag.shape)
    return energy

#****************************************************************************************
#****************************************************************************************
#CNN AUTO encoder for 498 sampling points as input

#CNN layers output
def CNNOutput(W,K,S,P):
    #W is the input volume - in your case 128
    #K is the Kernel size - in your case 5
    #S is the stride - which you have not provided.    
    #P is the padding - in your case 0 i believe    
    return (W-K+2*P)/S+1

class CNNAutoencoder_Shallow_498sp_1(nn.Module):
    def __init__(self):
        super(CNNAutoencoder_Shallow_498sp_1, self).__init__()
        # Encoder (Convolutional Layers) : Compressed into low-dimensional vectors.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=6, stride=2, padding=1),  # [batch, 32, 14, 14]
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=6, stride=2, padding=1),  # [batch, 64, 7, 7]
            nn.ReLU(True),
            nn.Conv1d(128, 256, kernel_size=3,stride=2, padding=1),  # [batch, 128, 1, 1]                        
        )
        # Decoder (Transpose Convolutional Layers) : Reconstruct low-dimensional vectors to original image dimensions.
        self.decoder = nn.Sequential(                       
            nn.ConvTranspose1d(256,128, kernel_size=3,stride=2, padding=1,output_padding=0),  # [batch, 64, 7, 7]
            nn.ReLU(True),
            nn.ConvTranspose1d(128,64, kernel_size=6, stride=2, padding=1,output_padding=0),  # [batch, 64, 7, 7]
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, kernel_size=6, stride=2, padding=1, output_padding=0),  # [batch, 32, 14, 14]
            nn.ReLU(True),            
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#   AE- CNN-RESNET-shallow/deep
#https://github.com/Navy10021/MDAutoEncoder/blob/main/notebooks/MDAutoEncoder.ipynb
# CNN based Auto-Encoder model

class CNNAutoencoder_Shallow(nn.Module):
    def __init__(self,input_size=32):
        super(CNNAutoencoder_Shallow, self).__init__()
        # Encoder (Convolutional Layers) : Compressed into low-dimensional vectors.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, input_size, kernel_size=3, stride=2, padding=1),  # [batch, 32, 14, 14]
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 7, 7]
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=7)  # [batch, 128, 1, 1]
        )
        # Decoder (Transpose Convolutional Layers) : Reconstruct low-dimensional vectors to original image dimensions.
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=7),  # [batch, 64, 7, 7]
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 32, 14, 14]
            nn.ReLU(True),
            nn.ConvTranspose1d(input_size, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 1, 28, 28]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def TrainAE_1(feat,
              labels,
              AE_type="CNN_shallow",
              lr=0.001,
              num_epochs=10,
              visualize=True,
             ):
    
    norm,norm_l,non_norm,non_norm_l = SeparateFeat_Norm_Anom(feat,labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #train
    norm = torch.Tensor(np.asarray(norm)).to(device) 
    norm_l = torch.Tensor(np.asarray(norm_l)).to(device) 
    norm_dataset = torch.utils.data.TensorDataset(norm,norm_l)
    train_loader = torch.utils.data.DataLoader(norm_dataset)
    
    if(len(norm.shape)<=2): 
        norm=norm[:,None,:]
        print("Features shape is corrected. Cur. shape - "+str(norm.shape))
    
    #test
    non_norm = torch.Tensor(np.asarray(non_norm)).to(device) 
    non_norm_l = torch.Tensor(np.asarray(non_norm_l)).to(device) 
    non_norm_dataset = torch.utils.data.TensorDataset(non_norm,non_norm_l)
    test_loader = torch.utils.data.DataLoader(non_norm_dataset)

    if(len(norm.shape)<=2): 
        non_norm=non_norm[:,None,:]
        print("Features shape is corrected. Cur. shape - "+str(non_norm.shape))
            
    #https://github.com/Navy10021/MDAutoEncoder/blob/main/notebooks/MDAutoEncoder.ipynb    
    # model 1 : CNN based model
    model=None
    if(AE_type=="CNN_shallow"):
        model = CNNAutoencoder_Shallow_498sp_1().to(device)#CNNAutoencoder_Shallow().to(device) #CNNAutoencoder_Shallow_500sp().to(device) #CNNAutoencoder_Shallow().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = num_epochs

    # Training
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
    
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for data, _ in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
    
        train_losses.append(epoch_loss / len(train_loader))

    if(visualize==True):
        # Plotting the training loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return model

#****************************************************************************************
#****************************************************************************************
#   RESNET autoencoder
def train_val_split(dataset, val_ratio=0.1):
    """
    Splits `dataset` into a training set and a validation set, by the given ratio `val_ratio`.
    """
    
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set

def to_img(x):
    """
    Denormalises Tensor `x` (normalised from -1 to 1) to image format (from 0 to 1).
    """
    
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv1d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv1d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose1d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose1d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm1d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)

class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv1d(1, 16, 3, 1, 1) # 16 32 32
        self.BN = nn.BatchNorm1d(16)
        self.rb1 = ResBlock(16, 16, 3, 2, 1, 'encode') # 16 16 16
        self.rb2 = ResBlock(16, 32, 3, 1, 1, 'encode') # 32 16 16
        self.rb3 = ResBlock(32, 32, 3, 2, 1, 'encode') # 32 8 8
        self.rb4 = ResBlock(32, 48, 3, 1, 1, 'encode') # 48 8 8
        self.rb5 = ResBlock(48, 48, 3, 2, 1, 'encode') # 48 4 4
        self.rb6 = ResBlock(48, 64, 3, 2, 1, 'encode') # 64 2 2
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1 = self.rb1(init_conv)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        return rb6

class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.rb1 = ResBlock(64, 48, 2, 2, 0, 'decode') # 48 4 4
        self.rb2 = ResBlock(48, 48, 2, 2, 0, 'decode') # 48 8 8
        self.rb3 = ResBlock(48, 32, 3, 1, 1, 'decode') # 32 8 8
        self.rb4 = ResBlock(32, 32, 2, 2, 0, 'decode') # 32 16 16
        self.rb5 = ResBlock(32, 16, 3, 1, 1, 'decode') # 16 16 16
        self.rb6 = ResBlock(16, 16, 2, 2, 0, 'decode') # 16 32 32
        self.out_conv = nn.ConvTranspose1d(16, 1, 3, 1, 1) # 3 32 32
        self.tanh = nn.Tanh()
        
    def forward(self, inputs):
        rb1 = self.rb1(inputs)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out_conv = self.out_conv(rb6)
        output = self.tanh(out_conv)
        return output

class Autoencoder_S1(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self):
        super(Autoencoder_S1, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p
    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def Train_ResNetAE(feat,labels,
                   init_lr=0.00005,
                   batch_size=30,
                   weight_decay=0.01,
                   num_epochs = 100,
                   visualize=True 
                  ):

    norm_f,norm_l,non_norm_f,non_norm_l = SeparateFeat_Norm_Anom(feat,labels)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    norm=torch.tensor(norm_f,dtype=torch.float32).to(device)
    non_norm=torch.tensor(non_norm_f,dtype=torch.float32).to(device)
    
    if(len(norm.shape) <= 2):
        norm=norm[:,None,:]
        print("Train features are adjsuted to shape: "+str(norm.shape))
    
    if(len(non_norm.shape) <= 2) and (non_norm.shape[0]!=0):
        non_norm=non_norm[:,None,:]
        print("Test features are adjsuted to shape: "+str(non_norm.shape))
        
    norm_dataset = torch.utils.data.TensorDataset(norm,torch.tensor(norm_l,dtype=torch.float32).to(device))
    train_loader = torch.utils.data.DataLoader(norm_dataset,batch_size=batch_size)
    non_norm_dataset = torch.utils.data.TensorDataset(non_norm,torch.tensor(non_norm_l,dtype=torch.float32).to(device))
    test_loader = torch.utils.data.DataLoader(non_norm_dataset,batch_size=batch_size)
    
    # Instantiate a network model
    ae = Autoencoder_S1().to(device)
    # Define optimizer
    optimizer = optim.SGD(ae.parameters(), lr=init_lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 60, 0.1)
    
    loss_hist=[]
    
    for epoch in range(num_epochs):
      epoch_loss = 0
      # Train the model
      #with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:        
      for data, _ in Bar(train_loader): #for i, batch in enumerate(train_loader):
                images = data.to(device)            
                # Zero all gradients
                optimizer.zero_grad()            
                # Calculating the loss
                preds = ae(images)
                loss = F.mse_loss(preds, images)            
                """
                if data % 10 == 0:
                    with torch.no_grad():
                        val_images, _ = next(iter(train_loader))
                        val_preds = ae(val_images)
                        val_loss = F.mse_loss(val_preds, val_images)
                        m.track_loss(val_loss, val_images.size(0), mode='val')
                    #print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch+1,i*hparams.batch_size,round(loss.item(), 6),round(val_loss.item(), 6)))
                """
                epoch_loss += loss.item()
                # Backpropagate
                loss.backward()
                # Update the weights
                optimizer.step()            
                #m.track_loss(loss, images.size(0), mode='train')        
      loss_hist.append(epoch_loss / len(train_loader))
      print('Training ResNet AutoEncoder... Epoch: {}, Loss: {:.3f}'.format(epoch, epoch_loss))  
    
    if(visualize==True):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), loss_hist, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.show()       
    return ae,loss_hist
    
#****************************************************************************************
#****************************************************************************************
#   CLASSIFIER OBJECT

class S_Classif:
    def __init__(self):
        self.classifier=None
        self.orig_labels=[]
        self.intern_labels=[]
        #for torch autoencoder_1
        self.tocrh_autoencoder_threshold_factor=3.5
        #train sets
        self.train_feat = None
        self.train_labs = None
        #autoencoder 2 (DAGMM)
        self.autoenc2_train_phi=0
        self.autoenc2_train_mu=0
        self.autoenc2_train_cov=0
        self.autoenc2_args=0
        
    def AssignClassif(self,classif,orig_labels,intern_labels,categ_names=[]):
        self.classifier=classif
        self.orig_labels=orig_labels              
        self.intern_labels=intern_labels    
        self.categ_names=[]
        print("Classifier of type assigned: "+str(type(self.classifier)))
        
    def np_predict(self,feat): #input is numpy array               
        cls_type=type(self.classifier)        
        shp=np.shape(feat)
        labs=np.empty
        if(str(cls_type) == "<class 'xgboost.sklearn.XGBClassifier'>"):
            labs=self.classifier.predict(feat)      
        if(str(cls_type) == "<class 'sklearn.ensemble._iforest.IsolationForest'>"):            
            labs_tmp=self.classifier.predict(feat)
            labs=[]
            for i in range(0,len(labs_tmp)):
                if(labs_tmp[i]==1):
                    labs.append(0)
                else:
                    labs.append(1)
            labs=np.asarray(labs)
        if(str(cls_type) == "<class 'sklearn.cluster._dbscan.DBSCAN'>"):            
            labs_tmp=self.classifier.fit_predict(feat)
            labs=[]
            for i in range(0,len(labs_tmp)):
                if(labs_tmp[i]==1):
                    labs.append(0)
                else:
                    labs.append(1)
            labs=np.asarray(labs)
        if(str(cls_type) == "<class 'sklearn.svm._classes.OneClassSVM'>"):      
            labs_tmp=self.classifier.predict(feat)
            labs=[]
            for i in range(0,len(labs_tmp)):
                if(labs_tmp[i]==1):
                    labs.append(0)
                else:
                    labs.append(1)
            labs=np.asarray(labs)
        if(str(cls_type) == "<class 'SHelpers.Autoencoder'>"):           
            labs=PredictTorchAutoencoder(model=self.classifier,feat=feat,thresh_fact=self.tocrh_autoencoder_threshold_factor)
            labs=np.asarray(labs)

        if(str(cls_type) == "<class 'SHelpers.DAGMM'>"): 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = torch.Tensor(feat).to(device) 
            y = torch.zeros(x.shape[0]).to(device) 
            dataset = torch.utils.data.TensorDataset(x,y)
            dataloader = torch.utils.data.DataLoader(dataset)
            energies = model_run(self.classifier,
                                 self.autoenc2_args,
                                 dataloader,
                                 self.autoenc2_train_phi,
                                 self.autoenc2_train_mu,
                                 self.autoenc2_train_cov)#model,args,anomaly_dataloader,train_phi,train_mu,train_cov
            labs=[]
            for m in range(0,len(energies)):
                ind=0
                while(True):
                    if(ind*self.tocrh_autoencoder_threshold_factor > energies[m]):
                        break
                    else: ind+=1
                labs.append(int(ind))

        if( str(cls_type) == 'SHelpers.CNNAutoencoder_Shallow_498sp_1') or str(cls_type) ==('SHelpers.Autoencoder_S1'): #class 'SHelpers.CNNAutoencoder_Shallow_498sp_1'

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #x = torch.Tensor(feat).to(device) 
            #y = torch.zeros(x.shape[0]).to(device) 
            
            threshold_factor=self.tocrh_autoencoder_threshold_factor
            proximity=3 
            
            non_norm_torch=torch.tensor(feat,dtype=torch.float32).to(device)
            if(len(non_norm_torch.shape)<=2): non_norm_torch=non_norm_torch[:,None,:]
            preds=self.classifier.forward(non_norm_torch)
            losses = torch.mean((preds - non_norm_torch)**2, dim=1)
            labs=[]            
            threshold =losses.mean() + threshold_factor * losses.std()
            anomalies = losses > threshold
            labs=[]
            lk=int(anomalies.shape[1]/proximity)
            for l in range(0,anomalies.shape[0]):
                count=torch.sum(anomalies[l]==True)
                if(count>lk):labs.append(1)
                else: labs.append(0)
                        
        return labs
        
    def getClassifType(self):
        cls_type=type(self.classifier)
        if(str(cls_type) == "<class 'xgboost.sklearn.XGBClassifier'>"):
            return "XGBoost"
        if(str(cls_type) == "<class 'sklearn.ensemble._iforest.IsolationForest'>"):
            return "IsolationTrees"
        if(str(cls_type) == "<class 'sklearn.cluster._dbscan.DBSCAN'>"):
            return "DBSCAN"
        if(str(cls_type) == "<class 'sklearn.svm._classes.OneClassSVM'>"):
            return "SVM"
        if(str(cls_type) == "<class 'SHelpers.Autoencoder'>"):
            return "TorchAutoEnc_1"
#example
#cl=S_Classif()
#cl.AssignClassif(CLASSIFIER,None)