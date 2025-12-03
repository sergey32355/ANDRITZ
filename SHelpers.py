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
import logging
import threading
import sys
import trace

#classifiers
from xgboost import XGBClassifier
import torch
import torchaudio


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
            cur_l=len(self.sigments_labels[l])
            for pp in range(0,cur_l):
                labels_list.append(self.sigments_labels[l][pp][2])
        unique_l=list(set(labels_list))
        return unique_l

def OpenDataFromFolder(PATH="",
                       SEGMENTATION_REF_CHAN_NAME="Trigger",
                       SEGMENTATION_THRESHOLD="automatic",
                       SEGMENTATION_SEGMENTS_NAMES_LIST=[],
                       ONLY_SINGLE_FILE=False,
                       SINGLE_FILE_PATH_BIN="",
                       SINGLE_FILE_PATH_TXT="",
                      ):

    #print(SEGMENTATION_SEGMENTS_NAMES_LIST)            
    arr_txt=[]
    arr_bin=[]
    if(ONLY_SINGLE_FILE==False):        
        arr = next(os.walk(PATH))[2]
        for k in arr:
            filename, file_extension = os.path.splitext(k)
            if(file_extension==".txt"):
                arr_txt.append(k)
                for l in arr:
                    filename_1, file_extension_1 = os.path.splitext(l)
                    if((l!=k) and (file_extension_1==".bin") and (filename_1 in filename)):
                        arr_bin.append(l)
                        break
    else:        
        arr_txt.append(SINGLE_FILE_PATH_TXT)
        arr_bin.append(SINGLE_FILE_PATH_BIN)
        
    plates=[]             
    for l in range(0,len(arr_txt)):
        if(ONLY_SINGLE_FILE==False):
            path_txt=PATH+"\\"+arr_txt[l]
            path_bin=PATH+"\\"+arr_bin[l]    
        else:
            path_txt=arr_txt[l]
            path_bin=arr_bin[l]
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
        elif(len(cur_plate.segments_names)>len(cur_plate.sigments_sign)):
            num_to_remove = len(cur_plate.segments_names) - len(cur_plate.sigments_sign)
            for k in range(0,num_to_remove):
                del cur_plate.segments_names[-1]
                
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

#show all segments with labels assigned by user (for the given plate)
def ShowAllSingleSegmentsWithLabels(fig_id, plate,colors_code=None,indx_chan=0,aplpha=0.1):
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
    
    fig=plt.figure(fig_id)
    fig.clf()
    
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
            if(plate.sigments_labels[kks]!=[]):
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
        
    fig.legend(sectors,labels_tags)
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show() 
    #print(border)
#use example
#ShowAllSingleSegmentsWithLabels("ssdfsfsdf", PLATES_ARRAY[0],colors_code=cc_dd,indx_chan=0,aplpha=0.1)

#show signals and labelling in the figure
def ShowSingleSegmentWithLabels(fig_id, plate,indx_segment=0,colors_code=None,indx_chan=0,aplpha=0.1):
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

    #%matplotlib qt
    if plt.fignum_exists(fig_id)==True:
        fig=plt.figure(fig_id) #fig=Figure(fig_id)
    else:
        fig=plt.figure(fig_id) 
    fig.clf()  
    
    for i in range(0,len(chan_num)):
        plt.plot(plate.sigments_sign[indx_segment][chan_num[i]])
        if(plate.sigments_labels[indx_segment]!=[]):
            for k in plate.sigments_labels[indx_segment]:     
                start=k[0]
                end=k[1]
                label=k[2]                
                c_ind=unique_labels.index(label)
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

#****************************************************************************************
#****************************************************************************************
#   CLASSIFIER
class S_Classif:
    def __init__(self):
        self.classifier=None
        self.orig_labels=[]
        self.intern_labels=[]
    def AssignClassif(self,classif,orig_labels,intern_labels,categ_names=[]):
        self.classifier=classif
        self.orig_labels=orig_labels              
        self.intern_labels=intern_labels    
        self.categ_names=[]
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

#example
#cl=S_Classif()
#cl.AssignClassif(CLASSIFIER,None)

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
        num_snipps=np.round(sign_shape[1]/snip_size)
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
            return self.sign_4.cpu().numpy().copy()      

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

#******************************************************************************************************
#******************************************************************************************************

#read settings before execuring processing
def ReadSettings(window):
    settings={}
    #snippet 
    snip_size=int(window.ui.classification_snippet_size_text.text())
    settings["snippet_size"] = snip_size
    #preprocessing
    preproc=window.ui.classification_preproc_dropdown.currentText()
    settings["preprocessing"] = preproc
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
    
    return settings

#setings for display and graphics
def ReadGraphSettings(window):
    settings={}
    #labeling data - show
    classif_show_type= window.ui.classification_plot_choice_dropdown_3.currentText()
    settings["classification_show_labeled_data_type"] = classif_show_type
    return settings
    
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