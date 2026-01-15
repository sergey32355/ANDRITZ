
import array
from turtle import color
import numpy as np
import os
from os import walk
import sys
from time import sleep
import traceback
import joblib
import pandas as pd
import uuid 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
import distinctipy as dc
import time
import logging
import threading
import sys
import trace
import glob
import multiprocessing
from barbar import Bar
import warnings
import pyqtgraph as pg
import webcolors
#import pylab as pl

import PySide6
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit,QFileDialog
from PySide6.QtCore import QThread, Signal, QObject
from gui_files.empa_gui import Ui_EmpaGUI
from mod.mod_gui_data_loader_bpp_lines import DataLoaderBPPLines
from mod.mod_gui_feature_extractor_bpp_lines import FeatureExtractorBPPLines
from mod.mod_gui_model_selector_bpp_lines import ModelSelectorBPPLines, PredictorScorerBPPLines
from PySide6.QtWidgets import QMessageBox
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QVBoxLayout
from PySide6.QtWidgets import QMessageBox

#additional libs
#import librosa
import spcm #the DAQ card library
from spcm import units # spcm uses the pint library for unit handling (units is a UnitRegistry object)
import datetime

#classifiers
from xgboost import XGBClassifier
import pywt
import ptwt

import torch
import torchaudio
from torch import nn
import torchaudio.functional as F
import torchaudio.transforms as T

import SHelpers as shlp

#LINKS TOOLS

#pyside6-designer - call designer
#LINKS
##https://github.com/EnsiyeTahaei/DeepAnT-Time-Series-Anomaly-Detection/blob/main/deepant/model.py


#************************************************************************
#*************************************************************************
# Main Window Class
#*************************************************************************
#************************************************************************


DEBUG_MODE = False

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_EmpaGUI()
        self.ui.setupUi(self)
        # Make output text boxes read-only
        # Connect button click
        self.ui.load_data_button.clicked.connect(self.load_data)
        self.ui.load_model_button.clicked.connect(self.load_model)
        #self.ui.train_button.clicked.connect(self.train_model)
        self.ui.List_segments_names_button.clicked.connect(self.List_segment_names_click)
        #self.ui.plot_button.clicked.connect(self.plot_signal)
        #self.ui.predict_button.clicked.connect(self.predict)
        #SSH ADD-ON
        self.ui.Brows_data_folder_button.clicked.connect(self.browse_data_folder_click)
        self.ui.load_data_plate_type_dropdown.currentIndexChanged.connect(self.PlateLayout_click) #choose the layout of the plate
        self.ui.train_dropdown.currentIndexChanged.connect(self.ModelChoiceDropDown_click)#choose the model
        #SSH TOOLS tab
        #page 1
        self.ui.plot_plate_dropdown.currentIndexChanged.connect(self.PlateChoiceChanged_click)#drop box with plates names list
        self.ui.plot_button.clicked.connect(self.Plot_selected_segment_click)
        self.ui.platre_data_button_2.clicked.connect(self.Plate_info_click)
        self.ui.unload_as_csv_button_data_button_3.clicked.connect(self.Save_signal_as_scv_click)
        self.ui.calssification_plot_segment.clicked.connect(self.Classification_Plot_Segment) #for supervised clasification labelling we plot the signal
        self.ui.classification_set_category.clicked.connect(self.Classification_Set_Category_Click)
        self.ui.classification_clear_labels_button.clicked.connect(self.Classification_ClearLabelling_Click)
        self.ui.classification_clear_all_segments_labels_button_2.clicked.connect(self.Classification_ClearLabellingForAllSegments_Click)
        self.ui.classification_run_classification_button.clicked.connect(self.RunClassificationClick)
        self.ui.calssification_segment_savetofolder_button.clicked.connect(self.SaveLabelingAsCSV_Click)
        self.ui.Classification_extract_save_features_to_file_button.clicked.connect(self.SaveExtractedFeatToFIle)
        self.ui.Classification_run_test_on_complete_segment_button.clicked.connect(self.RunClassifForSegment_Click)
        self.ui.classification_dataset_info_button.clicked.connect(self.Classification_Dataset_Info_Click)
        self.ui.classification_entire_segment_to_label_button.clicked.connect(self.Classification_Entire_Segm_To_Label_Click)
        self.ui.classification_all_segments_to_label_button_2.clicked.connect(self.Classification_All_Segm_To_Label_Click)
        self.ui.tools_show_scpetrogram_button.clicked.connect(self.Tool_Show_Spectrograms_Botton_Click)
        #classification add-onns
        self.ui.Classification_save_to_file_labeling_button.clicked.connect(self.SaveLabelingToFile_Click)
        self.ui.Classification_load_from_file_labeling_button_2.clicked.connect(self.LoadLabelingFromFile_Click)
        #settings
        #colors
        self.ui.COlors_list_generate_button.clicked.connect(self.GenerateCOlors_Botton_Click)
        self.ui.save_interface_button.clicked.connect(self.Save_Interface_Click)
        self.ui.load_interface_button.clicked.connect(self.Load_Interface_Click)
        #classifiiers
        #autoencoder
        self.ui.Autoencoder_set_threshold_button.clicked.connect(self.Autoencoder_set_threshold_click)

        #SPECTROGARMS
        self.ui.Settings_test_MEL_Segment.clicked.connect(self.Tool_Test_Spectrograms_MEL_Botton_Click)
        self.ui.Settings_Test_MEL_snippet.clicked.connect(self.Tool_Show_Spectrograms_MEL_Botton_Click)
        
        #real time
        self.ui.Start_real_time_button.clicked.connect(self.Real_Time_Button_Click)
        self.ui.Stop_real_time_button_2.clicked.connect(self.Real_Time_Stop_Click)
        # Set dropdown options
        #model_names = [m.split(".")[0] for m in os.listdir("models") if m.endswith(".pkl")]
        #self.ui.load_model_dropdown.addItems(model_names)
        #set widgets starting conditions
        self.WidgetsStartingConditions()

        #local vairables
        self.load_data_thread = None
        self.load_data_worker = None
        self.extract_features_worker = None
        self.train_thread = None
        self.train_worker = None
        self.predict_thread = None
        self.predict_worker = None
        self.dl = None  # To store the DataLoaderBPPLines object
        self.fe = None  # To store the FeatureExtractorBPPLines object
        self.ms = None  # To store the ModelSelectorBPPLines object
        self.ps = None  # To store the PredictorScorerBPPLines object
        self.model = None  # To store the trained model
        self.threshold_params = None  # To store threshold parameters for trained model

        #SSH ADD-ON
        #plate
        self.plate_segments_id=[] #segnents of the plate as a string list
        self.plates=[] #plate segments with signals   
        #data preprocessing
        self.DataPreproc=None
        #classifier or regressor
        self.s_model = None
        #service
        self.proc_settings=None  #processing pipeline from GUI
        self.proc_graph_settings=None #display graphics settings from GUI        
        self.classif_plot_fig_id=""
        self.classif_fig=None
        #self.classif_tmp=None #here we store the classifier from the intermidiate test runs
        self.autoencoder_hist_form_id=""
        
        #threading for real time
        #folder check
        self.RT_Figure_if_orig_sgn = ""
        self.RT_Figure_if_proc_results_id = ""
        self.ExitFilesInFolderFlag=False
        self.Real_time_FolderTrackerThread=None
        #DAQ acquisition
        self.RT_SpectrumThread=None
        global EXIT_DAQ_FLAG
        EXIT_DAQ_FLAG=True
        #generate colors lists
        self.colors_id = None
        self.GenerateCOlors_Botton_Click() #dc.get_colors(500)   
        self.CheckToLoadDefaultGUI()

    def WidgetsStartingConditions(self):
        self.ui.Model_ready_label.setStyleSheet("background-color: red")

    ##GUI save/load
    def Save_Interface_Click(self):
        path=os.getcwd()
        path=path+"\\default_gui.cfg"
        try:
            shlp.SaveInterfaceIntoFile(self,path)
            print("GUI saved into: "+str(path))
        except Exception as exs:
            print("Could not save GUI to file. Exception: "+str(exs))

    def CheckToLoadDefaultGUI(self):
        path=os.getcwd()
        path=path+"\\default_gui.cfg"
        try:
            shlp.CheckOnStartToLoadGUI(self,path)
        except: print("Default GUI file is corrupted or absent")

    def Load_Interface_Click(self):
        path=os.getcwd()
        path=path+"\\default_gui.cfg"
        try:
            shlp.LoadInterfaceFromFile(self,path)
            print("GUI loaded: "+str(path))
        except Exception as exs: print("Cant load interface from file. Exception: "+str(exs))
        
    def GenerateCOlors_Botton_Click(self):        
        graph_sets=shlp.ReadGraphSettings(self)
        type_list=graph_sets.get("classif_color_list_type")
        col_num=int(graph_sets.get("classif_color_list_number"))
        
        def ColorsListGen(type_list,col_num):
            #https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
            colors_list=[]
            if(type_list=="Grades of red"):
                colors_list=['#228B22','#FF0000','#8B0000']
            if(type_list=="Random colors"):
                colors_list=dc.get_colors(col_num)
            return colors_list
        self.colors_id = shlp.ColorsListGen(type_list,col_num)
    
    #BUTTONS EVENTS
    #find folder with signal files
    def browse_data_folder_click(self):
        path_dir = QFileDialog.getExistingDirectory()         
        self.ui.load_data_path.setText(path_dir)

    def List_segment_names_click(self):
        print("")        
        print("SEGMENT NAMES LIST:")  
        print("Segm.num. - "+str(len(self.plate_segments_id)))
        print(self.plate_segments_id)

    def PlateLayout_click(self):        
        plate_type=self.ui.load_data_plate_type_dropdown.currentText()
        self.plate_segments_id=shlp.getSegmentNames(plate_type)
        
    def ModelChoiceDropDown_click(self):
        model_choice = self.ui.train_dropdown.currentIndex()  
        if(model_choice==1):
            path_dir = QFileDialog.getOpenFileName(self, str("Open File"),
                                                   "/home",
                                                   str("Model (*.plk)"))   
            if(path_dir==""):#if was canceled
                self.ui.load_data_plate_type_dropdown.setCurrentIndex(0)
                model_choice=0
            else:
                loaded_model_data = joblib.load(model_path)
                self.model = loaded_model_data['trained_model']
                self.threshold_params = loaded_model_data['best_threshold_params']
                
        if(model_choice==0):
            pass#this is for autoencoder

    #SSH TOOLS TAB EVENTS 
    def PlateChoiceChanged_click(self):
        if(self.plates is None):
            return        
        if(self.plates==[]):
            return
        cur_plate_name=self.ui.plot_plate_dropdown.currentText()
        indx_plate=-1
        for k in range(0,len(self.plates)):
            if(self.plates[k].name==cur_plate_name):
                indx_plate=k
                break
        #fill segments
        segm_available=self.plates[indx_plate].segments_names#
        self.ui.plot_segment_dropdown.clear()
        self.ui.plot_segment_dropdown.addItems(segm_available)
        self.ui.plot_segment_dropdown.setCurrentIndex(0)
        #fill channels
        chans_available=self.plates[indx_plate].chans_names.copy()
        chans_available.insert(0, "all")
        self.ui.Channel_segment_plot.clear()
        self.ui.Channel_segment_plot.addItems(chans_available)
        self.ui.Channel_segment_plot.setCurrentIndex(0)
        
    #plot sepctrogram for selected segment
    def Tool_Show_Spectrograms_Botton_Click(self):
        if(self.plates==None):
            print("plates are not loaded...")
            return
        if(self.plates==[]):
            print("plates are not loaded...")
            return
        self.proc_settings = shlp.ReadSettings(self)
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()   
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        signals=[]
        end_=indx_chan+1
        start_=indx_chan
        if(end_==-1): 
            end_ = len(self.plates[indx_plate].segment_sign[indx_segment]) #we taek all signals
            start=0
        SAMPLE_RATE=int(self.plates[indx_plate].sr)
        spec_type=self.proc_settings.get("spectrogrym_type")
        for i in range(start_,end_):
            # Define transform            
            waveform=torch.tensor(self.plates[indx_plate].sigments_sign[indx_segment][i])
            if(len(waveform.shape)==1): waveform = waveform[None, :]
            if(spec_type=="MEL"):
                    spectrogrym_MEL_nfft=int(self.proc_settings.get("spectrogrym_MEL_nfft"))
                    spectrogram = T.Spectrogram(n_fft=spectrogrym_MEL_nfft)
                    # Perform transform
                    spec = spectrogram(waveform)
                    #%matplotlib qt
                    fig, axs = plt.subplots(2, 1)
                    shlp.plot_waveform(waveform, SAMPLE_RATE, title="Original waveform", ax=axs[0])
                    shlp.plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
                    fig.tight_layout()
        
    #plot selected segment
    def Plot_selected_segment_click(self):
        
        if(self.plates==[]):
            return

        graph_settings=shlp.ReadGraphSettings(self)

        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()
        signal_type=self.ui.type_of_signal_dropdown.currentText()
        segment_name=self.ui.plot_segment_dropdown.currentText()   

        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        
        """
        #find plate in list
        indx_plate=-1
        for k in range(0,len(self.plates)):
            if(self.plates[k].name==p_name):
                indx_plate=k
                break
        #find channel in list
        indx_chan=-1
        if(ch_name!="all"):
            for k in range(0,len(self.plates[indx_plate].chans_names)):
                if(self.plates[indx_plate].chans_names[k]==ch_name):
                    indx_chan=k
                    break
        #find the segment
        indx_segment=-1
        for k in range(0,len(self.plates[indx_plate].segments_names)):
                if(self.plates[indx_plate].segments_names[k]==segment_name):
                    indx_segment=k
                    break
        """
        
        #print(ch_name)
        #print(indx_chan)
        #print(self.plates[indx_plate].chans_names)
        """
        %matplotlib qt
        if(plot_type=="Segment"):
            shlp.ShowSingleSegmentWithLabels(self.classif_plot_fig_id,self.plates[indx_plate],indx_segment=indx_segment,
                                                                                              colors_code=self.colors_id,
                                                                                              indx_chan=indx_chan)
        if(plot_type=="All_segments"):
            shlp.ShowAllSingleSegmentsWithLabels(self.classif_plot_fig_id, self.plates[indx_plate],
                                                 colors_code=self.colors_id,
                                                 indx_chan=indx_chan,
                                                 aplpha=0.1
                                                )
        """
        #%matplotlib qt
        if(signal_type=="Raw signals"):
            if(indx_chan!=-1):#this is the case of not all channels are selected
                fig=plt.figure(1)
                plt.clf()
                plt.plot(self.plates[indx_plate].time, self.plates[indx_plate].raw_signals[indx_chan])
                plt.show()
            else:
                fig=plt.figure(1)
                plt.clf()
                for kp in range(0,len(self.plates[indx_plate].chans_names)):
                    plt.plot(self.plates[indx_plate].time, self.plates[indx_plate].raw_signals[kp])
                plt.show()
        else: #segments
            if(indx_chan!=-1):#this is the case when single channel is selected                
                shlp.ShowSingleSegmentWithLabels(self.classif_plot_fig_id,
                                                 self.plates[indx_plate],
                                                 indx_segment=indx_segment,
                                                 colors_code=self.colors_id,
                                                 indx_chan=[indx_chan],
                                                 show_labels=False,
                                                 points_num_limit_check=bool(graph_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                 points_num_limit=int(graph_settings.get("GUI_show_results_points_number_limit_textbox")),

                                                )
            else:
                shlp.ShowAllSingleSegmentsWithLabels(self.classif_plot_fig_id, 
                                                     self.plates[indx_plate],
                                                     colors_code=self.colors_id,
                                                     indx_chan=indx_chan,
                                                     aplpha=0.1,
                                                     show_labels=False,
                                                     points_num_limit_check=bool(graph_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                     points_num_limit=int(graph_settings.get("GUI_show_results_points_number_limit_textbox")),
                                                     mark_segm_borders=bool(self.proc_settings.get("GUI_mark_segments_checkbox")),
                                                    )
                
                """
                t_start=self.plates[indx_plate].sigments_start_t
                d_t=self.plates[indx_plate].delta_t
                durat=d_t*len(self.plates[indx_plate].sigments_sign[indx_segment][indx_chan])
                time=np.linspace(t_start,t_start+durat,len(self.plates[indx_plate].sigments_sign[indx_segment][indx_chan]))                
                fig=plt.figure(1)
                plt.clf()
                plt.plot(time, self.plates[indx_plate].sigments_sign[indx_segment][indx_chan])
                plt.show()
                
            else:
                
                fig=plt.figure(1)
                plt.clf()
                for kp in range(0,len(self.plates[indx_plate].chans_names)):
                    t_start=self.plates[indx_plate].sigments_start_t
                    d_t=self.plates[indx_plate].delta_t
                    durat=d_t*len(self.plates[indx_plate].sigments_sign[indx_segment][indx_chan])
                    time=np.linspace(t_start,t_start+durat,len(self.plates[indx_plate].sigments_sign[indx_segment][indx_chan]))    
                    plt.plot(time, self.plates[indx_plate].sigments_sign[indx_segment][kp])
                plt.show()    
                """
                
    def Save_signal_as_scv_click(self):
        if(self.plates==[]):
            return
        
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()
        signal_type=self.ui.type_of_signal_dropdown.currentText()
        segment_name=self.ui.plot_segment_dropdown.currentText()   
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        
        path_dir = QFileDialog.getExistingDirectory()       
        if(path_dir==""): return
        else:
            if(indx_chan==-1):#we save all channels
                start_num=0
                chans_num=len(self.plates[indx_plate].chans_names)
            else:
                start_num=indx_chan
                chans_num=indx_chan+1
                
            for k in range(start_num,chans_num):
                    chan_sub_f=path_dir+"\\"+self.plates[indx_plate].chans_names[k]                    
                    if(os.path.exists(chan_sub_f)==False):
                        try: os.mkdir(chan_sub_f)
                        except:
                            print("Failed to creat a directory. Signals are not saved...")
                            return
                    f_name=chan_sub_f+"\\"+p_name+"_"+segment_name+"_"+ch_name+"_"+signal_type+".csv"
                    if os.path.exists(f_name)==True:
                        counter=0
                        while(True):
                            counter+=1
                            f_name=chan_sub_f+"\\"+str(counter)+"_"+p_name+"_"+segment_name+"_"+ch_name+"_"+signal_type+".csv"
                            if (os.path.exists(f_name)==False):
                                break
                    arr=self.plates[indx_plate].sigments_sign[indx_segment][k]
                    df = pd.DataFrame(arr)
                    df.to_csv(f_name)
            print("filed saved into folder...")
           
    #classification section - Tools->Page 1
    def Classification_Dataset_Info_Click(self):
                
        self.proc_settings = shlp.ReadSettings(self)
        print("")
        print("Settings:")
        print(self.proc_settings)
        


    def Classification_Plot_Segment(self):      
        
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return   
               
        self.proc_settings = shlp.ReadSettings(self)   
        show_data_type=str(self.proc_settings.get("classification_plot_choice_dropdown_3"))
        
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()    
                
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        if (self.classif_plot_fig_id==""):self.classif_plot_fig_id = str(uuid.uuid1())[:5]  

        #%matplotlib qt
        if(show_data_type == "Segment"):
            shlp.ShowSingleSegmentWithLabels(self.classif_plot_fig_id,self.plates[indx_plate],indx_segment=indx_segment,
                                                                                              colors_code=self.colors_id,
                                                                                              indx_chan=indx_chan,
                                                                                              show_labels=True,
                                                                                              points_num_limit_check=bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                                                              points_num_limit=int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox")),
                                                                                              )
        if(show_data_type=="All_segments"):
            segm_borders_show_flag=bool(self.proc_settings.get("GUI_mark_segments_checkbox"))
            shlp.ShowAllSingleSegmentsWithLabels(self.classif_plot_fig_id, self.plates[indx_plate],
                                                 colors_code=self.colors_id,
                                                 indx_chan=indx_chan,
                                                 aplpha=0.1,
                                                 show_labels=True,
                                                 points_num_limit_check=bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                 points_num_limit=int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox")),
                                                 mark_segm_borders=segm_borders_show_flag,
                                                )
            
        """
        if(plt.fignum_exists(self.classif_plot_fig_id)==False) or(self.classif_plot_fig_id==""):
            self.classif_plot_fig_id=str(uuid.uuid1())[:5]  
            %matplotlib qt
            self.classif_fig=plt.figure(self.classif_plot_fig_id)    
        shlp.ShowSignalInFigure(self.classif_fig,
                                plates=self.plates,
                                colors_code=self.colors_id,
                                indx_plate=indx_plate,
                                indx_segment=indx_segment,
                                indx_chan=indx_chan)       
        """
    #assign label to the segment pattern
    def Classification_Set_Category_Click(self):
        
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return
        
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()    
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        
        try:
            start_el = int(self.ui.classification_labeling_start_element_text.text())
            end_el=int(self.ui.classification_labeling_end_element_text.text())
            label=self.ui.classification_labeling_label_text.text()
            if(label=="" or label==" "):
                print("fill lable info and repeat...")
                return
            #self.plates[indx_plate].sigments_labels[indx_segment].append(list([start_el,end_el,label]))
            self.plates[indx_plate].AssignLabelToSegmentPattern(segment_indx=indx_segment,label=label,start_el=start_el,end_el=end_el)
        except:
            print("check the values in the text boxes")
            return
        
        print("")
        print("new label assigned to:")
        print("plate name - "+str(p_name))
        print("segment name - "+str(segment_name))
        print("patterns first samp.point - "+str(start_el))
        print("patterns last samp.point - "+str(end_el))
        print("total number of samp.points - "+str(end_el-start_el))
        print("label - "+str(label))
    
        if plt.fignum_exists(self.classif_plot_fig_id)==True:
            self.Classification_Plot_Segment()
    
    #assign label for entire segment
    def Classification_Entire_Segm_To_Label_Click(self):
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()    
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        
        try:
            label=(self.ui.classification_labeling_label_text.text())
            if(label=="" or label==" "):
                print("fill lable info and repeat...")
                return
            #self.plates[indx_plate].sigments_labels[indx_segment].append(list([start_el,end_el,label]))
            self.plates[indx_plate].AssignLabelToEntireSegment(segment_indx=indx_segment,label=label)
        except:
            print("check the values in the text boxes")
            return
        
        print("")
        print("label assigned to:")
        print("plate name - "+str(p_name))
        print("segment name - "+str(segment_name))       
        print("label - "+str(label))
        
        if plt.fignum_exists(self.classif_plot_fig_id)==True:
            self.Classification_Plot_Segment()
    
    #assign all labels to to category
    def Classification_All_Segm_To_Label_Click(self):
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return        
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()    
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        label_=(self.ui.classification_labeling_label_text.text())
        self.plates[indx_plate].AssignLabelToAllSegments(label=label_)

        print("")
        print("label assigned to:")
        print("plate name - "+str(p_name))
        print("segment num. - "+str(len(self.plates[indx_plate].sigments_sign)))       
        print("label - "+str(label_))

        if plt.fignum_exists(self.classif_plot_fig_id)==True:
            self.Classification_Plot_Segment()

    #clear fo only this segment
    def Classification_ClearLabelling_Click(self):
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()    
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        self.plates[indx_plate].sigments_labels[indx_segment]=[]

        print("")
        print("Labelling for plate - "+str(p_name)+" , segment - "+str(segment_name) +" erased...")
        
        if plt.fignum_exists(self.classif_plot_fig_id)==True:
            self.Classification_Plot_Segment()
        

    def Classification_ClearLabellingForAllSegments_Click(self):
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()   
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        l_sl=len(self.plates[indx_plate].sigments_labels)
        for jk in range(0,l_sl):
            self.plates[indx_plate].sigments_labels[jk]=[]
        if plt.fignum_exists(self.classif_plot_fig_id)==True:
            self.Classification_Plot_Segment()
        print("")
        print("Labelling for plate - "+str(p_name)+" is erased...")
                

    #************************************************************************************************************************************************************
    #**************************************RUN CLASSIFICATION****************************************************************************************************
    #************************************************************************************************************************************************************

    def RunClassificationClick(self):
        
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return
            
        self.proc_settings = shlp.ReadSettings(self)
        #read settings
        snip_size = int(self.proc_settings.get("snippet_size")) #int(self.ui.classification_snippet_size_text.text())
        test_size=int(self.ui.classification_test_text.text())/100.0
        classif_name= self.ui.classificationclassifier_dropdown.currentText()
        preproc= self.proc_settings.get("classification_preproc_dropdown")#self.ui.classification_preproc_dropdown.currentText()
        
        #start to create dataset from segment
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()    
        #collect general information
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)        
        channels_to_use=[indx_chan]

        CHANNELS_TO_USE=self.GetChannels()

        """
        try:
            if( int(self.proc_settings.get("classification_channels_choice_drop_down")) == 0):
                chan_index= int(self.proc_settings.get("chan_from_settings"))-1
                if(chan_index<0):
                    CHANNELS_TO_USE=list([0,1,2,3,4,5,6,7]) #all channels are selected
                else:
                    CHANNELS_TO_USE=list([chan_index])
            
            else:
                CHANNELS_TO_USE = [int(i) for i in self.proc_settings.get("classification_user_channels_text_box").split(",") if i.strip().isdigit()]
        except Exception as ex:
            print("Channels to use are defined incorrectly. Exception raised: "+str(ex))

        if(len(CHANNELS_TO_USE)==0):
            print("define channels to use and repeat...")
            return
        """

        """
        if(self.proc_settings.get("chans_to_use")=="From settings"):
            pass
        else:
            chans_string=self.proc_settings.get("chans_list_user")
            channels_to_use = shlp.ExtractChannelsFromString(chans_string,separator=",")
            if(channels_to_use is None):
                print("Exiting training. Check settings and repeat.")
                return
            else: pass
        """
                
        print("")
        print("*********************************************************************")
        print("Training start")
        print("User selected channels: "+str(CHANNELS_TO_USE))    
                
        #chans_to_use=window.ui.classification_channels_choice_drop_down.currentText()
        #settings["chans_to_use"] = chans_to_use
        #chans_list_user=window.ui.classification_user_channels_text_box.text()
        #settings["chans_list_user"] = chans_list_user
    
        #makes snippets
        #feat,labs = shlp.SplitIntoSnips(plates=self.plates,snip_size=snip_size,plate_name=p_name,chan_name=ch_name,segment_name=segment_name)
        unique_labs_tags=[]
        feat=[]
        labs=[]
        if(self.DataPreproc is None): self.DataPreproc=shlp.DataPreproc(n_fft=int(self.proc_settings.get("spectrogrym_MEL_nfft")),
                                                                        n_mels=int(self.proc_settings.get("Settings_MEL_num_MELS_2")),
                                                                        n_mfcc=int(self.proc_settings.get("Settings_nmfcc_num_MFCC_text")),
                                                                        )   

        plate_segm=self.proc_settings.get("plate_segm_process")        
        if(plate_segm == "this_plate_this_segment"):
            feat,labs= self.DataPreproc.SplitLabPlateSegmentIntoSnips(self.plates[indx_plate],
                                                                      snip_size=snip_size,
                                                                      segm_index=indx_segment,
                                                                      channs_indx=CHANNELS_TO_USE,#indx_chan,
                                                                      torch_tensor=False, 
                                                                      preproc_type=preproc
                                                                     )    
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            #print(np.shape(np.asarray(feat)))
            unique_labs_tags=self.plates[indx_plate].GetUniqueLabelsList()
            #print(unique_labs_tags)
        if(plate_segm == "this_plate_all_segments"):                        
            feat,labs= self.DataPreproc.SplitAllLabPlateSegmentsIntoSnips(self.plates[indx_plate],
                                                                          snip_size=snip_size,
                                                                          channs_indx=CHANNELS_TO_USE,#indx_chan,
                                                                          torch_tensor=False, 
                                                                          preproc_type=preproc
                                                                         )    
            
            feat,labs=self.DataPreproc.Helper_FlatListOfLabeledFeat(feat,labs)
            unique_labs_tags=self.plates[indx_plate].GetUniqueLabelsList()
            
        if(plate_segm == "all_plates_all_segments"):
            feat,labs= self.DataPreproc.SplitAllLabPlateOfAllSegmentsIntoSnips(self.plates,
                                                                               channs_indx=CHANNELS_TO_USE,#indx_chan,
                                                                               torch_tensor=False,
                                                                               snip_size=snip_size,
                                                                               preproc_type=preproc
                                                                              )
            rrt=[]
            for l in range(0,len(self.plates)):
                jk=self.plates[l].GetUniqueLabelsList()
                for k in range(0,len(jk)):
                    rrt.append(jk[k])
            unique_labs_tags=shlp.GetUniqueElements_List(list(rrt))#list(set(rrt))

        #le = preprocessing.LabelEncoder()
        #labs_num = le.fit_transform(labs)
        new_lab=[]
        tag_tmp=np.arange(0,len(unique_labs_tags))#linspace(0,len(unique_labs_tags),len(unique_labs_tags))
        for xx in range(0,len(labs)):
            indx=unique_labs_tags.index(labs[xx])
            tag=int(tag_tmp[indx])
            new_lab.append(tag)
                    
        global CLASSIF_FEAT
        CLASSIF_FEAT = feat
        global CLASSIF_LABS
        CLASSIF_LABS = new_lab
                        
        #*****************output info**************************
        try:
            print("")
            print("Start calssification...")
            print("Settings...")
            print("data source - "+str(plate_segm))
            if(plate_segm == "this_plate_this_segment"):
                print("plate name - "+str(p_name))
                print("segment name - "+str(segment_name))
            if(plate_segm == "this_plate_all_segments"):
                print("plate name - "+str(p_name))
                print("egm.num - "+str(len(self.plates[indx_plate].sigments_sign)))
            print("chan. name - "+str(ch_name))
            print("snip. length - "+str(snip_size))
            print("test set (% from training) - "+str(test_size*100))
            try:                 
                print("unique lab. - "+str(unique_labs_tags))  
                print("unique lab. encoding "+str(shlp.GetUniqueElements_List(new_lab)))
            except: print("unique labels are undefined")   
            print("feat.shape - " + str(np.shape(np.asarray(feat))))
            #print("train shape - "+str(np.shape(np.asarray(X_train))))
            #print("test shape - "+str(np.shape(np.asarray(X_test))))
        except:
            pass
        #******************************************************
        
        #classification     
        clf=None        
        self.s_model=None
                
        if(self.proc_settings.get("algorithm")=="XGBoost"):    

            unique_labs=shlp.GetUniqueElements_List(new_lab)
            if(len(unique_labs)<=1):
                print("")
                if(self.proc_settings.get("plate_segm_process")=="this_plate_this_segment"):
                    print("In this segment only one label is detected. Add another label to segment and inlcude other plates/segments.")                    
                if(self.proc_settings.get("plate_segm_process")=="this_plate_all_segments"):
                    print("In all plate segments only one label is detected. Add another label or inlcude other plates.")
                else:
                    print("In all plates only one label is detected.Add another label...")
                return
            X_train, X_test, y_train, y_test = train_test_split(feat, new_lab, test_size=test_size)
            print("training with XGBoost...")               
            clf,tests_l,orig_labs,intern_labs = shlp.XGBoostClassifRun(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                                                       trees_num=int(self.proc_settings.get("Settings_Trees_Trees_Number_2")),
                                                                       max_depth=int(self.proc_settings.get("Settings_Trees_Tree_depth_text_3")),
                                                                       learn_rate=float(self.proc_settings.get("Settings_Trees_learn_rate_value_text_4")),
                                                                       )   
            
            cm = confusion_matrix(y_test,tests_l)#, xgb_labs_back)            
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(clf,unique_labs_tags,tag_tmp)  
            print(str(type(self.s_model.classifier)))
            #%matplotlib qt
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
            disp.plot(cmap=plt.cm.Blues)            
            plt.show()               
            
        if(self.proc_settings.get("algorithm")=="IsolationForest"):
            
            #binary anomaly detector
            clf = shlp.IsolTreesTraining(feat=feat,labels=new_lab,
                                         trees_num=int(self.proc_settings.get("Settings_Trees_Trees_Number_2")),
                                         max_depth=int(self.proc_settings.get("Settings_Trees_Tree_depth_text_3")),
                                         contaminat=float(self.proc_settings.get("Settings_Trees_contaminations_value_text_5"))
                                         )                        
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(clf,None,None)  

        if(self.proc_settings.get("algorithm")=="DBSCAN"):
            clf = shlp.DBSCANTraining(feat=feat,labels=new_lab)                        
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(clf,None,None)  

        if(self.proc_settings.get("algorithm")=="OneClassSVM"):
            clf = shlp.OneCLassSVMTraining(feat=feat,labels=new_lab)#OneClassSVM(nu=0.1, kernel="rbf",gamma=0.2).fit(feat)                      
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(clf,None,None)  

        if(self.proc_settings.get("algorithm")=="Autoencoder_1"):
            threshold=float(self.proc_settings.get("autoencoder_torch_thershold"))
            lr=float(self.proc_settings.get("autoencoder_torch_learn_rate"))
            epochs_num=int(self.proc_settings.get("autoencoder_torch_epochs_num"))
            #train/test classifier
            model=shlp.TrainTorchAutoencoder(feat=feat,labels=new_lab,num_epochs=epochs_num,lr=lr)            
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(model,None,None)  
            self.s_model.tocrh_autoencoder_threshold_factor=threshold

        if(self.proc_settings.get("algorithm")=="Autoencoder_2"):
            clf,train_phi,train_mu,train_cov,args,loss=shlp.Train_Autoencoder_2(feat,new_lab,
                                                                      num_of_epochs=int(self.proc_settings.get("autoencoder_torch_epochs_num")),
                                                                      patience=10,
                                                                      lr=float(self.proc_settings.get("autoencoder_torch_learn_rate")),#1e-6,
                                                                      lr_milestones=[10],
                                                                      batch_size=10,
                                                                      latent_dim=50,
                                                                      n_gmm=5,
                                                                      lambda_energy=0.1,
                                                                      lambda_cov=0.00005,                                                                             
                                                                     )
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(clf,None,None)  
            self.s_model.autoenc2_train_phi=train_phi
            self.s_model.autoenc2_train_mu=train_mu
            self.s_model.autoenc2_train_cov=train_cov
            self.s_model.autoenc2_args=args
            threshold=float(self.proc_settings.get("autoencoder_torch_thershold"))
            self.s_model.tocrh_autoencoder_threshold_factor=threshold

        if(self.proc_settings.get("algorithm")=="AE_CNN_Shallow"):
            clf=shlp.TrainAE_1(feat,new_lab,
                                    AE_type="CNN_shallow",
                                    lr=float(self.proc_settings.get("autoencoder_torch_learn_rate")),#0.001,
                                    num_epochs=int(self.proc_settings.get("autoencoder_torch_epochs_num")),
                                    visualize=True,
                              )            
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(clf,None,None)  
            self.s_model.tocrh_autoencoder_threshold_factor=float(self.proc_settings.get("autoencoder_torch_thershold"))

        if(self.proc_settings.get("algorithm")=="ResNet_Shallow"):
            clf,_=shlp.Train_ResNetAE(feat,new_lab,
                                     init_lr=float(self.proc_settings.get("autoencoder_torch_learn_rate")),#0.001,
                                     batch_size=30,
                                     weight_decay=float(self.proc_settings.get("Autoencoder_ResNet_weight_decay_input_text")),
                                     num_epochs = int(self.proc_settings.get("autoencoder_torch_epochs_num")),
                                     visualize=True 
                                    )
            self.s_model=shlp.S_Classif()
            self.s_model.AssignClassif(clf,None,None)  
            self.s_model.tocrh_autoencoder_threshold_factor=float(self.proc_settings.get("autoencoder_torch_thershold"))
            
        if(self.proc_settings.get("algorithm")=="NonStatKern_1"):
            #literature links - non-stationary kernels
            #https://www.sgp-tools.com/tutorials/non_stationary_kernels.html
            #https://github.com/google/neural-tangents
            pass

        if(self.proc_settings.get("algorithm")=="GHKern"):
            #https://github.com/paulinebourigault/GHKernelAnomalyDetect
            pass        

        if(self.s_model!=None):
            self.s_model.train_feat=feat
            self.s_model.train_labs=new_lab
            self.ui.Model_ready_label.setStyleSheet("background-color: green") 
            global CLASSIFIER
            CLASSIFIER=clf#self.classif_tmp
            print("Training is complete. Classifier type: "+str(type(self.s_model.classifier)))#CLASSIFIER)))  
            
              
            
    def SaveLabelingAsCSV_Click(self):
        
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return        
            
        path_dir = QFileDialog.getExistingDirectory()          
        #start to create dataset from segment
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()    
        #collect general information
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)        
        if(indx_chan==-1):#save all channels
            start_=0
            end_=len(self.plates[indx_plate].chans_names)
        else:
            start_=indx_chan
            end_=indx_chan+1

        for l in range(0,len(self.plates[indx_plate].sigments_labels[indx_segment])):
            #save labels here
            label=int(self.plates[indx_plate].sigments_labels[indx_segment][l][2])
            labels_path=path_dir+"\\Label_"+str(label)
            if(os.path.exists(labels_path)==False):
                        try: os.mkdir(labels_path)
                        except:
                            print("Failed to creat a directory. Signals are not saved...")
                            return
            #save channels
            for k in range(start_,end_):             
                    chan_sub_f=labels_path+"\\"+self.plates[indx_plate].chans_names[k]                    
                    if(os.path.exists(chan_sub_f)==False):
                        try: os.mkdir(chan_sub_f)
                        except:
                            print("Failed to creat a directory. Signals are not saved...")
                            return
                    f_name=chan_sub_f+"\\"+p_name+"_"+segment_name+"_"+ch_name+"_"+"_label_"+str(label)+".csv"
                    if os.path.exists(f_name)==True:
                        counter=0
                        while(True):
                            counter+=1
                            f_name=chan_sub_f+"\\"+str(counter)+"_"+p_name+"_"+segment_name+"_"+ch_name+"_"+signal_type+"_label_"+str(label)+".csv"
                            if (os.path.exists(f_name)==False):
                                break
                    st1=int(self.plates[indx_plate].sigments_labels[indx_segment][l][0])            
                    st2=int(self.plates[indx_plate].sigments_labels[indx_segment][l][1])            
                    arr=self.plates[indx_plate].sigments_sign[indx_segment][k][st1:st2]
                    df = pd.DataFrame(arr)
                    df.to_csv(f_name)

    def SaveExtractedFeatToFIle(self):
        global CLASSIF_FEAT        
        global CLASSIF_LABS
        if(len(np.shape(np.asarray(CLASSIF_FEAT)))==0):
            print("features are not extracted. Extract features and repeat.")
            return
        else:
            path_file, _ = QFileDialog.getSaveFileName(self, "Save features", "", "Pickle Files (*.pkl);;All Files (*)")
            print("saveing feature tensor of shape: "+str(np.shape(np.asarray(CLASSIF_FEAT))))
            df=pd.DataFrame({'labels': CLASSIF_LABS, 'features': CLASSIF_FEAT})
            df.to_pickle(path_file)
            print("Features are saved into file.")
        """
        if(self.s_model is None):
            print("Prepare the model first and repeat....")
            self.ui.Model_ready_label.setStyleSheet("background-color: red")
        self.model=self.s_model
        self.ui.Model_ready_label.setStyleSheet("background-color: green")        
        #if(self.model is None): self.ui.Model_ready_label.setStyleSheet("background-color: red")
        #else: self.ui.Model_ready_label.setStyleSheet("background-color: green")        
        self.ui.train_dropdown.setCurrentText(self.ui.classificationclassifier_dropdown.currentText())  
        """
    #****************************************************************************************************************
    #****************************************************************************************************************
    #*********************READ CLASSification************************************************************************

    def RunClassifForSegment_Click(self):

        self.proc_settings = shlp.ReadSettings(self)
        show_results_scheme=str(self.proc_settings.get("Show_results_color_scheme_drop_down_1"))

        if(self.plates is None): return
        if(len(self.plates)==0): return
        if(self.s_model is None): 
            print("prepare the classifier first and repeat...")
            return
        #check the plate
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()   
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates,plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        snip_size=int(self.proc_settings.get("snippet_size"))
        CHANNELS_TO_USE=self.GetChannels()
        dp=shlp.DataPreproc(n_fft=int(self.proc_settings.get("spectrogrym_MEL_nfft")),
                            n_mels=int(self.proc_settings.get("Settings_MEL_num_MELS_2")),
                            n_mfcc=int(self.proc_settings.get("Settings_nmfcc_num_MFCC_text")),
                            )     

        """
        channels_to_use=[indx_chan]
        if(self.proc_settings.get("chans_to_use")=="From settings"):
            pass
        else:
            chans_string=self.proc_settings.get("chans_list_user")
            channels_to_use = shlp.ExtractChannelsFromString(chans_string,separator=",")
            if(channels_to_use is None):
                print("Exiting training. Check settings and repeat.")
                return
            else:
                pass
        """

        """
        #now this is a function
        if( int(self.proc_settings.get("classification_channels_choice_drop_down")) == 0):
            chan_index= int(self.proc_settings.get("chan_from_settings"))-1
            if(chan_index<0):
                CHANNELS_TO_USE=list([0,1,2,3,4,5,6,7]) #all channels are selected
            else:
                CHANNELS_TO_USE=list([chan_index])
            
        else:
            CHANNELS_TO_USE = [int(i) for i in self.proc_settings.get("classification_user_channels_text_box").split(",") if i.strip().isdigit()]
        """

        print("")
        print("*********************************************************************")
        print("Start test on segment.")
        print("Selected channels: "+str(CHANNELS_TO_USE))              
                    
        #%matplotlib qt
        #show figure with results
        
        if(self.proc_settings.get("plate_segm_process")=="this_plate_this_segment"):

            fd=dp.SplitEntireSignalIntoSnippets(signal=self.plates[indx_plate].sigments_sign[indx_segment],
                                            channs_indx=CHANNELS_TO_USE,
                                            torch_tensor=False,
                                            snip_size = snip_size,
                                            preproc_type = self.proc_settings.get("classification_preproc_dropdown")
                                           )        
        
            labels=self.s_model.np_predict(np.asarray(fd))
            classif_type=self.s_model.getClassifType()                   
            sgn=self.plates[indx_plate].sigments_sign[indx_segment][indx_chan] 
            
            labels_in_segment=[]
            for l in range(0,len(self.plates[indx_plate].sigments_sign)):
                if(l==indx_segment): labels_in_segment.append(labels)
                else: labels_in_segment.append([])

            fig_id=str("Segment_test_"+str(uuid.uuid1())[:5])
            shlp.ShowAllSingleSegmentsWithLabels(fig_id,self.plates[indx_plate],
                                                 colors_code = self.colors_id,
                                                 indx_chan = CHANNELS_TO_USE,
                                                 aplpha = 0.1,
                                                 show_labels = False,
                                                 points_num_limit_check = bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                 points_num_limit = int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox")),
                                                 show_proc_labels = True, #this is for processed labels
                                                 proc_labels_snip_size = snip_size,
                                                 proc_labels_color_scheme = self.proc_settings.get("Show_results_color_scheme_drop_down_text"),
                                                 proc_labels = labels_in_segment,   
                                                 mark_segm_borders=bool(self.proc_settings.get("GUI_mark_segments_checkbox")),
                                                 )


            #show everything
            """
            fig, ax = plt.subplots(1, sharex=True, figsize=(6, 6))        
            ax.plot(sgn)        
            if(True):            
                st_=0
                en_=snip_size
                colors_l=[]
                unique_l=[]
                line_v=[]
                
                for l in range(0,len(labels)):
                    cur_color=self.colors_id[int(labels[l])]
                    line_=ax.axvspan(st_, en_, alpha=0.1, color=cur_color)
                    st_=en_
                    en_=st_+snip_size   
                    if(int(labels[l]) not in unique_l):
                        unique_l.append(int(labels[l]))
                        colors_l.append(cur_color)
                        line_v.append(line_)    
                legend_tags=[]
                orig_tags=self.s_model.orig_labels
                for l in range(0,len(unique_l)):
                    if(orig_tags is not None):
                        legend_tags.append(str(orig_tags[l]))  
                    else:
                        legend_tags.append(str(unique_l[l]))  
                plt.legend(line_v,legend_tags)
                plt.show()
            """


        #only for this pattern
        if(self.proc_settings.get("plate_segm_process")=="this_plate_all_segments"):
            
            fig, ax = plt.subplots(1, sharex=True, figsize=(6, 6))        
            segm_num=len(self.plates[indx_plate].sigments_sign)

            borders=[]
            borders.append(0)
            ymin=[]
            ymax=[]

            unique_l=[]
            colors_l=[]
            
            labels_in_segment=[]

            for n_segm in range(0,segm_num):

                fd=dp.SplitEntireSignalIntoSnippets(signal=self.plates[indx_plate].sigments_sign[n_segm],
                                            channs_indx=CHANNELS_TO_USE,
                                            torch_tensor=False,
                                            snip_size = snip_size,
                                            preproc_type = self.proc_settings.get("classification_preproc_dropdown")
                                           )        

                labels=self.s_model.np_predict(np.asarray(fd))     
                labels_in_segment.append(labels)

            fig_id=str("Segment_test_"+str(uuid.uuid1())[:5])
            shlp.ShowAllSingleSegmentsWithLabels(fig_id,self.plates[indx_plate],
                                                 colors_code = self.colors_id,
                                                 indx_chan = CHANNELS_TO_USE,
                                                 aplpha = 0.1,
                                                 show_labels = False,
                                                 points_num_limit_check = bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                 points_num_limit = int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox")),
                                                 show_proc_labels = True, #this is for processed labels
                                                 proc_labels_snip_size = snip_size,
                                                 proc_labels_color_scheme = self.proc_settings.get("Show_results_color_scheme_drop_down_text"),
                                                 proc_labels = labels_in_segment,         
                                                 mark_segm_borders=bool(self.proc_settings.get("GUI_mark_segments_checkbox")),
                                                 )

            """
                classif_type=self.s_model.getClassifType()                        
                sgn=self.plates[indx_plate].sigments_sign[n_segm][indx_chan]  
                ymin.append(np.min(sgn))
                ymax.append(np.max(sgn))
                
                start_=borders[n_segm]
                l_shp=np.shape(np.asarray(sgn))#self.plates[indx_plate].sigments_sign[n_segm]))
                if(len(l_shp)==2): l_=l_shp[1]
                if(len(l_shp)==1): l_=l_shp[0]
                end_=start_+l_
                borders.append(start_+l_)
                x_=np.arange(start_,end_)
                #print("XXXXXXXXXXXXXXXXXX")
                #print(l_shp)
                #print(start_)
                #print(end_)
                #print(np.shape(sgn))
                ax.plot(x_,sgn,color="blue")     
                
                if(True):                  
                    st_=start_
                    en_=st_+snip_size
                    colors_l=[]
                    unique_l=[]
                    line_v=[]
                    #fill the backgrounds
                    for l in range(0,len(labels)):
                        cur_color=self.colors_id[int(labels[l])]
                        line_=ax.axvspan(st_, en_, alpha=0.1, color=cur_color)
                        st_=en_
                        en_=en_+snip_size   
                        if(int(labels[l]) not in unique_l):
                            unique_l.append(int(labels[l]))
                            colors_l.append(cur_color)
                            line_v.append(line_)    
                    legend_tags=[]
                    orig_tags=self.s_model.orig_labels
                
                for l in range(0,len(unique_l)):
                    if(orig_tags is not None):
                        legend_tags.append(str(orig_tags[l]))  
                    else:
                        legend_tags.append(str(unique_l[l]))  

            for jj in range(0,len(borders)):
                if(jj<len(borders) and jj<len(ymin) and jj<len(ymax)):
                    ax.vlines(borders[jj], ymin[jj], ymax[jj], colors="red")
            
            plt.legend(line_v,legend_tags)
            plt.show()
            """

    #save labeled data from file
    def SaveLabelingToFile_Click(self):
        #save all labels with plates into file
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return
        path_file, _ = QFileDialog.getSaveFileName(self, "Save Labeling", "", "Pickle Files (*.pkl);;All Files (*)")

        df=pd.DataFrame(columns=['Plate_name','Segment_name','Labeled_data'], index=['p_n','s_n','l_d'])
        for plate in self.plates:
            p_name=plate.name
            for s_indx in range(0,len(plate.sigments_sign)):
                s_name=plate.segments_names[s_indx]
                for label in plate.sigments_labels[s_indx]:
                    l_data=label
                    df_temp=pd.DataFrame({'Plate_name':[p_name],'Segment_name':[s_name],'Labeled_data':[l_data]})
                    df=pd.concat([df,df_temp],ignore_index=True)                     
        df.to_pickle(path_file)
        print("")
        print("Data successfully saved into file: "+str(path_file))

    #load labeled data from file
    def LoadLabelingFromFile_Click(self):

        #load all labels from file
        if(self.plates==[]) or(len(self.plates)==0) or (self.plates is None):
            print("Plates are not loaded")
            return

        path_file, _ = QFileDialog.getOpenFileName(self, "Load Labeling", "", "Pickle Files (*.pkl);;All Files (*)")
        try:
            df=pd.read_pickle(path_file)
        except:
            print("Failed to load the file. Repeat...")
            return
        #assign labels to plates
        for k in range(0,len(df)):
            p_name=df['Plate_name'][k]
            s_name=df['Segment_name'][k]
            l_data=df['Labeled_data'][k]
            #find plate and segment
            indx_plate=-1
            for m in range(0,len(self.plates)):
                if(self.plates[m].name==p_name):
                    indx_plate=m
                    break
            if(indx_plate==-1):
                print("Plate is not found in exsitsing plates list. Plate name - " + str(p_name))
                continue

            indx_segment=-1
            for n in range(0,len(self.plates[indx_plate].segments_names)):
                if(self.plates[indx_plate].segments_names[n]==s_name):
                    indx_segment=n
                    break
            if(indx_segment==-1):continue
            #assign label
            self.plates[indx_plate].sigments_labels[indx_segment].append(l_data)
        print("")
        print("Labels successfully loaded from file: "+str(path_file))
                
        
    def Plate_info_click(self):
        p_name = self.ui.plot_plate_dropdown.currentText()
        indx_plate=-1
        for k in range(0,len(self.plates)):
            if(self.plates[k].name==p_name):
                indx_plate=k
                break
        self.plates[indx_plate].plate_info()
        
    def set_buttons(self, enabled: bool):        
        self.ui.load_data_button.setEnabled(enabled)
        self.ui.load_model_button.setEnabled(enabled)
        self.ui.train_button.setEnabled(enabled)
        self.ui.predict_button.setEnabled(enabled)
        #self.ui.plot_button.setEnabled(enabled)

    def load_data(self):
        PATH=self.ui.load_data_path.text()        
        #update plates layout data        
        plate_type=self.ui.load_data_plate_type_dropdown.currentText()
        self.plate_segments_id=shlp.getSegmentNames(plate_type)          
        #clean everything
        self.plates=[]                        
        self.plates=None #plate segments with signals     
        self.ui.plot_plate_dropdown.clear()
        self.ui.plot_segment_dropdown.clear()
        self.ui.Channel_segment_plot.clear()    
        
        self.plates=shlp.OpenDataFromFolder(PATH=PATH,
                                            SEGMENTATION_REF_CHAN_NAME="Trigger",
                                            SEGMENTATION_THRESHOLD="automatic"    ,
                                            SEGMENTATION_SEGMENTS_NAMES_LIST=self.plate_segments_id
                                           )
        
        #fill the gui with plates collection
        plates_names=[]
        for k in self.plates:
            plates_names.append(k.name)
        
        global PLATES
        PLATES=self.plates
        
        self.ui.plot_plate_dropdown.clear()
        self.ui.plot_plate_dropdown.addItems(plates_names)     
        self.ui.plot_plate_dropdown.setCurrentIndex(0)
        #set teh available segments
        self.ui.plot_segment_dropdown.clear()
        segm_available=list(self.plates[0].segments_names.copy())        
        self.ui.plot_segment_dropdown.addItems(segm_available)
        self.ui.plot_segment_dropdown.setCurrentIndex(0)
        #channels for this plat        
        chans_available=self.plates[0].chans_names.copy()
        chans_available.insert(0, "all")
        self.ui.Channel_segment_plot.clear()
        self.ui.Channel_segment_plot.addItems(chans_available)
        self.ui.Channel_segment_plot.setCurrentIndex(1)
        #globals assign
        print("Loaded plates num.: "+str(len(self.plates)))
        global PLATES_ARRAY
        PLATES_ARRAY=self.plates.copy()        
        
    def load_data_error(self, err_text):
        # Option 1: append to text box        
        # # Option 2: pop up a message box
        # from PySide6.QtWidgets import QMessageBox
        # QMessageBox.critical(self, "Error", err_text)
        # ensure thread stops
        if hasattr(self, "load_data_thread") and self.load_data_thread.isRunning():
            self.load_data_thread.quit()
        # re-enable buttons
        self.set_buttons(enabled=True)

    def load_data_finished(self, result):
        # Store the DataLoaderBPPLines object
        self.dl = result
        filenames = [plate.identifier for plate in self.dl.list_bpp]
        # Add available options for plotting to dropdown
        self.ui.plot_plate_dropdown.clear()
        self.ui.plot_plate_dropdown.addItems(filenames)
        segments = ["ALL"]+[f"{seg[0]}_{seg[1]}" for seg in self.dl.segment_keys]
        self.ui.plot_segment_dropdown.clear()
        self.ui.plot_segment_dropdown.addItems(segments)

    def extract_features_finished(self, result):
        # Store the FeatureExtractorBPPLines object
        self.fe = result        
        self.set_buttons(enabled=True)
        # # Reset worker and thread references after cleanup
        # self.load_data_worker = None
        # self.extract_features_worker = None
        # self.load_data_thread = None

    def train_model(self):
        # Clear previous output        
        self.set_buttons(enabled=False)
        # Setup QThread + Worker
        self.train_thread = QThread()        
        self.train_worker.moveToThread(self.train_thread)
        # Connect signals:
        # When thread starts, run train_worker
        self.train_thread.started.connect(self.train_worker.run)
        # When the worker emits an error signal (using error.emit()), it triggers the self.train_error method
        self.train_worker.error.connect(self.train_error)
        # When the worker emits the message signal (using message.emit()), it triggers the self.train_finished method
        self.train_worker.message.connect(self.train_finished)
        # When the worker finishes, quit the thread
        self.train_worker.finished.connect(self.train_thread.quit)
        # When the thread completes (successfully or with error), it emits the finished signal
        # The deleteLater() method is called, which schedules the thread object for deletion
        # Memory is cleaned up automatically when Qt's event loop processes the deletion
        self.train_thread.finished.connect(self.train_thread.deleteLater)        
        # Start thread
        self.train_thread.start()

    def train_error(self, err_text):
        # Option 1: append to text box
        
        # # Option 2: pop up a message box
        # from PySide6.QtWidgets import QMessageBox
        # QMessageBox.critical(self, "Error", err_text)
        # ensure thread stops
        if hasattr(self, "train_thread") and self.train_thread.isRunning():
            self.train_thread.quit()
        # re-enable button
        self.set_buttons(enabled=True)

    def train_finished(self, result):
        # Store the ModelSelectorBPPLines object
        self.ms = result
        self.model = result.trained_model
        self.threshold_params = result.best_threshold_params        
        # Set dropdown options
        #self.ui.load_model_dropdown.clear()
        #model_names = [m.split(".")[0] for m in os.listdir("models") if m.endswith(".pkl")]
        #self.ui.load_model_dropdown.addItems(model_names)
        self.set_buttons(enabled=True)
        # # Reset worker and thread references
        # self.train_worker = None
        # self.train_thread = None

    def load_model(self):        
        self.set_buttons(enabled=False)
        try:
            #model_name = self.ui.load_model_dropdown.currentText()
            model_path = os.path.join("models", f"{model_name}.pkl")
            loaded_model_data = joblib.load(model_path)
            self.model = loaded_model_data['trained_model']
            self.threshold_params = loaded_model_data['best_threshold_params']        
        except Exception as e:
            pass
        finally:
            self.set_buttons(enabled=True)

    def predict(self):

        # Clear previous output
        self.ui.predict_output.clear()
        self.set_buttons(enabled=False)

        # Setup QThread + Worker
        self.predict_thread = QThread()
        self.predict_worker = Worker(PredictorScorerBPPLines, self.fe, self.model, self.threshold_params, output_widget=self.ui.predict_output)
        self.predict_worker.moveToThread(self.predict_thread)

        # Connect signals:
        # When thread starts, run train_worker
        self.predict_thread.started.connect(self.predict_worker.run)
        # When the worker emits an error signal (using error.emit()), it triggers the self.train_error method
        self.predict_worker.error.connect(self.predict_error)
        # When the worker emits the message signal (using message.emit()), it triggers the self.train_finished method
        self.predict_worker.message.connect(self.predict_finished)
        # When the worker finishes, quit the thread
        self.predict_worker.finished.connect(self.predict_thread.quit)
        # When the thread completes (successfully or with error), it emits the finished signal
        # The deleteLater() method is called, which schedules the thread object for deletion
        # Memory is cleaned up automatically when Qt's event loop processes the deletion
        self.predict_thread.finished.connect(self.predict_thread.deleteLater)        
        # Start thread
        self.predict_thread.start()

    def predict_error(self, err_text):
        # Option 1: append to text box
        self.ui.predict_output.append(f"ERROR\n{err_text}")
        # # Option 2: pop up a message box
        # from PySide6.QtWidgets import QMessageBox
        # QMessageBox.critical(self, "Error", err_text)

        # ensure thread stops
        if hasattr(self, "predict_thread") and self.predict_thread.isRunning():
            self.predict_thread.quit()

        # re-enable button
        self.set_buttons(enabled=True)

    def predict_finished(self, result):
        # Store the PredictorScorerBPPLines object
        self.ps = result
        self.ui.predict_output.append("DONE")
        self.set_buttons(enabled=True)

    def plot_signal(self):

        self.set_buttons(enabled=False)

        try:
            
            # Create your plot here - replace this with your actual plotting function
            plate_id = self.ui.plot_plate_dropdown.currentText()
            segment = self.ui.plot_segment_dropdown.currentText()
            plate = next((p for p in self.dl.list_bpp if p.identifier == plate_id), None)
            if plate is None:
                raise ValueError(f"Plate with identifier '{plate_id}' not found.")

            fig, ax = self.prepare_plot(plate = plate, segment = segment)
            
            # Create a new window to display the plot
            plot_dialog = QDialog(self)
            plot_dialog.setWindowTitle("Empa GUI Plot")
            # Get screen dimensions and fit plot to screen
            screen = QApplication.primaryScreen().geometry()
            plot_dialog.resize(int(screen.width()), int(screen.height()*0.9))
            
            layout = QVBoxLayout()
            canvas = FigureCanvasQTAgg(fig)
            layout.addWidget(canvas)
            plot_dialog.setLayout(layout)
            
            plot_dialog.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create plot: {str(e)}")

        finally:
            self.set_buttons(enabled=True)

    def prepare_plot(
        self,
        plate,
        segment: str,
        plot_every: int = 1,
        include_trigger: bool = False,
    ) -> None:
        """
        TBC
        """
        plot_kwargs = {
            "x": "Time",
            "xlabel": "Time [s]",
            "ylabel": "Amplitude [V]",
        }

        colors = ["purple", "orange", "blue"]

        # Get screen dimensions for maximum figure size
        # screen = QApplication.primaryScreen().geometry()
        # figsize = (screen.width()/100, screen.height()/100)  # Convert pixels to inches (approximate)

        data = plate.dataframe[::plot_every] if segment == "ALL" else plate.segments[tuple(segment.split("_"))][::plot_every]

        channels=(
            plate.data_channels + [plate.trigger_channel]
            if include_trigger
            else plate.data_channels
        )

        fig, ax = plt.subplots(nrows = len(channels), sharex=True)#figsize=figsize)

        for idx, channel in enumerate(channels):
            ax[idx] = data.plot(
                y=channel,
                ax=ax[idx],
                color = colors[idx % len(colors)],
                **plot_kwargs,
            )

        fig.suptitle(f"Segment {segment} from plate {plate.identifier}" if segment != "ALL" else f"Complete signal from plate {plate.identifier}")

        # Mark defective segments with a red vertical band
        if self.ps is not None and hasattr(self.ps, 'pred_defect_seg'):  
            if segment == "ALL":
                pred_defect_seg_plate = [(seg[1],seg[2]) for seg in self.ps.pred_defect_seg if seg[0]==plate.identifier]
                for seg in pred_defect_seg_plate:
                    seg_start, seg_end = plate.segments[seg]["Time"].iloc[[0, -1]]
                    for idx in range(len(channels)):
                        ax[idx].axvspan(seg_start, seg_end, alpha=0.2, color='red')
            else:
                seg_key = tuple(segment.split("_"))
                df_all = self.fe.df_all_locations
                df_seg = df_all[(df_all['plate']==plate.identifier) & (df_all['segment_type']==seg_key[0]) & (df_all['segment_number']==seg_key[1])]
                
                for _, row in df_seg.iterrows():
                    for idx in range(len(channels)):
                        ax[idx].axvspan(row["start_time"], row["end_time"], alpha=row["pred_proba"], color='red')

        return fig, ax

    #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #"""""""""""""""""""""""""""""""""Settings"""""""""""""""""""""""""""""""""""""""""""""""""""""
    #autoencoder settings
    def Autoencoder_set_threshold_click(self):        
        if(self.s_model is None):
            print("Classifier is not defined. Prepare the calssifier and repeat")
            return
        cls_type=type(self.s_model.classifier)
        if(str(cls_type) != "<class 'SHelpers.Autoencoder'>"):
            print("Current classifier is not the Autoencoder type. Make autoencoder model and repeat...")
            return
        self.proc_settings = shlp.ReadSettings(self)
        self.s_model.tocrh_autoencoder_threshold_factor=float(self.proc_settings.get("autoencoder_torch_thershold"))     
        #extract features
        feat=  self.s_model.train_feat
        thresh=self.s_model.tocrh_autoencoder_threshold_factor
        bins=int(self.proc_settings.get("autoencoder_bin_num")) #Autoencoderbins_text_input
        if (self.autoencoder_hist_form_id==""):self.autoencoder_hist_form_id = str(uuid.uuid1())[:5]  
        #%matplotlib qt
        fig=plt.figure(self.autoencoder_hist_form_id)
        ax=fig.subplots()
        labs_tests=shlp.PredictTorchAutoencoder(model=self.s_model.classifier, feat=feat,thresh_fact=thresh,show_hist=True,bins_num=bins,fig_ax=ax)

    #spectrograms settings
    def Tool_Test_Spectrograms_MEL_Botton_Click(self):
        if(self.plates==None):
            print("plates are not loaded...")
            return
        if(self.plates==[]):
            print("plates are not loaded...")
            return
        self.proc_settings = shlp.ReadSettings(self)
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()   
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        signals=[]
        end_=indx_chan+1
        start_=indx_chan
        if(end_==-1): 
            end_ = len(self.plates[indx_plate].segment_sign[indx_segment]) #we taek all signals
            start=0
        SAMPLE_RATE=int(self.plates[indx_plate].sr)        
        for i in range(start_,end_):
            # Define transform            
            waveform=torch.tensor(self.plates[indx_plate].sigments_sign[indx_segment][i])
            if(len(waveform.shape)==1):waveform = waveform[None, :]
            spectrogrym_MEL_nfft=int(self.proc_settings.get("spectrogrym_MEL_nfft"))
            spectrogram = T.Spectrogram(n_fft=spectrogrym_MEL_nfft)
            # Perform transform
            spec = spectrogram(waveform)
            #%matplotlib qt
            fig, axs = plt.subplots(2, 1)
            shlp.plot_waveform(waveform, SAMPLE_RATE, title="Original waveform", ax=axs[0])
            shlp.plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
            fig.tight_layout()
        
        
    def Tool_Show_Spectrograms_MEL_Botton_Click(self):
        if(self.plates==None):
            print("plates are not loaded...")
            return
        if(self.plates==[]):
            print("plates are not loaded...")
            return
        self.proc_settings = shlp.ReadSettings(self)
        p_name = self.ui.plot_plate_dropdown.currentText()
        ch_name=self.ui.Channel_segment_plot.currentText()        
        segment_name=self.ui.plot_segment_dropdown.currentText()   
        indx_plate,indx_chan, indx_segment = shlp.FindPlateInArray(plates=self.plates.copy(),plate_name=p_name,chan_name=ch_name,segm_name=segment_name)
        signals=[]
        end_=indx_chan+1
        start_=indx_chan
        if(end_==-1): 
            end_ = len(self.plates[indx_plate].segment_sign[indx_segment]) #we taek all signals
            start=0
        SAMPLE_RATE=int(self.plates[indx_plate].sr)        
        snip_size=int(self.proc_settings.get("snippet_size"))
        spectrogrym_MEL_nfft=int(self.proc_settings.get("spectrogrym_MEL_nfft"))
        for i in range(start_,end_):
            # Define transform            
            waveform=torch.tensor(self.plates[indx_plate].sigments_sign[indx_segment][i])         
            sls=None
            try: sls=waveform.data[:snip_size]#snip_tensor = torch.take(waveform, torch.tensor([0:snip_size]))#
            except:
                print("waveform is too short to line it into snippets...")
                return            
            if(len(sls.shape)==1):sls=sls[None, :]   
            if(len(waveform.shape)==1):waveform=waveform[None, :]              
            spectrogram = T.Spectrogram(n_fft=spectrogrym_MEL_nfft)
            # Perform transform
            spec = spectrogram(sls)
            ymin=torch.min(waveform)
            ymax=torch.max(waveform)
            #%matplotlib qt            
            #matplotlib.pyplot.vlines(x, ymin, ymax,colors=red)
            fig, axs = plt.subplots(2, 1)
            shlp.plot_waveform(waveform, SAMPLE_RATE, title="Original waveform", ax=axs[0])
            shlp.plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
            axs[0].vlines(snip_size, ymin, ymax,colors="red")
            fig.tight_layout()


    def GetChannels(self):
        #self.proc_settings=shlp.ReadSettings(self)
        CHANNELS_TO_USE=[]
        if( int(self.proc_settings.get("classification_channels_choice_drop_down")) == 0):
            chan_index= int(self.proc_settings.get("chan_from_settings"))-1
            if(chan_index<0):
                CHANNELS_TO_USE=list([0,1,2,3,4,5,6,7]) #all channels are selected
            else:
                CHANNELS_TO_USE=list([chan_index])            
        else:
            CHANNELS_TO_USE = [int(i) for i in self.proc_settings.get("classification_user_channels_text_box").split(",") if i.strip().isdigit()]
        return CHANNELS_TO_USE
    #**********************************************************************************************
    #**********************************************************************************************
    #******************************FILES FOLDERS/DAQ TRACKING**************************************
    #**********************************************************************************************
        
    def Real_Time_Stop_Click(self):     
        
        global EXIT_RT_FLAG
        EXIT_RT_FLAG=True

        if(self.Real_Time_Thread is None):
            pass
        else:        
            self.Real_Time_Thread.do_run = False
            try: 
                if(self.Real_Time_Thread.is_alive()):
                    time.sleep(0.05)
            except: pass
            self.Real_Time_Thread=None
            self.ui.real_time_status_label.setText("Not active")   
            print("Real time is terminated...")
                       
        if(self.RT_SpectrumThread is None):
            pass
        else:
            
            self.RT_SpectrumThread.do_run = False
            try:
                while (self.Real_time_FolderTrackerThread.is_alive()):z=0 
            except: pass
            self.ui.real_time_status_label.setText("Not active")            
            self.RT_SpectrumThread=None
            print("")
            
        
    def Real_Time_Button_Click(self):    
        
        #if(self.s_model is None):
        #    print("Load model first and repeat...")
        #    return
        
        self.proc_settings = shlp.ReadSettings(self)  
        rt_source=self.proc_settings.get("real_time_source")
        channs_indx=[0]
        snip_size=int(self.proc_settings.get("snippet_size"))        
        colors=self.colors_id        
        preprocessing=self.proc_settings.get("classification_preproc_dropdown")
        
        self.rt_data_proc=shlp.DataPreproc(n_fft=int(self.proc_settings.get("spectrogrym_MEL_nfft")),
                                           n_mels=int(self.proc_settings.get("Settings_MEL_num_MELS_2")),
                                           n_mfcc=int(self.proc_settings.get("Settings_nmfcc_num_MFCC_text")),
                                           )
        global EXIT_RT_FLAG
        EXIT_RT_FLAG=False        
        
        #if to show the processed signals graph
        if(bool(self.proc_settings.get("RealT_show_processed_signals_checkbox_3"))==True):                        
            if(self.RT_Figure_if_proc_results_id == ""):
                self.RT_Figure_if_proc_results_id="Processed_results_"+str(uuid.uuid1())[:5]        
            #THIS IS QTFRAME for plotting
            #self.RT_fig_proc_results=shlp.ChartWindow(chart_name=self.RT_Figure_if_proc_results_id)
            #self.RT_fig_proc_results.show()
            #self.RT_fig_proc_results_ax=self.RT_fig_proc_results.Canvas.axes
            # this is matlotlib standard stuff           
            """          
            plt.ion()         
            self.RT_fig_proc_results = plt.figure(self.RT_Figure_if_proc_results_id)
            self.RT_fig_proc_results_ax = self.RT_fig_proc_results.add_subplot(111)
            self.RT_fig_proc_results.show()
            """
            #print("Prepaired processing results graph. Object type: "+str(type(shlp.ChartWindow)))   
            
            #this is pyqt graphics
            #https://pyqtgraph.readthedocs.io/en/latest/getting_started/installation.html
            #self.RT_fig_proc_results = pg.plot(title="self.RT_Figure_if_proc_results_id") #pg.GraphicsLayoutWidget()  # Automatically generates grids with multiple items            
            self.RT_fig_proc_results=RTPlotWidget(parent=self)

            #this is pylab
            """
            self.RT_fig_proc_results = pl.figure(self.RT_Figure_if_proc_results_id)
            self.RT_fig_proc_results.show()
            """

        if(rt_source != "Folder") and (rt_source != "RealTime"):
            return      

        self.show_proc_results_thread=threading.Thread()
        self.show_proc_result_in_progress=False
        self.RT_Frame_Counter=0
                
        if(rt_source == "Folder"):
                        
            rt_path = self.proc_settings.get("real_time_folder_text")                
            if not os.path.exists(rt_path):
                try: os.makedirs(rt_path)
                except: 
                    print("")
                    print("The real time folder cant be created. Check admin rights and repeat...")
                    return            
                    
            self.ui.Real_time_Frame_counter_label.setText(str(0))              
            #self.ExitFilesInFolderFlag=False  
            
            self.Real_Time_Thread = threading.Thread(target=self.FilesFolderTracking1_1,args=[rt_path])
                                                                 #args=(rt_folder,channs_indx,snip_size,preprocessing,s_model,colors,fig,fig_ax))                                                                       
            self.Real_Time_Thread.start()          

        if(rt_source == "RealTime"):
            self.Real_Time_Thread = threading.Thread(target=self.Spectrum_card_tracking)                                                                                                           
            self.Real_Time_Thread.start()  

        """
            #read settings         
            try:
                trig_level=float(self.proc_settings.get("trigger_level"))
                sampling_rate=float(self.proc_settings.get("sampling_rate"))
                pre_trigger_duration=float(self.proc_settings.get("pre_trigger_duration"))
                post_trigger_duration=float(self.proc_settings.get("post_trigger_duration"))
                ampl_per_channel=float(self.proc_settings.get("ampl_per_channel"))
                trig_chan_num=int(self.proc_settings.get("trig_chan_num"))
                show_info=bool(self.proc_settings.get("show_info"))
                show_original_signals=bool(self.proc_settings.get("show_original_signals"))
                show_proc_signals=bool(self.proc_settings.get("show_proc_signals"))
                only_single_shot=bool(self.proc_settings.get("only_single_shot"))                
            except:
                print("non correct values in textboxes in real time settings. Correct and restart...")
                return
            
            #set globals for exit loop in thread
            global EXIT_DAQ_FLAG
            EXIT_DAQ_FLAG=False
                       
            #%matplotlib qt            
            if(show_proc_signals==True):
                plt.ion()
                if(self.RT_Figure_if_proc_results_id == ""):self.RT_Figure_if_proc_results_id="Spectrum_original_signals"+str(uuid.uuid1())[:5]             
                fig_processed = plt.figure(self.RT_Figure_if_proc_results_id)
                fig_ax_processed = fig_processed.add_subplot(111)
                plt.show()
                #fig_processed, fig_ax_processed = plt.subplots()
                #fig_processed.canvas.set_window_title(self.RT_Figure_if_proc_results_id)
            else:
                fig_processed, fig_ax_processed=None,None

            if(show_original_signals==True):
                plt.ion()
                if(self.RT_Figure_if_orig_sgn==""):self.RT_Figure_if_orig_sgn="Spectrum_original_signals"+str(uuid.uuid1())[:5]  
                fig_orig_signals = plt.figure(self.RT_Figure_if_orig_sgn)
                fig_ax_orig_signals = fig_orig_signals.add_subplot(111)
                plt.show()
                #fig_orig_signals.canvas.set_window_title(self.RT_Figure_if_orig_sgn)
            else:
                fig_orig_signals, fig_ax_orig_signals = None, None
                
            
            self.RT_SpectrumThread= threading.Thread(target=self.DAQCard,args=(trig_level,
                                                                               sampling_rate,
                                                                               ampl_per_channel,
                                                                               [0,3,5],
                                                                               trig_chan_num,
                                                                               post_trigger_duration,
                                                                               pre_trigger_duration,
                                                                               snip_size,
                                                                               preprocessing,
                                                                               only_single_shot,
                                                                               s_model,
                                                                               colors,
                                                                               show_original_signals,
                                                                               fig_orig_signals,
                                                                               fig_ax_orig_signals,
                                                                               show_info,                                                                    
                                                                               show_proc_signals,                                                                               
                                                                               fig_processed,
                                                                               fig_ax_processed)) 
            self.RT_SpectrumThread.start()   
            """

        self.ui.real_time_status_label.setText("Active")    
        print("")
        print("Real time started. Waiting for signals...")

    #****************************************************************************************************************************
    #*******************************TRACKING SPECTRUM CARD***********************************************************************

    def Spectrum_card_tracking(self):

        import SHelpers as shlp
        global EXIT_RT_FLAG

        #check for settings
        AMPLITUDE=float(self.proc_settings.get("ampl_per_channel"))
        SAMPLING_RATE=float(self.proc_settings.get("sampling_rate"))
        TRIGGER_LEVEL=float(self.proc_settings.get("trigger_level"))
        PRETRIG_DURATION=float(self.proc_settings.get("pre_trigger_duration"))
        POSTTRIG_DURATION=float(self.proc_settings.get("post_trigger_duration"))
        TRIG_CHAN_NUM=int(self.proc_settings.get("trig_chan_num"))
        CHAN_NAMES=["chan_0","chan_1","chan_2","chan_3","chan_4","chan_5","chan_6","chan_7"]

        delay_between_meas_flag=bool(self.proc_settings.get("RT_impose_delay_between_measurements_checkbox_3"))
        delay_between_meas_value=int(self.proc_settings.get("RT_impose_delay_between_measurements_textbox_5"))        

        signals=[]
        for i in range(0,8):
            signals.append(np.array([]))
                    
        t = threading.current_thread()

        card:spcm.Card
        with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:
            
            #https://github.com/SpectrumInstrumentation/spcm/blob/master/src/examples/01_acquisition/01_acq_single.py
            #https://github.com/SpectrumInstrumentation/spcm/blob/master/src/examples/01_acquisition/02_acq_single_2ch.py

            # do a simple standard setup
            card.card_mode(spcm.SPC_REC_STD_SINGLE)       # single trigger standard mode
            card.timeout(5 * units.s)                     # timeout 5 s
                    
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

            # Channel triggering
            trigger.ch_or_mask0(channels[TRIG_CHAN_NUM].ch_mask())
            trigger.ch_mode(channels[TRIG_CHAN_NUM], spcm.SPC_TM_POS)
            trigger.ch_level0(channels[TRIG_CHAN_NUM], int(TRIGGER_LEVEL) * units.mV, return_unit=units.mV) # trigger level - float(TRIGGER_LEVEL)
                        
            data_transfer = spcm.DataTransfer(card)
            data_transfer.duration((PRETRIG_DURATION+POSTTRIG_DURATION)*units.ms, post_trigger_duration=POSTTRIG_DURATION*units.ms)

            print("")
            print("DAQ card info:")            
            print("Card type: "+str(card.card_type))
            print("Card family:"+str(card.family()))
            print("Chans.num.: "+str(card.num_channels()))
            print("Trig.active chan.:"+str(card.active_channels()))
            print("Trigger lev.: "+str(int(TRIGGER_LEVEL)))
            print("Samp.rate(MHz): "+str(SAMPLING_RATE))
            print("Pretrigger(ms): "+str(PRETRIG_DURATION))
            print("Pretrigger(samp.points): "+str(PRETRIG_DURATION*SAMPLING_RATE*1e3))
            print("Posttrigger(ms): "+str(POSTTRIG_DURATION))
            print("Pretrigger(samp.points): "+str(POSTTRIG_DURATION*SAMPLING_RATE*1e3))            
            print("")

            self.RT_Frame_Counter=1

            while(getattr(t, "do_run", True)):

                try:
                    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)              
                except:
                    continue

                if(EXIT_RT_FLAG==True):
                    print("Data acquisition is terminating...")
                    break      

                data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)                                                     
                    
                sign_empty=False

                with warnings.catch_warnings():

                    warnings.filterwarnings("ignore")#, category=DeprecationWarning)
                    signals[0]=channels[0].convert_data(data_transfer.buffer[channels[0], :], units.V)                
                    signals[1]=channels[1].convert_data(data_transfer.buffer[channels[1], :], units.V)                
                    signals[2]=channels[2].convert_data(data_transfer.buffer[channels[2], :], units.V)                
                    signals[3]=channels[3].convert_data(data_transfer.buffer[channels[3], :], units.V)                
                    signals[4]=channels[4].convert_data(data_transfer.buffer[channels[4], :], units.V)                
                    signals[5]=channels[5].convert_data(data_transfer.buffer[channels[5], :], units.V)                
                    signals[6]=channels[6].convert_data(data_transfer.buffer[channels[6], :], units.V)                
                    signals[7]=channels[7].convert_data(data_transfer.buffer[channels[7], :], units.V)
                    
                    signals[0]=np.asarray(signals[0])
                    if(len(signals[0])==0): sign_empty=True
                    signals[1]=np.asarray(signals[1])
                    if(len(signals[1])==0): sign_empty=True
                    signals[2]=np.asarray(signals[2])
                    if(len(signals[2])==0): sign_empty=True
                    signals[3]=np.asarray(signals[3])
                    if(len(signals[3])==0): sign_empty=True
                    signals[4]=np.asarray(signals[4])
                    if(len(signals[4])==0): sign_empty=True
                    signals[5]=np.asarray(signals[5])
                    if(len(signals[5])==0): sign_empty=True
                    signals[6]=np.asarray(signals[6])
                    if(len(signals[6])==0): sign_empty=True
                    signals[7]=np.asarray(signals[7])
                    if(len(signals[7])==0): sign_empty=True
                
                if(EXIT_RT_FLAG==True):
                    print("Data acquisition is terminating...")
                    break               
                
                        
                if(sign_empty):continue
                
                print("")
                print("********************************************************************")
                print("***********************MEASUREMENT "+str(self.RT_Frame_Counter)+"********************")        
                self.RT_Frame_Counter=self.RT_Frame_Counter+1
                                
                try:
                    rt_plate=shlp.SPlate()
                    rt_plate.raw_signals=signals
                    rt_plate.chans_names=CHAN_NAMES
                    rt_plate.time=np.arange(0,len(signals[0]))*(1.0/(SAMPLING_RATE*1e6))
                    rt_plate.get_segments(ref_chan_name=TRIG_CHAN_NUM,threshold="automatic")#TRIGGER_LEVEL)
                    if(rt_plate.sigments_sign==[]): 
                        print("No segments were detected...")                        
                    else: self.Process_RT_Data(rt_plate)
                except Exception as ex:
                    print("Impossible to process data. Exception raised: "+str(ex))

                print("********************************************************************")
                print("")
                print("")


                if(bool(self.proc_settings.get("only_single_shot")) ==True):
                    break

                if(EXIT_RT_FLAG==True):
                    print("Data acquisition is terminating...")
                    break      

                if(delay_between_meas_flag): time.sleep( delay_between_meas_value / 1000.0 ) #ms to s                

            card.stop()
            card.reset()
            card.close()
            self.Real_Time_Stop_Click()

    #****************************************************************************************************************************
    #*******************************TRACKING FOLDER******************************************************************************

    def FilesFolderTracking1_1(self,rt_folder):     

        import SHelpers as shlp

        txtfiles = []    
        self.RT_Frame_Counter=0
        global EXIT_RT_FLAG
                                       
        t = threading.current_thread()
        while(getattr(t, "do_run", True)):
            #check for exit
            if(EXIT_RT_FLAG==True):
                print("Data acquisition is terminating...")
                break            
            txtfiles=[]
            for x in os.listdir(rt_folder):#for file in glob.glob("*.txt"):
                #check for exit
                if(EXIT_RT_FLAG==True):
                    print("Data acquisition is terminating...")
                    break                
                #txtfiles.append(file)                
                if x.endswith(".txt"):
                    txtfiles.append(x)
                    
            if(len(txtfiles)==0):
                continue

            for l in range(0,len(txtfiles)):
                #check for exit
                if(EXIT_RT_FLAG==True):
                    print("Data acquisition is terminating...")
                    break     
                txt_path=rt_folder+"\\"+txtfiles[l]                
                #open plate - first check 
                f_base=os.path.splitext(os.path.basename(txtfiles[l]))[0]                
                f_bin=""
                for x in os.listdir(rt_folder):
                    if x.endswith(".bin"):
                        x_base=os.path.splitext(os.path.basename(x))[0]          
                        if(str(x_base) in str(f_base)):
                            f_bin=x
                if(f_bin==""):
                    continue
                bin_path=rt_folder+"\\"+f_bin     
                #check if files are occupied or not                
                try:
                    os.rename(bin_path,bin_path)
                    os.rename(txt_path,txt_path)
                except:                     
                    continue                                

                print("")
                print("********************************************************************")
                print("***********************MEASUREMENT " +str(self.RT_Frame_Counter)+"*************************")                         
                #open/processing                
                plates=[]
                try: 
                    plates= shlp.OpenDataFromFolder(ONLY_SINGLE_FILE=True,SINGLE_FILE_PATH_BIN=bin_path,SINGLE_FILE_PATH_TXT=txt_path)
                    self.Process_RT_Data(plates[0])
                except Exception as ex:                     
                    print("Impossible to open files. Exception raised: "+str(ex))                     
                print("********************************************************************")
                print("")
                print("")
                self.RT_Frame_Counter=self.RT_Frame_Counter+1
                #delete files after processing if needed
                if(bool(self.proc_settings.get("RealT_filse_folders_delete_files_checkbox"))==True):
                    try: os.remove(txt_path)                    
                    except: pass
                    try: os.remove(bin_path)           
                    except: pass

                if(bool(self.proc_settings.get("only_single_shot")) ==True):
                    break

                if(bool(self.proc_settings.get("RT_impose_delay_between_measurements_checkbox_3")) ==True):
                    time_to_wait=int(self.proc_settings.get("RT_impose_delay_between_measurements_textbox_5"))/1000
                    time.sleep(time_to_wait)
                
            txtfiles=[] #clean up
        #finally we press virtuall stop vbutton
        self.Real_Time_Stop_Click()


    #****************************************************************************************************************************
    #****************************************************************************************************************************
    #****************************************************************************************************************************
        
    def Process_RT_Data(self,plate):

        import SHelpers as shlp
        #*********************************processing***************************         
        preprocessing = self.proc_settings.get("classification_preproc_dropdown")      
        snip_size=int(self.proc_settings.get("MAIN_classification_snippet_size_text"))   
                   
        display_time=[]
        proc_time_total=[]
        snips_num=0        
        labels_in_segment=[]

        if( int(self.proc_settings.get("classification_channels_choice_drop_down")) == 0):
            chan_index= int(self.proc_settings.get("chan_from_settings"))-1
            if(chan_index<0):
                CHANNELS_TO_USE=list([0,1,2,3,4,5,6,7]) #all channels are selected
            else:
                CHANNELS_TO_USE=list([chan_index])
            
        else:
            CHANNELS_TO_USE = [int(i) for i in self.proc_settings.get("classification_user_channels_text_box").split(",") if i.strip().isdigit()]
                     
        sgn_len= len(plate.sigments_sign)
                
        proc_time_total.append(time.time())
        #*********************************************************************************************
        for seg_i in range(0,sgn_len):
            #take segment
            segm_signal=plate.sigments_sign[seg_i]            
            if(len(segm_signal)==0):
                continue
            #cut snippets
            snips_len=0
            snippets=self.rt_data_proc.SplitEntireSignalIntoSnippets(signal=segm_signal,channs_indx=CHANNELS_TO_USE,torch_tensor=False, snip_size = snip_size,preproc_type = preprocessing)
            if snippets is None:                
                labels_in_segment.append(list())
            else:
                if(self.s_model is not None): 
                    if(len(snippets.shape)!=0):
                        shp_snip=np.shape(np.asarray(snippets))
                        proc_labels=self.s_model.np_predict(np.asarray(snippets))
                        labels_in_segment.append(proc_labels)
                    else:
                        labels_in_segment.append(list())
                else:
                    labels_in_segment.append(list())
                snips_len=len(snippets)
            snips_num=snips_num+snips_len        
            shlp.print_progress_bar(seg_i+1, sgn_len, "segments proc.progress")    
        #**********************************************************************************************
        proc_time_total.append(time.time())

        #SHOW RESULTS        
        skipped_results = False
        if(bool(self.proc_settings.get("RealT_show_processed_signals_checkbox_3"))):                        
            try:                       
                #multiprocessing.Process
                if(self.show_proc_result_in_progress==False):#self.show_proc_results_thread.is_alive()==False):
                    self.show_proc_result_in_progress==True
                    self.curMeas_num=self.RT_Frame_Counter #we fix here the current frame counter number that corresponds to this measurements                    
                    self.show_proc_results_thread = threading.Thread (target=self.ShowProcResults_REAL_TIME,args=(plate,labels_in_segment))                    
                    self.show_proc_results_thread.start()
                    display_time.append(0)
                    display_time.append(1)
                else: 
                    skipped_results=True
                
            except Exception as Ex: print("Cant display processed data. Exception: "+str(Ex))

        if(bool(self.proc_settings.get("show_info"))==True):
            now = datetime.datetime.now()
            try:
                print("")
                print("GENERAL INFO: ")            
                print("Data received: "+str(now))
                print("Data length/chan.: "+str(len(plate.raw_signals[0])))
                print("Processed channels: "+str(CHANNELS_TO_USE))
                print("Segments num.: "+str(sgn_len))
                print("Snippets num.: "+str(snips_num))       
                if(snips_num==0): print("ATTENTION: segments are to short to create snippets, analysis in bot possible...")
                print("Total proc.time(s): "+str(proc_time_total[1]-proc_time_total[0]))
                if(snips_num!=0): 
                    try: print("Proc.-snippet/s: "+str((proc_time_total[1]-proc_time_total[0])/snips_num))
                    except: pass
                #if(show_results): 
                    #if(skipped_results==False): print("Display time of results (s): "+str(display_time[1]-display_time[0]))
                    #else: print("Results are not shown")
            except Exception as ex:
                print("Cant display data.Exception: "+str(ex))

            """
                shlp.ShowAllSingleSegmentsWithLabels(self.RT_fig_proc_results, 
                                                     plate,
                                                     colors_code=self.colors_id,
                                                     indx_chan=CHANNELS_TO_USE,
                                                     aplpha=0.1,
                                                     show_labels=False, #this is for ground truth labels
                                                     points_num_limit_check=bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                     points_num_limit=int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox")),
                                                     show_proc_labels=False, #this is for processed labels
                                                     proc_labels_snip_size=int(self.proc_settings.get("snippet_size")),
                                                     proc_labels_color_scheme="all",
                                                     proc_labels=None,
                                                     )
            """
            

            """
            preproc_t_strart=time.time()
            snippets=self.rt_data_proc.SplitEntireSignalIntoSnippets(signal=segm_signal,
                                                                         channs_indx=CHANNELS_TO_USE,
                                                                         torch_tensor=False,
                                                                         snip_size = snip_size,
                                                                         preproc_type = preprocessing
                                                                         )
            preproc_t_end=time.time()
            prerpoc_time.append(preproc_t_strart-preproc_t_end)

            
                        
                        start_proc=time.time()
                        labels=s_model.np_predict(np.asarray(snippets))
                        snip_number_processed+=len(labels)
                        end_proc=time.time()                        
                        proc_time.append(end_proc-start_proc)
                        
                        all_labels.append(labels) 
                        all_segm_sign.append(segm_signal)
                    #show everything in the chart   
                    if(SHOW_PROC_RESULTS==True):
                        try:                        
                            viz_start_t=time.time()
                            shlp.ShowResultsInFigure_AllSegmentsInRow(all_segm_sign,
                                                                      all_labels,
                                                                      CHANNELS_TO_USE,
                                                                      snip_size,
                                                                      colorscheme,
                                                                      SHOW_PROC_RESULTS_FIG,
                                                                      SHOW_PROC_RESULTS_AX,
                                                                      None)
                            viz_end_t=time.time()                        
                        except:
                            print("Figure is unavailable.")
                            
                    #**********************************************************************
                    
                    if(DATA_PRINT==True):                    
                        now = datetime.datetime.now()
                        signals_counter+=1
                        trig_time_st=int((PRETRIG_DURATION/1000)*SAMPLING_RATE*1000000)
                        proc_mean_t=-1                        
                        try: 
                            if(len(proc_time)!=0):proc_mean_t=sum(proc_time) / float(len(proc_time))
                        except: pass
                        print("")
                        print("******************************************")
                        print("Data received: "+str(now))
                        print("Data length/chan.: "+str(len(signals[0])))
                        print("Chans.num.: "+str(len(signals)))
                        print("Frame num.:"+str(signals_counter))
                        print("Trig. chan.: "+str(TRIGGER_CHANNEL))     
                        print("Trig. chan. max/min: "+str(np.max(signals[TRIGGER_CHANNEL]))+"/"+str(np.min(signals[TRIGGER_CHANNEL])))     
                        print("Trig.timestamp: "+str(trig_time_st/1000)+" ms")
                        print("Trig. val.:"+str(signals[TRIGGER_CHANNEL][trig_time_st]))
                        if(len(prerpoc_time)>0): print("Average preproc. time(s): "+str(float(sum(prerpoc_time) / float(len(prerpoc_time)))))
                        if(float(proc_mean_t) >= 0): print("Average proc.time(s): "+str(proc_mean_t))
                        print("Number of proc.snips.: "+str(snip_number_processed))
                        try: print("Vizualization time: "+str(viz_end_t-viz_start_t))                            
                        except: pass
                        
                        print("******************************************")
                        print("")
                    
                    # Plot the acquired data
                    if(SHOW_ORIGIN_SIGNALS==True) and (SHOW_ORIGIN_SIGNALS_FIG is not None) and (SHOW_ORIGIN_SIGNALS_AX is not None):            
                        time_data_s = data_transfer.time_data()
                        #fig, ax = plt.subplots()                                                
                        #print(channel0)
                        #print("\tMinimum: {:.3~P}".format(np.min(unit_data_V)))
                        #print("\tMaximum: {:.3~P}".format(np.max(unit_data_V)))                    
                        SHOW_ORIGIN_SIGNALS_AX.clear()
                        for ws in range(0,len(CHANNELS_TO_USE)):
                            chan_index=CHANNELS_TO_USE[ws]
                            SHOW_ORIGIN_SIGNALS_AX.plot(time_data_s, signals[chan_index], label=("channel "+str(chan_index)))
                        SHOW_ORIGIN_SIGNALS_AX.yaxis.set_units(units.mV)
                        SHOW_ORIGIN_SIGNALS_AX.xaxis.set_units(units.us)
                        SHOW_ORIGIN_SIGNALS_AX.axvline(0, color='k', linestyle='--', label='Trigger')
                        SHOW_ORIGIN_SIGNALS_AX.legend()
                        SHOW_ORIGIN_SIGNALS_FIG.canvas.draw()
                        SHOW_ORIGIN_SIGNALS_FIG.canvas.flush_events()
                    
                    if(EXIT_DAQ_FLAG==True):
                        print("Data acquisition is terminating...")
                        break

                    if(ONLY_SINGLE_SHOT==True):
                        break

            """#end of processing
               

    def ShowProcResults_REAL_TIME(self,plate,labels_in_segment):

        import SHelpers as shlp

        warnings.filterwarnings( "ignore")

        chan_num=self.GetChannels()
        snip_size=int(self.proc_settings.get("MAIN_classification_snippet_size_text"))   
        impose_addit_delay_showing_results=bool(self.proc_settings.get("GUI_impose_delay_checkbox_2"))
        addit_delay_value=int(self.proc_settings.get("GUI_impose_measurements_delay_value_textbox_2"))
        points_num_limit_check=bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox"))
        points_num_limit=int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox"))
        show_labels=bool(self.proc_settings.get("GUI_show_labels_checkbox"))
        mark_segm_borders=bool(self.proc_settings.get("GUI_mark_segments_checkbox"))
        #show the labels
        show_results_scheme=str(self.proc_settings.get("Show_results_color_scheme_drop_down_1"))
        
        if(isinstance(self.RT_fig_proc_results,plt.Figure)):                     
            self.RT_fig_proc_results.clf()
            fig_ax= self.RT_fig_proc_results.add_subplot(111)
            draw_window_type="plt_figure"   
        if(isinstance(self.RT_fig_proc_results,RTPlotWidget)):#,pg.widgets.PlotWidget.PlotWidget)):
            #https://pyqtgraph.readthedocs.io/en/latest/getting_started/plotting.html            
            self.RT_fig_proc_results.clear()
            draw_window_type="pg_plot"

        #start_display = pg.time.ptime.time()pg.time.ptime.time()#pg.ptime.time()
        #******************************************************************************************************************************
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
                    #sample_stamp[-1].append(len(sfg1)+sample_stamp[-1][len(sample_stamp[-1])-1])            
                sample_stamp[-1].append(len(full_sign[-1]))
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
                    self.RT_fig_proc_results.plot(points_steps,reduced_signal)

                else:                 
                    self.RT_fig_proc_results.plot(full_sign[m])
        else:                
            for m in range(0,len(full_sign)):
                self.RT_fig_proc_results.plot(full_sign[m])

        """
        if(show_labels==True):

            unique_labels=plate.GetUniqueLabelsList()
            labels_tags=[]
            sectors=[]

            for kkj in range(0,len(chan_num)):
                cnt=0            
                for k in range(0,len(plate.sigments_labels)):                
                    if (k!=0): cnt=cnt+len(plate.sigments_sign[k-1][kkj])
                    for j in range(0,len(plate.sigments_labels[k])):     
                        curs_start=int((plate.sigments_labels[k][j][0]+cnt))
                        curs_end=int((plate.sigments_labels[k][j][1]+cnt))
                        label=plate.sigments_labels[k][j][2]                
                        c_ind=unique_labels.index(label)
                        if(c_ind>len(self.colors_id)-1): c_ind=len(self.colors_id)-1
                        sector_=plt.axvspan(curs_start, curs_end, facecolor=self.colors_id[c_ind], alpha=0.1)    
                        if(label not in labels_tags):
                           labels_tags.append(label)
                           sectors.append(sector_)
        """

        min_v=np.min(np.asarray(full_sign),axis=None)
        max_v=np.max(np.asarray(full_sign),axis=None)

        if(mark_segm_borders==True):           
                if(isinstance(self.RT_fig_proc_results,RTPlotWidget)):
                    #fig_ax.plot(pg.InfiniteLine(sample_stamp[0][b]))                    
                    #line_=pg.InfiniteLine(pos=cur_pos,angle=90,bounds=[min_v,max_v])
                    self.RT_fig_proc_results.vlines(sample_stamp[0][:],min_v,max_v)#self.RT_fig_proc_results.addItem(pg.InfiniteLine(pos=cur_pos,angle=90))#,bounds=[min_v,max_v]))
                else:
                    for b in range(0,len(sample_stamp[0])):
                        self.RT_fig_proc_results.vlines(sample_stamp[0][b],ymin=min_v,ymax=max_v,colors="black",linestyles="solid")

        if (labels_in_segment is not None) and (len(labels_in_segment)!=0):                           
            cnt = 0        
            chan_n=chan_num[0]
            pnts=[]
            x=[]
            y=[]
            unique_l=[]
            proceed=True
            for p in range(0,len(labels_in_segment)):                
                if(p>0): cnt=cnt+len(plate.sigments_sign[p-1][chan_n])   
                if(labels_in_segment[p] is None): #or (not labels_in_segment[p]):
                    #proceed=False
                    #break
                    x.append(cnt+snip_size/2)
                    y.append(-1)
                else:
                    for k in range(0,len(labels_in_segment[p])):
                        cur_pnt=[]
                        st_ = k * snip_size + cnt
                        en_ = st_ + snip_size
                        if(labels_in_segment[p][k] not in unique_l): unique_l.append(labels_in_segment[p][k])
                        x.append(st_+(st_-en_)/2)
                        y.append(labels_in_segment[p][k])
                        st_= en_
                                        
            #show in scatter plot
            if(isinstance(self.RT_fig_proc_results,RTPlotWidget)):                 
                self.RT_fig_proc_results.plot_results(x,y,unique_l,self.colors_id)
        #end_display = pg.time.ptime.time()
        #******************************************************************************************************************************
        text_to_add="Measurement "+str(self.curMeas_num)
        if(isinstance(self.RT_fig_proc_results,plt.Figure)):
            if(bool(self.proc_settings.get("GUI_settitle_figure_checkbox_2"))): self.RT_fig_proc_resultsg.set_title(text_to_add)
            self.RT_fig_proc_resultsg.legend(sectors,labels_tags)
            self.RT_fig_proc_results.canvas.draw_idle() #draw_idle() #draw()    
            self.RT_fig_proc_results.show()
            self.RT_fig_proc_results.canvas.flush_events()        
        elif(isinstance(self.RT_fig_proc_results,RTPlotWidget)):        
            if(bool(self.proc_settings.get("GUI_settitle_figure_checkbox_2"))):                                
                self.RT_fig_proc_results.label.setText(text_to_add)
        if(bool(self.proc_settings.get("GUI_impose_delay_checkbox_2"))==True):
            time.sleep(float(self.proc_settings.get("RT_impose_delay_between_measurements_textbox_5"))/1000)
        #******************************************************************************************************************************        
        #at the and
        self.show_proc_result_in_progress==False

    #this are the old versions below, now substituted by the newer ones
    def ShowProcessedResults(self,plate,labels_in_segment):

        snip_size=int(self.proc_settings.get("MAIN_classification_snippet_size_text"))   
        impose_addit_delay_showing_results=bool(self.proc_settings.get("GUI_impose_delay_checkbox_2"))
        addit_delay_value=int(self.proc_settings.get("GUI_impose_measurements_delay_value_textbox_2"))
        
        if( int(self.proc_settings.get("classification_channels_choice_drop_down")) == 0):
            chan_index= int(self.proc_settings.get("chan_from_settings"))-1
            if(chan_index<0):
                CHANNELS_TO_USE=list([0,1,2,3,4,5,6,7]) #all channels are selected
            else:
                CHANNELS_TO_USE=list([chan_index])            
        else:
            CHANNELS_TO_USE = [int(i) for i in self.proc_settings.get("classification_user_channels_text_box").split(",") if i.strip().isdigit()]
        
        shlp.ShowAllSingleSegmentsWithLabels(self.RT_fig_proc_results,
                                             plate,
                                             colors_code = self.colors_id,
                                             indx_chan = CHANNELS_TO_USE,aplpha=0.1,
                                             show_labels=False,
                                             points_num_limit_check=bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                             points_num_limit=int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox")),
                                             show_proc_labels=True, #this is for processed labels
                                             proc_labels_snip_size=snip_size,
                                             proc_labels_color_scheme=self.proc_settings.get("Show_results_color_scheme_drop_down_1"),
                                             proc_labels=labels_in_segment,
                                             mark_segm_borders=bool(self.proc_settings.get("GUI_mark_segments_checkbox")),
                                             show_wait_time=impose_addit_delay_showing_results,
                                             show_wait_time_value=addit_delay_value/1000,#delay of plotting in ms
                                             add_title_text=bool(self.proc_settings.get("GUI_settitle_figure_checkbox_2")),
                                             title_text="Measurement "+str(self.RT_Frame_Counter),
                                            )

        
        self.show_proc_result_in_progress==False 
        
        """
        if(isinstance(self.RT_fig_proc_results,plt.Figure)):          
            self.RT_fig_proc_results.canvas.draw_idle() #draw_idle() #draw()                
            self.RT_fig_proc_results.canvas.flush_events()
        else:                       
            self.RT_fig_proc_results.Canvas.fig.canvas.draw_idle()#draw()
            self.RT_fig_proc_results.Canvas.fig.canvas.flush_events()
        """

        """
        self.show_proc_results_thread = threading.Thread (target=shlp.ShowAllSingleSegmentsWithLabels,args=(self.RT_fig_proc_results,plate),
                                                                                                   kwargs={"colors_code" : self.colors_id,
                                                                                                           "indx_chan" : CHANNELS_TO_USE,
                                                                                                            "aplpha":0.1,
                                                                                                            "show_labels":False, #this is for ground truth labels
                                                                                                            "points_num_limit_check":bool(self.proc_settings.get("GUI_show_results_points_number_limit_checkbox")),
                                                                                                            "points_num_limit":int(self.proc_settings.get("GUI_show_results_points_number_limit_textbox")),
                                                                                                            "show_proc_labels":True, #this is for processed labels
                                                                                                            "proc_labels_snip_size":snip_size,
                                                                                                            "proc_labels_color_scheme":self.proc_settings.get("Show_results_color_scheme_drop_down_1"),
                                                                                                            "proc_labels":labels_in_segment,            
                                                                                                         })
                    """

    #this is an old version
    def FilesFolderTracking1(self,rt_folder,chan_indx,snip_size,preprocessing,s_model,colorscheme,fig,fig_ax): 

        import SHelpers as shlp
        #import matplotlib
        #matplotlib.use('agg')
        #%matplotlib qt    
        
        Counter=0
        txtfiles = []          
        proc_data=shlp.DataPreproc(n_fft=int(self.proc_settings.get("spectrogrym_MEL_nfft")),
                                   n_mels=int(self.proc_settings.get("Settings_MEL_num_MELS_2")),
                                   n_mfcc=int(self.proc_settings.get("Settings_nmfcc_num_MFCC_text")),
                                  )
        
        Proceed_to_proc=False
        plates=[]
        data_proc=shlp.DataPreproc(n_fft=int(self.proc_settings.get("spectrogrym_MEL_nfft")),
                                   n_mels=int(self.proc_settings.get("Settings_MEL_num_MELS_2")),
                                   n_mfcc=int(self.proc_settings.get("Settings_nmfcc_num_MFCC_text")),
                                   )
        all_labels=[]
        all_segm_sign=[]

        start_t=0
        end_t=0

        viz_start_t=0
        viz_end_t=0

        t = threading.current_thread()
        while(getattr(t, "do_run", True)):
            #check for exit
            if(self.ExitFilesInFolderFlag==True):
                print("Exiting file tracker")
                break

            txtfiles=[]
            for x in os.listdir(rt_folder):#for file in glob.glob("*.txt"):
                #check for exit
                if(self.ExitFilesInFolderFlag==True):
                    print("Exiting file tracker")
                    break                
                #txtfiles.append(file)                
                if x.endswith(".txt"):
                    txtfiles.append(x)
                    
            if(len(txtfiles)==0):
                continue

            for l in range(0,len(txtfiles)):
                #check for exit
                if(self.ExitFilesInFolderFlag==True):
                    print("Exiting file tracker")
                    break     
                txt_path=rt_folder+"\\"+txtfiles[l]                
                #open plate - first check 
                f_base=os.path.splitext(os.path.basename(txtfiles[l]))[0]                
                f_bin=""
                for x in os.listdir(rt_folder):
                    if x.endswith(".bin"):
                        x_base=os.path.splitext(os.path.basename(x))[0]          
                        if(str(x_base) in str(f_base)):
                            f_bin=x
                if(f_bin==""):
                    continue
                bin_path=rt_folder+"\\"+f_bin     
                #check if files are occupied or not                
                try:
                    os.rename(bin_path,bin_path)
                    os.rename(txt_path,txt_path)
                except:                     
                    continue

                #open/processing                
                plates=[]
                try: plates= shlp.OpenDataFromFolder(ONLY_SINGLE_FILE=True,SINGLE_FILE_PATH_BIN=bin_path,SINGLE_FILE_PATH_TXT=txt_path)
                except Exception as ex: 
                    print("")
                    print("Impossible to open data. Exception raised: "+str(ex))
                    
                #preprocessing
                if(len(plates)==0):Proceed_to_proc=False
                elif(plates is None): Proceed_to_proc=False
                elif(plates[0] is None): Proceed_to_proc=False
                else: Proceed_to_proc=True
                if(Proceed_to_proc==True):
                    sgn_len = len(plates[0].sigments_sign)
                    all_labels=[]
                    all_segm_sign=[]
                    for seg_i in range(0,sgn_len):
                        #take segment
                        segm_signal=plates[0].sigments_sign[seg_i]
                        snippets=data_proc.SplitEntireSignalIntoSnippets(signal=segm_signal,
                                                                         channs_indx=chan_indx,
                                                                         torch_tensor=False,
                                                                         snip_size = snip_size,
                                                                         preproc_type = preprocessing
                                                                         )
                        start_t=time.time()
                        labels=s_model.np_predict(np.asarray(snippets))
                        end_t=time.time()
                        all_labels.append(labels) 
                        all_segm_sign.append(segm_signal)
                    #show everything in the chart   
                    try:
                        #check for exit
                        if(self.ExitFilesInFolderFlag==True):
                            print("Exiting file tracker")
                            break
                        viz_start_t=time.time()
                        shlp.ShowResultsInFigure_AllSegmentsInRow(all_segm_sign,all_labels,chan_indx,snip_size,colorscheme,fig,fig_ax,None)
                        viz_end_t=time.time()
                    except:
                        print("Figure is anavailable.")
                #end of processing

                #clean files
                try:  os.remove(txt_path)
                except: pass
                try:  os.remove(bin_path)                    
                except: pass  #not_deleted_files.append(rt_folder+"\\"+txtfiles[l])
                Counter+=1  
                print("")
                print("Measurements number - "+str(Counter))
                print("Processing time - "+str(end_t-start_t))
                print("Vizualization time - "+str(viz_end_t-viz_start_t))
            #self.ui.Real_time_Frame_counter_label.setText(str(self.rt_FrameCounter))       
            
            #check for exit
            if(self.ExitFilesInFolderFlag==True):
                print("Exiting file tracker")
                break

    #*****************************************************************************************************
    #*****************************************************************************************************
    #*****************SPECTRUM DAQ************************************************************************

    def DAQCard(self,
                trigger_level,#=2200, # mV
                sampling_rate,#=0.1, #MHz
                amplitude,#=5000, # mV
                channels_to_use,#[0,3,5],                
                trigger_channel,#=0,
                post_trig_duarat,#=1, # ms
                pretrig_duaration,#=0.1,#ms
                snip_size,#500
                preprocessing,
                only_single_shot,
                s_model,       
                colorscheme,#this is to show
                show_origin_signals,#=False,
                show_origin_signals_fig,
                show_origin_signals_ax,#=None,
                info_print,#=True,
                #processing
                show_proc_signals,                                              
                fig_show_proc_signals,
                fig_ax_proc_signals,
            ):

        import SHelpers as shlp
        
        # EXIT_FLAG=exit_flag#False
        TRIGGER_LEVEL=trigger_level#2200 # mV
        SAMPLING_RATE=sampling_rate#0.1 #MHz
        AMPLITUDE=amplitude#5000 # mV
        CHANNELS_TO_USE=channels_to_use#[0,3,5]
        TRIGGER_CHANNEL=trigger_channel#0
        POSTTRIG_DURATION=post_trig_duarat #1 # ms
        PRETRIG_DURATION=pretrig_duaration#0.1#ms
        SHOW_ORIGIN_SIGNALS=show_origin_signals#False
        SHOW_ORIGIN_SIGNALS_FIG=show_origin_signals_fig
        SHOW_ORIGIN_SIGNALS_AX=show_origin_signals_ax#None
        DATA_PRINT=info_print#True
        SHOW_PROC_RESULTS= show_proc_signals
        SHOW_PROC_RESULTS_FIG=fig_show_proc_signals
        SHOW_PROC_RESULTS_AX=fig_ax_proc_signals
        ONLY_SINGLE_SHOT=only_single_shot
        #https://spectruminstrumentation.github.io/spcm/spcm.html#Trigger
        #def SPECTRUM_DAQ():

        print("")
        print("Spectrum card real time settings: ")
        print("Trigger level(mV): " +str(TRIGGER_LEVEL))
        print("Sampling rate(MHz): " +str(SAMPLING_RATE))
        print("Amplitude/channel(mV): " +str(AMPLITUDE))
        print("Channels to use: " +str(CHANNELS_TO_USE))
        print("Trigger channel: " +str(TRIGGER_CHANNEL))
        print("Trigger level(mV): " +str(TRIGGER_LEVEL))
        print("Post trigger duration(ms): " +str(POSTTRIG_DURATION))
        print("Pre trigger duration(ms): " +str(PRETRIG_DURATION))
        print("SHow original signals: " +str(SHOW_ORIGIN_SIGNALS))
        print("Signal info output: " +str(DATA_PRINT))
        print("Show preprocessing results: " +str(SHOW_PROC_RESULTS))
        print("")
        print("LAUNCH REAL TIME...")
        print("")
        
        signals_counter=0
        signals=[]
        for i in range(0,8):
            signals.append([])        
        plate=None
        data_proc=shlp.DataPreproc(n_fft=int(self.proc_settings.get("spectrogrym_MEL_nfft")),
                                   n_mels=int(self.proc_settings.get("Settings_MEL_num_MELS_2")),
                                   n_mfcc=int(self.proc_settings.get("Settings_nmfcc_num_MFCC_text")),
                                   )
        all_labels=[]
        all_segm_sign=[]
        proc_time=[]
        viz_time=[]
        
        card:spcm.Card
        #prepare the figure
        #with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
        # with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
        # with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
          
        with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type
            ##threading termination - https://stackoverflow.com/questions/18018033/how-to-stop-a-looping-thread-in-python
            t = threading.current_thread()
            while(getattr(t, "do_run", True)):   
                if(EXIT_DAQ_FLAG==True):
                    print("Data acquisition is terminating...")
                    break
                
                # do a simple standard setup
                card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
                card.timeout(5 * units.s)                     # timeout 5 s
            
                trigger = spcm.Trigger(card)
                trigger.or_mask(spcm.SPC_TMASK_NONE)       # trigger set to none #software
                trigger.and_mask(spcm.SPC_TMASK_NONE)      # no AND mask
            
                clock = spcm.Clock(card)
                clock.mode(spcm.SPC_CM_INTPLL)            # clock mode internal PLL
                clock.sample_rate(SAMPLING_RATE * units.MHz, return_unit=units.MHz)
            
                # setup the channels
                #channel0
                channel0, = spcm.Channels(card, card_enable=spcm.CHANNEL0) # enable channel 0
                channel0.amp(AMPLITUDE * units.mV)
                channel0.offset(0 * units.mV)
                channel0.termination(0)
                #channel1
                channel1, = spcm.Channels(card, card_enable=spcm.CHANNEL1) # enable channel 1
                channel1.amp(AMPLITUDE * units.mV)
                channel1.offset(0 * units.mV)
                channel1.termination(0)
                #channel2
                channel2, = spcm.Channels(card, card_enable=spcm.CHANNEL2) # enable channel 1
                channel2.amp(AMPLITUDE * units.mV)
                channel2.offset(0 * units.mV)
                channel2.termination(0)
                #channel3
                channel3, = spcm.Channels(card, card_enable=spcm.CHANNEL3) # enable channel 1
                channel3.amp(AMPLITUDE * units.mV)
                channel3.offset(0 * units.mV)
                channel3.termination(0)
                #channel4
                channel4, = spcm.Channels(card, card_enable=spcm.CHANNEL4) # enable channel 1
                channel4.amp(AMPLITUDE * units.mV)
                channel4.offset(0 * units.mV)
                channel4.termination(0)
                #channel5
                channel5, = spcm.Channels(card, card_enable=spcm.CHANNEL5) # enable channel 1
                channel5.amp(AMPLITUDE * units.mV)
                channel5.offset(0 * units.mV)
                channel5.termination(0)
                #channel6
                channel6, = spcm.Channels(card, card_enable=spcm.CHANNEL6) # enable channel 1
                channel6.amp(AMPLITUDE * units.mV)
                channel6.offset(0 * units.mV)
                channel6.termination(0)
                #channel7
                channel7, = spcm.Channels(card, card_enable=spcm.CHANNEL7) # enable channel 1
                channel7.amp(AMPLITUDE * units.mV)
                channel7.offset(0 * units.mV)
                channel7.termination(0)
    
                if(EXIT_DAQ_FLAG==True):
                    print("Data acquisition is terminating...")
                    break
                
                # Channel triggering
                #https://github.com/SpectrumInstrumentation/spcm
                #trigger = spcm.Trigger(card)
                trigger.or_mask(spcm.SPC_TMASK_EXT0) # set the ext0 hardware input as trigger source
                trigger.ext0_mode(spcm.SPC_TM_POS) # wait for a positive edge
                trigger.ext0_level0(float(TRIGGER_LEVEL) * units.mV)
                trigger.ext0_coupling(spcm.COUPLING_DC) # set DC coupling

                """
                if(TRIGGER_CHANNEL==0):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT0)
                    trigger.ext0_mode(spcm.SPC_TM_POS)
                    trigger.ext0_level0(float(TRIGGER_LEVEL) * units.mV)
                    trigger.ext0_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)                
                    #print("Trigger channel 0")
                if(TRIGGER_CHANNEL==1):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT1)
                    trigger.ext1_mode(spcm.SPC_TM_POS)
                    trigger.ext1_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)
                    trigger.ext1_level0(float(TRIGGER_LEVEL) * units.mV)
                    print("Trigger channel 1")
                if(TRIGGER_CHANNEL==2):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT2)
                    trigger.ext2_mode(spcm.SPC_TM_POS)
                    trigger.ext2_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)
                    trigger.ext2_level0(float(TRIGGER_LEVEL) * units.mV)
                    
                if(TRIGGER_CHANNEL==3):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT3)
                    trigger.ext3_mode(spcm.SPC_TM_POS)
                    trigger.ext3_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)
                    trigger.ext3_level0(float(TRIGGER_LEVEL) * units.mV)                
                if(TRIGGER_CHANNEL==4):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT4)
                    trigger.ext4_mode(spcm.SPC_TM_POS)
                    trigger.ext4_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)
                    trigger.ext4_level0(float(TRIGGER_LEVEL) * units.mV)
                    
                if(TRIGGER_CHANNEL==5):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT5)
                    trigger.ext5_mode(spcm.SPC_TM_POS)
                    trigger.ext5_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)
                    trigger.ext5_level0(float(TRIGGER_LEVEL) * units.mV)
                    
                if(TRIGGER_CHANNEL==6):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT6)
                    trigger.ext6_mode(spcm.SPC_TM_POS)
                    trigger.ext6_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)
                    trigger.ext6_level0(float(TRIGGER_LEVEL) * units.mV)
                    
                if(TRIGGER_CHANNEL==7):                
                    trigger.and_mask(spcm.SPC_TMASK_NONE)
                    trigger.or_mask(spcm.SPC_TMASK_EXT7)
                    trigger.ext7_mode(spcm.SPC_TM_POS)
                    trigger.ext7_coupling(spcm.COUPLING_DC)
                    trigger.termination(termination=0)
                    trigger.ext7_level0(float(TRIGGER_LEVEL) * units.mV)
                """
                
                # define the data buffer
                data_transfer = spcm.DataTransfer(card)
                data_transfer.duration((PRETRIG_DURATION+POSTTRIG_DURATION)*units.ms, post_trigger_duration=POSTTRIG_DURATION*units.ms)
            
                if(True):#while(True)
    
                    if(EXIT_DAQ_FLAG==True):
                        print("Data acquisition is terminating...")
                        break
                    
                    # start card and wait until recording is finished
                    try:
                        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
                    except:
                        continue
                    #print("Finished acquiring...")
                    
                    # Start DMA transfer and wait until the data is transferred
                    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)                                                     
                    
                    signals[0]=channel0.convert_data(data_transfer.buffer[channel0, :], units.V)
                    signals[1]=channel1.convert_data(data_transfer.buffer[channel1, :], units.V)
                    signals[2]=channel2.convert_data(data_transfer.buffer[channel2, :], units.V)
                    signals[3]=channel3.convert_data(data_transfer.buffer[channel3, :], units.V)
                    signals[4]=channel4.convert_data(data_transfer.buffer[channel4, :], units.V)
                    signals[5]=channel5.convert_data(data_transfer.buffer[channel5, :], units.V)
                    signals[6]=channel6.convert_data(data_transfer.buffer[channel6, :], units.V)
                    signals[7]=channel7.convert_data(data_transfer.buffer[channel7, :], units.V)

                    #*********************************processing***************************                    
                    plate=shlp.SPlate()
                    plate.raw_signals=signals
                    plate.time=np.arange(0,len(signals[0]))
                    plate.get_segments(ref_chan_name=trigger_channel)
                    sgn_len = len(plate.sigments_sign)
                    all_labels=[]
                    all_segm_sign=[]   
                    prerpoc_time=[]
                    proc_time=[]
                    snip_number_processed=0
                    for seg_i in range(0,sgn_len):
                        #take segment
                        segm_signal=plate.sigments_sign[seg_i]
                        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                        #print(len(segm_signal))
                        if(len(plate.sigments_sign[seg_i])==0):
                            continue
                        preproc_t_strart=time.time()
                        snippets=data_proc.SplitEntireSignalIntoSnippets(signal=segm_signal,
                                                                         channs_indx=CHANNELS_TO_USE,
                                                                         torch_tensor=False,
                                                                         snip_size = snip_size,
                                                                         preproc_type = preprocessing
                                                                         )
                        preproc_t_end=time.time()
                        prerpoc_time.append(preproc_t_strart-preproc_t_end)
                        
                        start_proc=time.time()
                        labels=s_model.np_predict(np.asarray(snippets))
                        snip_number_processed+=len(labels)
                        end_proc=time.time()                        
                        proc_time.append(end_proc-start_proc)
                        
                        all_labels.append(labels) 
                        all_segm_sign.append(segm_signal)
                    #show everything in the chart   
                    if(SHOW_PROC_RESULTS==True):
                        try:                        
                            viz_start_t=time.time()
                            shlp.ShowResultsInFigure_AllSegmentsInRow(all_segm_sign,
                                                                      all_labels,
                                                                      CHANNELS_TO_USE,
                                                                      snip_size,
                                                                      colorscheme,
                                                                      SHOW_PROC_RESULTS_FIG,
                                                                      SHOW_PROC_RESULTS_AX,
                                                                      None)
                            viz_end_t=time.time()                        
                        except:
                            print("Figure is unavailable.")
                            
                    #**********************************************************************
                    
                    if(DATA_PRINT==True):                    
                        now = datetime.datetime.now()
                        signals_counter+=1
                        trig_time_st=int((PRETRIG_DURATION/1000)*SAMPLING_RATE*1000000)
                        proc_mean_t=-1                        
                        try: 
                            if(len(proc_time)!=0):proc_mean_t=sum(proc_time) / float(len(proc_time))
                        except: pass
                        print("")
                        print("******************************************")
                        print("Data received: "+str(now))
                        print("Data length/chan.: "+str(len(signals[0])))
                        print("Chans.num.: "+str(len(signals)))
                        print("Frame num.:"+str(signals_counter))
                        print("Trig. chan.: "+str(TRIGGER_CHANNEL))     
                        print("Trig. chan. max/min: "+str(np.max(signals[TRIGGER_CHANNEL]))+"/"+str(np.min(signals[TRIGGER_CHANNEL])))     
                        print("Trig.timestamp: "+str(trig_time_st/1000)+" ms")
                        print("Trig. val.:"+str(signals[TRIGGER_CHANNEL][trig_time_st]))
                        if(len(prerpoc_time)>0): print("Average preproc. time(s): "+str(float(sum(prerpoc_time) / float(len(prerpoc_time)))))
                        if(float(proc_mean_t) >= 0): print("Average proc.time(s): "+str(proc_mean_t))
                        print("Number of proc.snips.: "+str(snip_number_processed))
                        try: print("Vizualization time: "+str(viz_end_t-viz_start_t))                            
                        except: pass
                        
                        print("******************************************")
                        print("")
                    
                    # Plot the acquired data
                    if(SHOW_ORIGIN_SIGNALS==True) and (SHOW_ORIGIN_SIGNALS_FIG is not None) and (SHOW_ORIGIN_SIGNALS_AX is not None):            
                        time_data_s = data_transfer.time_data()
                        #fig, ax = plt.subplots()                                                
                        #print(channel0)
                        #print("\tMinimum: {:.3~P}".format(np.min(unit_data_V)))
                        #print("\tMaximum: {:.3~P}".format(np.max(unit_data_V)))                    
                        SHOW_ORIGIN_SIGNALS_AX.clear()
                        for ws in range(0,len(CHANNELS_TO_USE)):
                            chan_index=CHANNELS_TO_USE[ws]
                            SHOW_ORIGIN_SIGNALS_AX.plot(time_data_s, signals[chan_index], label=("channel "+str(chan_index)))
                        SHOW_ORIGIN_SIGNALS_AX.yaxis.set_units(units.mV)
                        SHOW_ORIGIN_SIGNALS_AX.xaxis.set_units(units.us)
                        SHOW_ORIGIN_SIGNALS_AX.axvline(0, color='k', linestyle='--', label='Trigger')
                        SHOW_ORIGIN_SIGNALS_AX.legend()
                        SHOW_ORIGIN_SIGNALS_FIG.canvas.draw()
                        SHOW_ORIGIN_SIGNALS_FIG.canvas.flush_events()
                    
                    if(EXIT_DAQ_FLAG==True):
                        print("Data acquisition is terminating...")
                        break

                    if(ONLY_SINGLE_SHOT==True):
                        break
                    
        #reproting of the data aquisition exit
        now = datetime.datetime.now()
        print("")
        print("******************************************")
        print("Data acquisition is terminated on event")
        print(str(now))
        print("******************************************")
#*****************************************************************************************************
#*****************************************************************************************************

#**********real time plot item**************

class RTPlotWidget(): #PySide6.QtWidgets.QWidget): #PySide6.QtWidgets.QGraphicsScene):
    
    def __init__(self,**kargs):
        
        #https://www.pythonguis.com/tutorials/plotting-pyqtgraph/
        self.window=PySide6.QtWidgets.QWidget() #QMainWindow()        
        self.window.setAttribute(PySide6.QtCore.Qt.WA_DeleteOnClose)
        self.layout=PySide6.QtWidgets.QGridLayout()        
        
        self.label =  PySide6.QtWidgets.QLabel("Measurements...")
        self.label.setMinimumWidth(130)

        self.plot_graph = pg.plot(parent=self.window)#PlotWidget(parent=self.window)
        self.scat_graph = pg.plot(parent=self.window,height=3)#PlotWidget(parent=self.window)
                
        self.layout.addWidget(self.label, 1, 0)              
        self.layout.addWidget(self.plot_graph, 2, 0)          
        self.layout.addWidget(self.scat_graph, 3, 0)          
        
        #self.layout.addWidget(self.plot_graph_1, 3, 0)             
        
        self.window.setLayout(self.layout)
        #self.window.setCentralWidget(self.plot_graph)
        self.window.show()
        
        
        """
        #https://www.geeksforgeeks.org/python/pyqtgraph-getting-parent-item-of-scatter-plot-graph/        
        layout = PySide6.QtWidgets.QGridLayout()
        self.window = PySide6.QtWidgets.QMainWindow()        #QMainWindow() #QWidget()    
        self.window.setLayout(layout)
        self.label =  PySide6.QtWidgets.QLabel("Measurements...")
        self.label.setMinimumWidth(130)
        self.plotItem = pg.plot()  
        layout.addWidget(self.label, 1, 0)              
        layout.addWidget(self.plotItem, 2, 0)
        #self.window.setCentralWidget(self.plotItem)
        self.window.show()     
        """

        """
        #https://www.qtcentre.org/threads/12135-PyQt-QTimer-problem-FIXED
        #self.window = PySide6.QtWidgets.QWidget()
        PySide6.QtWidgets.QWidget()
        #PySide6.QtWidgets.QGraphicsScene.__init__(self)       
        self.plotItem = pg.plot(title="self.RT_Figure_if_proc_results_id",parent=self)
        # make sure the item gets a parent
        #self.plotItem.setParent(self)
        #self.setCentralItem(self.plotItem)
        #plotItem.show()        
        """

    def plot(self,x,y):    
        self.plot_graph.clear()
        if(None in x):
            self.plot_graph.plot(y)
        else:
            self.plot_graph.plot(x,y)
    
    def clear(self):
        self.plot_graph.clear()

    #show vertical markers
    def vlines(self,cur_pos,min_v,max_v):
        if(isinstance(cur_pos,list) or isinstance(cur_pos,np.array)):
            for k in range(0,len(cur_pos)):
                #    self.plot_graph.plot((cur_pos[k],cur_pos[k]),(min_v,max_v),color="g")            
                #self.plot_graph.addItem(pg.InfiniteLine(pos=cur_pos[k],angle=90,color="g"))
                #draw vertical lines
                #https://stackoverflow.com/questions/61407911/pyqtgraph-color-specific-regions-in-plot
                l=self.plot_graph.addLine(x=cur_pos[k], pen=(50, 150, 50), markers=[('^', 0, 10)])
           
    def hex_to_rgb(self,value):
        h = value.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
        #value = value.lstrip('#')
        #lv = len(value)
        #return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        

    def plot_results(self,x,y,unique_l,color_scheme):

        import webcolors
        if(len(x)==0) or (len(y)==0):
            return
        #ATTENTION - we count here that zero is the normal weld, the rest are anomalies
        self.scat_graph.clear()        
        back_gr=[]
        curves=[]
        pfills=[]
        for m in range(0,len(unique_l)):
            if(unique_l[m]==-1):continue #-1 is when the system could not do analysis for some reason
            color_indx=int(unique_l[m])
            if(color_indx>len(color_scheme)):
                color_indx=len(color_scheme)-1
            hex_col=color_scheme[color_indx]
            rgb_col=self.hex_to_rgb(hex_col)
            offset=0.5
            if(m<len(unique_l)-1):offset=(unique_l[m+1]-unique_l[m])/2
            back_gr.append(np.ones(len(x))*unique_l[m]+offset)
            curve=pg.PlotCurveItem(x,back_gr[-1],pen =(rgb_col[0], rgb_col[1], rgb_col[2]))#self.scat_graph.addLine(x=x,y=back_gr[-1], pen=(rgb_col[0], rgb_col[1], rgb_col[2]), markers=[('o', 0, 5)]) 
            br=pg.mkBrush(rgb_col[0], rgb_col[1], rgb_col[2], 70)            
            if(m == 0): 
                x1 = np.ones(len(x))*unique_l[m]-offset
                zero_curve_=pg.PlotCurveItem(x,x1,pen =(rgb_col[0], rgb_col[1], rgb_col[2]))#self.scat_graph.addLine(x=x,y=x1, pen=(rgb_col[0], rgb_col[1], rgb_col[2]), markers=[('o', 0, 5)])
                pfill = pg.FillBetweenItem(zero_curve_,curve, brush = br)            
            else: 
                pfill = pg.FillBetweenItem(curve,curves[-1], brush = br)
            self.scat_graph.addItem(pfill)
            curves.append(curve)
            pfills.append(pfill)
        #l=pg.PlotCurveItem(x,y,color="red",)
        #self.scat_graph.addItem(l)
        self.scat_graph.plot(x,y,pen="b")
        
        #lena=len(arr)
        #for n in range(0,lena):
        #    hex_col=arr[n][2]
        #    rgb_col=self.hex_to_rgb(arr[n][2])
        #   color_name=webcolors.rgb_to_name(rgb_col, spec='css3')
        """
            scat.addPoints([{
                        "pos": (arr[n][0],arr[n][1]), 
                        "pen": pg.mkPen(width=5, color=color_name),#'white',
                        "symbol": 'o',
                        }])          
            
        self.scat_graph.addItem(scat)
        """
        #self.scat_graph.addLine(x=(arr[n][0]+(arr[n][1]-arr[n][0])/2), pen=(rgb_col[0], rgb_col[1], rgb_col[0]), markers=[('^', 0, 10)])


#*******************************************

if __name__ == "__main__":       
    global app
    app = QApplication(sys.argv)     
    window = MainWindow()
    window.setAttribute(PySide6.QtCore.Qt.WA_DeleteOnClose)
    window.show()        
    sys.exit(app.exec())
