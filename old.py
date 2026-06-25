class RTPlotWidget_0(): #PySide6.QtWidgets.QWidget): #PySide6.QtWidgets.QGraphicsScene):
    
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

        # import webcolors
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
        
        """This function tracks the spectrum card and acquires data in real time. It is designed to work with the Spectrum M2P series of data acquisition cards."""

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
                card.timeout(10 * units.s)                     # timeout 5 s
            
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
                    sgn_len = len(plate.segments_sign)
                    all_labels=[]
                    all_segm_sign=[]   
                    prerpoc_time=[]
                    proc_time=[]
                    snip_number_processed=0
                    for seg_i in range(0,sgn_len):
                        #take segment
                        segm_signal=plate.segments_sign[seg_i]
                        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                        #print(len(segm_signal))
                        if(len(plate.segments_sign[seg_i])==0):
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