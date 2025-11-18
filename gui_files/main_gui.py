import os
import sys
import traceback
import joblib
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit
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
matplotlib.use('Qt5Agg')

DEBUG_MODE = False

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_EmpaGUI()
        self.ui.setupUi(self)

        # Make output text boxes read-only
        self.ui.load_data_output.setReadOnly(True)
        self.ui.load_model_output.setReadOnly(True)
        self.ui.train_output.setReadOnly(True)

        # Connect button click
        self.ui.load_data_button.clicked.connect(self.load_data)
        self.ui.load_model_button.clicked.connect(self.load_model)
        self.ui.train_button.clicked.connect(self.train_model)
        self.ui.plot_button.clicked.connect(self.plot_signal)
        self.ui.predict_button.clicked.connect(self.predict)

        # Set dropdown options
        model_names = [m.split(".")[0] for m in os.listdir("models") if m.endswith(".pkl")]
        self.ui.load_model_dropdown.addItems(model_names)

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

    def set_buttons(self, enabled: bool):
        self.ui.load_data_button.setEnabled(enabled)
        self.ui.load_model_button.setEnabled(enabled)
        self.ui.train_button.setEnabled(enabled)
        self.ui.predict_button.setEnabled(enabled)
        self.ui.plot_button.setEnabled(enabled)

    def load_data(self):

        self.set_buttons(enabled=False)

        # Clear previous output
        self.ui.load_data_output.clear()

        data_path = self.ui.load_data_path.text()
        bpp_type = self.ui.load_data_plate_type_dropdown.currentText()

        # Setup QThread + Worker
        self.load_data_thread = QThread()
        load_data_args = [bpp_type, data_path]
        self.load_data_worker = Worker(DataLoaderBPPLines, *load_data_args, output_widget=self.ui.load_data_output)
        extract_features_args = [self.dl]  # will be updated later
        self.extract_features_worker = Worker(FeatureExtractorBPPLines, *extract_features_args, output_widget=self.ui.load_data_output)
        self.load_data_worker.moveToThread(self.load_data_thread)
        self.extract_features_worker.moveToThread(self.load_data_thread)

        # Connect signals:
        # When thread starts, run load_data_worker
        self.load_data_thread.started.connect(self.load_data_worker.run)
        # When the workers emit an error signal (using error.emit()), it triggers the self.load_data_error method
        self.load_data_worker.error.connect(self.load_data_error)
        # When the load_data_worker emits the message signal (using message.emit()), it triggers the self.load_data_finished method
        self.load_data_worker.message.connect(self.load_data_finished)
        # When load_data_worker emits the message, use it to update the input dl for extract_features_worker
        self.load_data_worker.message.connect(self.extract_features_worker.update_args)
        # When load_data_worker finishes, run extract_features_worker
        self.load_data_worker.finished.connect(self.extract_features_worker.run)
        # When the workers emit an error signal (using error.emit()), it triggers the self.load_data_error method
        self.extract_features_worker.error.connect(self.load_data_error)
        # When the extract_features_worker emits the message signal (using message.emit()), it triggers the self.extract_features_finished method
        self.extract_features_worker.message.connect(self.extract_features_finished)
        # When extract_features_worker finishes, quit the thread
        self.extract_features_worker.finished.connect(self.load_data_thread.quit)
        # When the thread completes (successfully or with error), it emits the finished signal
        # The deleteLater() method is called, which schedules the thread object for deletion
        # Memory is cleaned up automatically when Qt's event loop processes the deletion
        self.load_data_thread.finished.connect(self.load_data_thread.deleteLater)
        
        # Start thread
        self.load_data_thread.start()

    def load_data_error(self, err_text):
        # Option 1: append to text box
        self.ui.load_data_output.append(f"ERROR\n{err_text}")
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
        self.ui.load_data_output.append("Extracted features successfully.")
        self.ui.load_data_output.append("DONE")
        self.set_buttons(enabled=True)
        # # Reset worker and thread references after cleanup
        # self.load_data_worker = None
        # self.extract_features_worker = None
        # self.load_data_thread = None

    def train_model(self):

        # Clear previous output
        self.ui.train_output.clear()
        self.set_buttons(enabled=False)

        # Setup QThread + Worker
        self.train_thread = QThread()
        self.train_worker = Worker(ModelSelectorBPPLines, self.fe, output_widget=self.ui.train_output)
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
        self.ui.train_output.append(f"ERROR\n{err_text}")
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
        self.ui.train_output.append("DONE")
        # Set dropdown options
        self.ui.load_model_dropdown.clear()
        model_names = [m.split(".")[0] for m in os.listdir("models") if m.endswith(".pkl")]
        self.ui.load_model_dropdown.addItems(model_names)
        self.set_buttons(enabled=True)
        # # Reset worker and thread references
        # self.train_worker = None
        # self.train_thread = None

    def load_model(self):
        self.ui.load_model_output.clear()
        self.set_buttons(enabled=False)
        try:
            model_name = self.ui.load_model_dropdown.currentText()
            model_path = os.path.join("models", f"{model_name}.pkl")
            loaded_model_data = joblib.load(model_path)
            self.model = loaded_model_data['trained_model']
            self.threshold_params = loaded_model_data['best_threshold_params']
            self.ui.load_model_output.append(f"Loaded model {model_name}.\nDONE")
        except Exception as e:
            self.ui.load_model_output.append(f"ERROR loading model {model_name}:\n{str(e)}")
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

# Runs functions in background threads to prevent UI freezing
class Worker(QObject):
    message = Signal(object)  # Changed to object to pass any result type
    error = Signal(str)
    finished = Signal()

    def __init__(self, func, *args, output_widget=None):
        super().__init__()
        self.func = func
        self.args = args
        self.output_widget = output_widget
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def update_args(self, *new_args):
        self.args = new_args

    def run(self):
        try:
            # Redirect stdout/stderr to this worker if output_widget is provided
            if self.output_widget and not DEBUG_MODE:
                sys.stdout = self
                sys.stderr = self
            
            result = self.func(*self.args)   # run your module
            self.message.emit(result)  # Emit the actual result object
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            # Restore original stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.finished.emit()

    def write(self, text):
        if text.strip() and self.output_widget:
            self.output_widget.append(text.strip())
        

    def flush(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())