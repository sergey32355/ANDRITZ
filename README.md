# Anomaly detection GUI for welding of BPPs

Anomaly detection GUI for the identification of welding defects in BPPs from features extracted from signals from optical sensors.

## Terminology

* plate: a bipolar plate.
* segment: a segment of the plate welded without interruption; it has a specific name.
* snippet: a small chunk of a segment (typical length: 1mm).
* features: different statistical features extracted for all signal channels. Computed at the snippet level.
* labels: whether a segment is defective or not. There are no labels at the snippet level.

## Intended use

The GUI can load data (with or without labels), plot data, extract features, train/apply an anomaly detection model. More in details:
* the training can be done only with labelled data; only the snippets from non-defective segments are used;
* the main model outputs an anomaly score for each snippet to which it is applied;
* a second layer aggregates the anomaly scores of the snippets and decides which segments are to be considered anomalous (defective).

The following sequences of actions can be taken:

1. Load data and plot it.
2. Load data without labels, load a pre-trained model, run it on the loaded data (-> output: segments predicted to be defective). Then you can also plot the data (-> plot all segments of a plate: the segments predicted to be defective are marked in red; plot only one segment: the snippets with a darker shade of red have a higher anomaly score).
3. Load data with labels, then either load a pre-trained model, and run it on the loaded data (-> output: performance metrics of the model) or train a model on the data (-> output: trained model, and its performance metrics; the train-test split is done automatically).

## Practical notes

The program assumes that there is a base data folder inside the folder "data", which stays mostly fixed (but can be changed through the config.py file) and contains other data sub-folders. When you use the GUI, you can type in the name of the desired sub-folder, and all the data contained therein will be loaded. Moreover, in the base data folder there should be a folder called "labels". If the data of the sub-folder "example" has labels, they should be contained in the file "example.xlsx" in the "labels" folder. Only with the correct notation the labels will be loaded without errors.

The trained models are saved in the "models" folder. The default naming convention for models is as follows:

{model kind} _ {name of data sub-folder used for training} _ {date and time of training} .pkl

However, the model names can be changed manually. Just be aware that a model can only be applied to data obtained with the same sensors as the data used for training.

## Terminal commands

To open and use the GUI:
python -m gui_files.main_gui

To edit the appearance of the GUI manually:
pyside6-designer gui_files/empa_gui.ui

To save the manual edits to the empa_guy.py file:
pyside6-uic gui_files/empa_gui.ui -o gui_files/empa_gui.py

## Requisites

This program requires a python 3.12 installation. The required dependences can be installed either with poetry (through pyproject.toml / poetry.lock) or using the requirements.txt file.

## Licensing

This project’s source code is licensed under the [MIT License](LICENSE). It uses [PySide6](https://wiki.qt.io/Qt_for_Python), which is licensed under the [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.html). See the [NOTICE](NOTICE) for details.