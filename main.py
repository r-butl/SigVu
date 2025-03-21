import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider, QSpacerItem, QSizePolicy, QSpinBox, QMessageBox, QFrame
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Input a .TFrecord and a .csv.")
    parser.add_argument('input_file', type=str, help='Path to the input file (.tfrecord)')
    parser.add_argument('output_file', type=str, help='Path to the output file (.tfrecord)')
    parser.add_argument('meta_data_file', type=str, help='Path to the meta data file (.csv)')
    args = parser.parse_args()

    # Check if all required arguments are provided
    if not args.input_file or not args.output_file or not args.meta_data_file:
        parser.print_help()
        sys.exit(1)

    return args
    
class EventBus:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, subscriber):
        """Add a new subscriber"""
        if subscriber not in self.subscribers:
            self.subscribers.append(subscriber)

    def unsubscribe(self, subscriber):
        """Remove an existing subscriber"""
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)

    def publish(self, event, *args):
        """Notify all subscribers about an event"""
        logger.info(f"Event published : {event}")
        for subscriber in self.subscribers:
            subscriber.notify(event, *args)

class DataControllerView(QWidget):
    def __init__(self,  input_file, output_file, event_bus, parent=None):
        super().__init__()
        self.parent = parent

        self.input_file = input_file
        self.current_index = 0

        self.output_file = output_file

        self.dataset = self.load_tfrecords(input_file)
        self.current_signal = None

        self.event_bus = event_bus
        self.event_bus.subscribe(self)

        self.init_UI()

    def notify(self, event, *args):
        if event == "labeler.queue_next_signal":
            self.update_sample(1)   # if the labeler requests a new signal, push the indexer forward 1
        elif event == "data.request_signal_copy":
            self.publish_signal()
        elif event == "meta_file.meta_data":
            meta_data = args[0]
            self.write_data(meta_data)

    def get_signal(self):
        return self.current_signal

    def load_tfrecords(self, input_file):
        """
        Loads and parses TFRecords into a tf.data.Dataset.
        """
        def parse_function(example_proto):
            """
            Parses a single TFRecord example.
            """
            feature_description = {
                'sample': tf.io.FixedLenFeature([], tf.string),  # Serialized tensor
                'label': tf.io.FixedLenFeature([], tf.int64)     # Integer label
            }
            
            parsed_example = tf.io.parse_single_example(example_proto, feature_description)
            sample = tf.io.parse_tensor(parsed_example['sample'], out_type=tf.float32)
            
            return sample
        
        dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(input_file))
        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        self.total_records = tf.data.experimental.cardinality(dataset).numpy()

        return list(dataset)

    def update_sample(self, move_direction):
        self.current_index += move_direction

        try:
            self.current_signal = self.dataset[self.current_index]
        except StopIteration:
            self.label.setText(f"Index out of range: {self.current_index}")
            self.closeout()

        self.publish_signal()

    def publish_signal(self):
        self.event_bus.publish("data.signal_update", self.current_signal)
        self.event_bus.publish("data.index_update", self.current_index)

    def trigger_write_data(self):
        # Create a confirmation dialog
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Confirmation Write Data")
        msg_box.setText("You are about to write the labels and samples to the output file.")
        msg_box.setInformativeText("This action will overwrite any existing data in the output file. Do you wish to continue?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        response = msg_box.exec_()

        if response == QMessageBox.Yes:
            self.event_bus.publish("data.request_meta_data")

    def write_data(self, meta_file):
        
        output_file_writer = tf.io.TFRecordWriter(self.output_file)
        rows = meta_file[meta_file['file'] == self.input_file]

        if not rows.empty:
            for _, row in rows.iterrows():
                index = row['index']
                label = row['label']
                logger.info(f"Writing index {index} with label {label} to file {self.output_file}")
                example = tf.train.Example(features=tf.train.Features(feature={
                    'sample': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(self.dataset[index]).numpy()])
                    ),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])
                    )
                }))
                output_file_writer.write(example.SerializeToString())

        output_file_writer.close()

    def init_UI(self):
        layout = QVBoxLayout()

        windowLabel = QLabel("<b>Nav & Output</b>")
        windowLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(windowLabel)
        layout.addStretch()

        self.nextButton = QPushButton('Next', self)
        self.nextButton.setFixedWidth(100)
        self.nextButton.clicked.connect(lambda: self.update_sample(1))
        layout.addWidget(self.nextButton)

        self.prevButton = QPushButton('Previous', self)
        self.prevButton.setFixedWidth(100)
        self.prevButton.clicked.connect(lambda: self.update_sample(-1))
        layout.addWidget(self.prevButton)

        self.writeDataButton = QPushButton('Write Data', self)
        self.writeDataButton.setFixedWidth(100)
        self.writeDataButton.clicked.connect(lambda: self.trigger_write_data())
        layout.addWidget(self.writeDataButton)

        layout.addStretch()

        self.setLayout(layout)
        self.update_sample(0)

class SpectrogramModelView(FigureCanvas):
    def __init__(self, get_fresh_signal_function, event_bus, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)

        self.sample_rate = 4000
        self.frame_length = 2048
        self.frame_step = 32
        self.max_freq = 200

        self.event_bus = event_bus
        self.event_bus.subscribe(self)
        self.get_signal = get_fresh_signal_function

        self.update_plot()

    def notify(self, event, *args):
        if event=="data.signal_update":
            self.update_plot()  # Update grabs the signal automatically

    def update_param(self, param, value):

        if hasattr(self, param):
            setattr(self, param, value)
        else:
            logger.error(f"Spectrogram controller: Param {param} does not exist.")
        
        self.update_plot()

    def apply_spectrogram(self, signal, frame_length, frame_step, sample_rate, max_frequency):
        
        freq_resolution = sample_rate / frame_length
        bins_to_grab = int(max_frequency / freq_resolution)

        stft = tf.signal.stft(
            signal,
            frame_length=frame_length,
            frame_step=frame_step,
            window_fn=tf.signal.hann_window
        )
        stft = stft[:, 2:bins_to_grab]
        stft = tf.math.log(tf.abs(stft) + 1e-10)
        return stft.numpy().T

    # def update_plot(self):
    #     self.ax.clear()

    #     signal = self.get_signal()  # Grab a fresh signal
    #     spec = self.apply_spectrogram(signal, self.frame_length, self.frame_step, self.sample_rate, self.max_freq)
        
    #     time_axis = np.arange(spec.shape[1]) * (self.frame_step / self.sample_rate)
    #     img = self.ax.imshow(   spec, aspect='auto', origin='lower', 
    #                             extent=[time_axis[0], time_axis[-1] + (self.frame_step / self.sample_rate), 0, 
    #                             self.max_freq], cmap='Reds')

    #     self.ax.set_ylim(1, self.max_freq)  # Set y-axis limits
    #     self.ax.set_title('Spectrogram')
    #     self.ax.set_xlabel('Time (s)')
    #     self.ax.set_ylabel('Frequency (Hz)')
            
    #     self.draw()

    def update_plot(self):
        # Clear the figure completely
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)  # Recreate the main axis

        signal = self.get_signal()  # Grab a fresh signal
        spec = self.apply_spectrogram(signal, self.frame_length, self.frame_step, self.sample_rate, self.max_freq)

        time_axis = np.arange(spec.shape[1]) * (self.frame_step / self.sample_rate)
        img = self.ax.imshow(spec, aspect='auto', origin='lower',
                            extent=[time_axis[0], time_axis[-1] + (self.frame_step / self.sample_rate), 0,
                                    self.max_freq], cmap='Reds')

        self.ax.set_ylim(1, self.max_freq)  # Set y-axis limits
        self.ax.set_title('Spectrogram')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

        # Add new colorbar
        self.cbar = self.figure.colorbar(img, ax=self.ax)
        self.cbar.set_label("Intensity (dB)")

        self.draw()

class SpectrogramControllerView(QWidget):
    def __init__(self, spectrogram_viewer, parent=None):
        super().__init__()

        self.setParent(parent)
        self.spectrogram_viewer = spectrogram_viewer

        self.init_UI()

    def add_spinbox(self, layout, label_text, param_name, min_value, max_value, initial_value):
        contoller_layout = QHBoxLayout()

        label_text_label = QLabel(label_text)
        label_text_label.setFixedWidth(100)
        min_value_label = QLabel(f"{min_value}")
        min_value_label.setFixedWidth(10)
        max_value_label = QLabel(f"{max_value}")
        max_value_label.setFixedWidth(50)

        spinbox = QSpinBox(self)
        spinbox.setMinimum(min_value)
        spinbox.setMaximum(max_value)
        spinbox.setValue(initial_value)
        spinbox.setFixedWidth(75)

        spinbox.valueChanged.connect(lambda value: self.update_spinbox(param_name, value))

        contoller_layout.addWidget(label_text_label)
        contoller_layout.addWidget(spinbox)
        layout.addLayout(contoller_layout)

    def update_spinbox(self, param_name, value):
        self.spectrogram_viewer.update_param(param_name, value)

    def init_UI(self):
        layout = QVBoxLayout()

        windowLabel = QLabel("<b>Adjust Spectrogram Window</b>")
        windowLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(windowLabel)
        layout.addStretch()

        self.add_spinbox(layout, 'Frame Length', 'frame_length', 1, self.spectrogram_viewer.sample_rate, self.spectrogram_viewer.frame_length)
        self.add_spinbox(layout, 'Frame Step', 'frame_step', 1, 256, self.spectrogram_viewer.frame_step)
        self.add_spinbox(layout, 'Max Frequency', 'max_freq', 1, 2048, self.spectrogram_viewer.max_freq)
        
        layout.addStretch()

        self.setLayout(layout)
        
class LabelerControllerView(QWidget):
    def __init__(self, event_bus, parent=None):
        super().__init__()
        self.parent=parent

        self.event_bus = event_bus

        self.init_UI()

    def label_sample(self, label):
        # Send the label to the meta file
        self.event_bus.publish("labeler.signal_labeled", label)

        # Request the next signal
        self.event_bus.publish("labeler.queue_next_signal")

    def close_output_file(self):
        self.output_file_writer.close()
        exit()

    def init_UI(self):
        layout = QVBoxLayout()

        # Labeler layout
        windowLabel = QLabel("<b>Apply Label</b>")
        windowLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(windowLabel)
        layout.addStretch()

        self.positiveButton = QPushButton('Elephant', self)
        self.positiveButton.clicked.connect(lambda: self.label_sample(1))
        layout.addWidget(self.positiveButton)

        self.negativeButton = QPushButton('Not Elephant', self)
        self.negativeButton.clicked.connect(lambda: self.label_sample(0))
        layout.addWidget(self.negativeButton)

        self.nolabelButton = QPushButton('No Label', self)
        self.nolabelButton.clicked.connect(lambda: self.label_sample(-1))
        layout.addWidget(self.nolabelButton)

        layout.addStretch()

        self.setLayout(layout)

class MetaFileControllerView(QWidget):
    def __init__(self, meta_file, input_file, event_bus, parent=None):
        super().__init__()
        self.parent = parent
        self.meta_file = meta_file
        self.input_file = input_file
        self.event_bus = event_bus
        self.event_bus.subscribe(self)
        self.stored_index = 0
        self.current_label = '-'

        self.load_meta_file(meta_file)

        self.init_UI()

        self.update_label_count()
        self.update_UI()

    def load_meta_file(self, meta_file):
        if not os.path.exists(meta_file):
            self.meta_data = pd.DataFrame(columns=['file', 'index', 'label'])
            self.elephant_count = 0
            self.nonelephant_count = 0
        else:
            self.meta_data = pd.read_csv(meta_file)
            self.update_label_count()

    def notify(self, event, *args):
        if event == "data.index_update":
            self.stored_index = args[0]

            existing_row = self.meta_data[(self.meta_data['file'] == self.input_file) & (self.meta_data['index'] == self.stored_index)]

            # Grab the meta information if it is available
            if not existing_row.empty:
                self.current_label = existing_row['label'].iloc[0]
            else:
                self.current_label = '-'

            self.update_UI()

        elif event == "labeler.signal_labeled":
            label = args[0]
            self.write_line(self.input_file, self.stored_index, label)
            self.update_label_count()

        elif event == "data.request_meta_data":
            self.event_bus.publish("meta_file.meta_data", self.meta_data)

    def write_line(self, file, index, label):

        existing_row = self.meta_data[(self.meta_data['file'] == file) & (self.meta_data['index'] == index)]

        if label == -1:
            if not existing_row.empty:
                self.meta_data = self.meta_data.drop(existing_row.index)
                logger.info(f"Removing row with file: {file} and index: {index}.")
            else:
                logger.info(f"No row found with file: {file} and index: {index} to remove.")
        else:
            new_row = pd.DataFrame({
                'file': [file],
                'index': [index],
                'label': [label]
            })

            if not existing_row.empty:
                self.meta_data.loc[existing_row.index, 'label'] = label
                logger.info(f"Updated row with file: {file} and index: {index} to label: {label}.")
            else:
                self.meta_data = pd.concat([self.meta_data, new_row], ignore_index=True)
                logger.info(f"Added row with file: {file} and index: {index} to label: {label}.")

        self.meta_data.to_csv(self.meta_file, index=False)
        self.update_label_count()

    def update_label_count(self):
        existing_row = self.meta_data[(self.meta_data['file'] == self.input_file) & (self.meta_data['index'] == self.stored_index)]

        if not existing_row.empty:
            self.current_label = existing_row['label'].iloc[0]

        self.elephant_count = len(self.meta_data[(self.meta_data['file'] == self.input_file) & (self.meta_data['label'] == 1)])
        self.nonelephant_count = len(self.meta_data[(self.meta_data['file'] == self.input_file) & (self.meta_data['label'] == 0)])

    def update_UI(self):
        self.labelValue.setText(f"{self.current_label}")
        self.indexValue.setText(f"{self.stored_index}")
        self.elephantValue.setText(f"{self.elephant_count}")
        self.nonelephantValue.setText(f"{self.nonelephant_count}")
    
    def init_UI(self):
        metaFileLayout = QVBoxLayout()

        sectionTitle = QLabel("<b>Status</b>")
        sectionTitle.setAlignment(Qt.AlignCenter)
        metaFileLayout.addWidget(sectionTitle)

        labelLayout = QHBoxLayout()
        self.labelLabel = QLabel("Label")
        self.labelValue = QLabel("-")
        self.labelValue.setAlignment(Qt.AlignRight)
        labelLayout.addWidget(self.labelLabel)
        labelLayout.addWidget(self.labelValue)

        indexLayout = QHBoxLayout()
        self.indexLabel = QLabel("Index")
        self.indexValue = QLabel(f"{self.stored_index}")
        self.indexValue.setAlignment(Qt.AlignRight)
        indexLayout.addWidget(self.indexLabel)
        indexLayout.addWidget(self.indexValue)

        elephantLayout = QHBoxLayout()
        self.elephantLabel = QLabel("Elephant")
        self.elephantValue = QLabel(f"{self.stored_index}")
        self.elephantValue.setAlignment(Qt.AlignRight)
        elephantLayout.addWidget(self.elephantLabel)
        elephantLayout.addWidget(self.elephantValue)

        nonelephantLayout = QHBoxLayout()
        self.nonelephantLabel = QLabel("Non-Elephant")
        self.nonelephantValue = QLabel(f"{self.stored_index}")
        self.nonelephantValue.setAlignment(Qt.AlignRight)
        nonelephantLayout.addWidget(self.nonelephantLabel)
        nonelephantLayout.addWidget(self.nonelephantValue)

        metaFileLayout.addStretch()
        metaFileLayout.addLayout(indexLayout)
        metaFileLayout.addLayout(labelLayout)
        metaFileLayout.addLayout(elephantLayout)
        metaFileLayout.addLayout(nonelephantLayout)
        metaFileLayout.addStretch()

        self.setFixedWidth(200)

        self.setLayout(metaFileLayout)

class MainWindow(QMainWindow):
    def __init__(self, input_tfrecord_file, output_tfrecord_file, meta_file):
        super().__init__()

        # load the meta file into a pandas dataframe
        self.meta_file = meta_file
        self.input_tfrecord_file = input_tfrecord_file
        self.output_tfrecord_file = output_tfrecord_file

        self.initUI()

    def initUI(self):
        self.setWindowTitle('TFRecord Labeler')
        self.setGeometry(100, 100, 1000, 800)

        self.eventBus = EventBus()

        # Data Controller
        self.dataControllerView = DataControllerView(
            parent=self, 
            input_file=self.input_tfrecord_file, 
            output_file=self.output_tfrecord_file, 
            event_bus=self.eventBus
        )

        # Spectrogram View
        self.spectrogramModelView = SpectrogramModelView(
            parent=self, 
            get_fresh_signal_function=self.dataControllerView.get_signal, 
            event_bus=self.eventBus, 
            width=5, 
            height=4, 
            dpi=100
        )

        # Spectogram Controller
        self.spectrogramContollerView = SpectrogramControllerView(
            parent=self, 
            spectrogram_viewer=self.spectrogramModelView
        )

        # Data Labeler
        self.labelerControllerView = LabelerControllerView(
            event_bus=self.eventBus, 
            parent=self
        )

        self.metaFileControllerView = MetaFileControllerView(
            input_file=self.input_tfrecord_file,
            meta_file=self.meta_file,
            event_bus=self.eventBus
        )

        mainLayout = QVBoxLayout()

        mainLayout.addWidget(self.spectrogramModelView)

        dashboardLayout = QHBoxLayout()

        dashboardSpacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        dashboardLayout.addItem(dashboardSpacer)
        dashboardLayout.addWidget(self.dataControllerView)
        dashboardLayout.addWidget(self.labelerControllerView)
        dashboardLayout.addWidget(self.spectrogramContollerView)
        dashboardLayout.addWidget(self.metaFileControllerView)
        dashboardLayout.addItem(dashboardSpacer)

        dashboardContainer = QWidget()
        dashboardContainer.setLayout(dashboardLayout)
        dashboardContainer.adjustSize()
        dashboardContainer.setFixedHeight(dashboardContainer.sizeHint().height()) # dashboard size fixes to initial size

        mainLayout.addWidget(dashboardContainer)

        # Set the layout for the central widget
        centralWidget = QWidget(self)
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        # Show the main window
        self.show()

def main():
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    meta_data_file = args.meta_data_file

    app = QApplication(sys.argv)
    mainWin = MainWindow(input_file, output_file, meta_data_file)
    mainWin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
