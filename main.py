import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('meta_data_file', type=str, help='Path to the meta data file')
    return parser.parse_args()

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
    def __init__(self,  input_file, event_bus, parent=None):
        super().__init__()
        self.parent = parent

        self.input_file = input_file
        self.current_index = 0
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

    def init_UI(self):
        layout = QVBoxLayout()

        windowLabel = QLabel("Change Sample")
        windowLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(windowLabel)
        layout.addStretch()

        self.prevButton = QPushButton('Previous', self)
        self.prevButton.setFixedWidth(100)
        self.prevButton.clicked.connect(lambda: self.update_sample(-1))
        layout.addWidget(self.prevButton)

        self.nextButton = QPushButton('Next', self)
        self.nextButton.setFixedWidth(100)
        self.nextButton.clicked.connect(lambda: self.update_sample(1))
        layout.addWidget(self.nextButton)

        layout.addStretch()

        self.setLayout(layout)
        self.update_sample(0)

class SpectrogramModelView(FigureCanvas):
    def __init__(self, get_fresh_signal_function, event_bus, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)

        self.sample_rate = 4000
        self.frame_length = 1024
        self.frame_step = 64
        self.max_freq = 512

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

    def update_plot(self):
        self.ax.clear()

        signal = self.get_signal()  # Grab a fresh signal
        spec = self.apply_spectrogram(signal, self.frame_length, self.frame_step, self.sample_rate, self.max_freq)

        img = self.ax.imshow(spec, aspect='auto', origin='lower', 
                             extent=[0, len(signal) / self.sample_rate, 0, self.max_freq],
                             cmap='Reds')

        self.ax.set_ylim(1, self.max_freq)  # Set y-axis limits
        self.ax.set_title('Spectrogram')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        self.draw()

class SpectrogramControllerView(QWidget):
    def __init__(self,spectrogram_viewer, parent=None):
        super().__init__()

        self.setParent(parent)
        self.spectrogram_viewer = spectrogram_viewer

        self.init_UI()

    def add_slider(self, layout, label_text, param_name, min_value, max_value, initial_value):
        slider_layout = QHBoxLayout()
        slider = QSlider(self)
        slider.setOrientation(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(initial_value)
        value_label = QLabel(f"{slider.value()}")
        slider.valueChanged.connect(lambda value: self.update_param(param_name, value, value_label))

        label_text_label = QLabel(label_text)
        min_value_label = QLabel(f"{min_value}")
        max_value_label = QLabel(f"{max_value}")

        label_text_label.setFixedWidth(100)
        min_value_label.setFixedWidth(10)
        max_value_label.setFixedWidth(50)
        value_label.setFixedWidth(50)

        slider_layout.addWidget(label_text_label)
        slider_layout.addWidget(min_value_label)
        slider_layout.addItem(QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Minimum))
        slider_layout.addWidget(slider)
        slider_layout.addItem(QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Minimum))
        slider_layout.addWidget(max_value_label)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)

    def update_param(self, param_name, value, value_label):
        self.spectrogram_viewer.update_param(param_name, value)
        value_label.setText(f"{value}")

    def init_UI(self):
        layout = QVBoxLayout()

        windowLabel = QLabel("Adjust Spectrogram Window")
        windowLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(windowLabel)
        layout.addStretch()

        self.add_slider(layout, 'Frame Length', 'frame_length', 1, self.spectrogram_viewer.sample_rate, self.spectrogram_viewer.frame_length)
        self.add_slider(layout, 'Frame Step', 'frame_step', 1, 256, self.spectrogram_viewer.frame_step)
        self.add_slider(layout, 'Max Frequency', 'max_freq', 1, 2048, self.spectrogram_viewer.max_freq)
        
        layout.addStretch()

        self.setLayout(layout)
        
class LabelerControllerView(QWidget):
    def __init__(self, output_file, get_fresh_signal_function, event_bus, parent=None):
        super().__init__()
        self.parent=parent
        
        self.output_file = output_file
        self.output_file_writer = tf.io.TFRecordWriter(output_file)

        self.event_bus = event_bus
        self.get_signal = get_fresh_signal_function

        self.init_UI()

    def label_sample(self, label):
        """
        Append a new TFRecord
        """

        example = tf.train.Example(features=tf.train.Features(feature={
                'sample': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(self.get_signal()).numpy()])
                ),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])
                )
        }))
        self.output_file_writer.write(example.SerializeToString())

        # Notify the system that a signal is being labeled
        self.event_bus.publish("labeler.signal_labeled", label)

        # Request the next signal
        self.event_bus.publish("labeler.queue_next_signal")

    def close_output_file(self):
        self.output_file_writer.close()
        exit()

    def init_UI(self):
        layout = QVBoxLayout()

        # Labeler layout
        windowLabel = QLabel("Change Sample")
        windowLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(windowLabel)
        layout.addStretch()

        self.keepButton = QPushButton('Elephant', self)
        self.keepButton.clicked.connect(lambda: self.label_sample(1))
        layout.addWidget(self.keepButton)

        self.discardButton = QPushButton('Not Elephant', self)
        self.discardButton.clicked.connect(lambda: self.label_sample(0))
        layout.addWidget(self.discardButton)

        layout.addStretch()

        self.setLayout(layout)

class MetaFileControllerView(QWidget):
    def __init__(self, meta_file, input_file, event_bus, parent=None):
        super().__init__()
        self.parent = parent
        self.meta_file = meta_file
        self.input_file = input_file
        self.meta_data = self.load_meta_file(meta_file)
        self.event_bus = event_bus
        self.event_bus.subscribe(self)

        self.stored_index = 0

        self.init_UI()

    def notify(self, event, *args):
        if event == "data.index_update":
            self.stored_index = args[0]

            existing_row = self.meta_data[(self.meta_data['file'] == self.input_file) & (self.meta_data['index'] == self.stored_index)]

            print(existing_row)
            # Grab the meta information if it is available
            if not existing_row.empty:
                current_label = existing_row['label'].iloc[0]
                logger.info("Found existing label.")
            else:
                current_label = '-'
                logger.info("No pre-existing label found.")

            self.update_UI(self.stored_index, current_label)

        elif event == "labeler.signal_labeled":
            label = args[0]
            self.write_line(self.input_file, self.stored_index, label)

    def write_line(self, file, index, label):
        new_row = pd.DataFrame({
            'file': [file],
            'index': [index],
            'label': [label]
        })

        existing_row = self.meta_data[(self.meta_data['file'] == file) & (self.meta_data['index'] == index)]

        if not existing_row.empty:
            self.meta_data.loc[existing_row.index, 'label'] = label
            logger.info("Overwritting sample.")
        else:
            self.meta_data = pd.concat([self.meta_data, new_row], ignore_index=True)
            logger.info("Writing new sample.")

        self.meta_data.to_csv(self.meta_file, index=False)

    def load_meta_file(self, meta_file):
        if not os.path.exists(meta_file):
            meta_data = pd.DataFrame(columns=['file', 'index', 'label'])
        else:
            meta_data = pd.read_csv(meta_file)

        return meta_data

    def update_UI(self, index, label):
        self.labelValue.setText(f"{label}")
        self.indexValue.setText(f"{index}")
    
    def init_UI(self):
        metaFileLayout = QVBoxLayout()

        labelLayout = QHBoxLayout()
        self.labelLabel = QLabel("Label")
        self.labelValue = QLabel("-")
        labelLayout.addWidget(self.labelLabel)
        labelLayout.addWidget(self.labelValue)

        indexLayout = QHBoxLayout()
        self.indexLabel = QLabel("Index")
        self.indexValue = QLabel(f"{self.stored_index}")
        indexLayout.addWidget(self.indexLabel)
        indexLayout.addWidget(self.indexValue)

        metaFileLayout.addLayout(indexLayout)
        metaFileLayout.addLayout(labelLayout)

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
            output_file=self.output_tfrecord_file, 
            get_fresh_signal_function=self.dataControllerView.get_signal, 
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