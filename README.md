# Set Up

Using python=3.10, run the following command to install the necessary packages. 

```
conda env create -f environment.yml -n new_env
```

```
conda activate new_emv
```

# Usage
```
python main.py <input (.tfrecord)> <output (.tfrecord)> <output (.csv)>
```

Input a .tfrecord of Examples with a 'sample' key. They data should be tf.string
(serialized tensor) of 1 dimension (audio data) with a sampling frequency set to 4000 Hz.

Define an Output .tfrecord that the labeled data will be written to.

Define a .csv meta file that will store the state of each sample.

Example:
```
python main.py audio_test.tfrecord audio_test_cherrypick.tfrecord audio_test_cherrypick_meta.csv
```

# SigVu Overview
SigVu is a Python based signal manipulation and labeling program.
I started developing this project when working on the Cornel & CSU, Chico 
Elephant Listening Project, when the need to relabel some of the samples was 
raised. 

This project utilizes standard ML, plotting, and data handling libraries
to process and label data, all bottled up in a simply PyQt5 GUI.

<img src="assets/current_gui.png" width="600px" />

It serves a few purposes for me:
1. Interactive manipulation of data view signal processing functionality.
2. Real time view of the signal.
3. Simplified data labeling.
4. Python software engineering practice.

So far, this is the high-level UML for the archtecture of the project.

<img src="assets/highlevel-uml.png" width="600px" />