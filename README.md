# ICOS_bentoML_Models
In this project, the first 5 LSTM and first 5 keras models of Anastasios Giannopoulos are saved to bentoML.

To install requirements use: 
pip install -r .\requirements.txt

**src/RNN/rnn.py** is the main file created by Anastasios Giannopoulos.
The models created by running this file are saved to src/RNN/models folder with keras save model. 

**src/Anomaly_Detector/Anomaly_Detector.py** is the file created by Anastasios Giannopoulos for Anomaly Detection.
The models created by running this file are saved to src/Anomaly_Detector/models folder with pkl.

**BentoML_save_{}.py** transfers the created models to bentoML.
rnn -> keras
Anomaly_Detector -> scikit learn

**BentoML_load_{}.py** loads the saved models and runs a sample validation.
It also saves a sample set to send to Ceadar 
