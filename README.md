# ICOS_bentoML_Models
In this project, the first 5 LSTM and first 5 keras models of Anastasios Giannopoulos are saved to bentoML.

To install requirements use: 
pip install -r .\requirements.txt

**rnn.py** is the main file created by Anastasios Giannopoulos.
The models created by running this file are saved to models folder

**BentoML_save.py** transfers the created model from rnn.py to bentoML.

**BentoML_load.py** loads the saved models and runs a sample validation.
