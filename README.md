# CS 4775 Final Project

## Reproducting Results

From the from the ./final-project/src/models directory run

```bash
python3 evaluate.py
```

This loads in our test set generator and model for inference and displays the 
accuracy and AUC ROC.

# Running project from scratch

## Download and Preprocess Data

from the ./final-project/src/data directory run

```bash
python3 scrape.py
python3 summary_to_csv.py
python3 get_summary_dicts.py
python3 preprocess.py
```

This scrapes all the datasets we use from http://cnn.csail.mit.edu/motif_discovery/, 
generates data structures for faster training, and preprocesses the data for 
faster training.

## Training And Testing Model 

To train and the model run. It should be noted that we trained on a single 
NVIDIA GeForce GTX 1080 Ti and took us roughly 3.5 hours to train and evaluate

```bash
python3 my_classes.py
```

(Recomended) To run in the background and output to file

```bash
nohup python3 -u my_classes.py > output.txt &
```

The results of our run can be found in ./final_project/src/models/output_100_epoch.txt.
The model parameters we use are the ones already coded in my_classes.py

Hyperparameters for the model itself and parameters for calculating metrics can 
be set in the my_classes.py folder directly. 


