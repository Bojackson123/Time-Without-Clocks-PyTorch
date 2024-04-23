
import os
import cv2
import numpy as np
from joblib import load

from src.parameters import Parameters
from src.AlexNetNew import AlexNet
from src.updateNetwork import UpdateNetwork

from src.updateNetworkData import ExportUpdateNetworkData

import warnings
warnings.filterwarnings("ignore")


def analyze_video(frames):
    
    # Load the frames from the given directory
    frame_paths = [os.path.join(frames, file) for file in os.listdir(frames) if file.endswith(".jpg")]

    # Load the trained regression model
    models_list = load('pretrained_regression/regression_model.joblib')
    
    # Initialize the AlexNet model and create a dictionary to store the figures and JSON data
    alexnet = AlexNet()
    figures = {}
    json_datas = {}
    
    # Different attention % of baseline values
    c_values = [1/.5, 1/1, 1/1.5, 1/2]
    upNets = {}
    params_list = {}
    
    # Create an instance of Parameters and UpdateNetwork for each C value
    name_counter = 0.0
    for c in c_values:
        name_counter += 1.0
        param = Parameters(C=c)
        params_list[c] = param.params
        upNets[name_counter] = UpdateNetwork(params_list[c])

    for frame_path in frame_paths:
        # Load the frame and shape it
        frame = cv2.imread(frame_path)
        frame = shape_frame(frame)

        # Run the frame through the AlexNet to get the distances at each layer
        label = alexnet.run(frame)

        # Run the frame through each UpdateNetwork instance and calculate the accumulators
        for c, upNet in upNets.items():
            upNet.run(frame, alexnet.features, alexnet.output_prob, alexnet.states, alexnet.labels)

    # Pull the accumulator values from each UpdateNetwork instance and predict the time
    for c, upNet in upNets.items():
        # Convert the accumulator values to a 1x4 array
        acc = list(upNet.accumulator.values())
        acc_array = np.array(acc).reshape(1, -1)
        
        estimate = average_predict(models_list, acc_array)
        
        # Plots the graphs and exports the data to JSON
        figure = upNet.plot(alexnet.states, alexnet.labels, False, estimate)
        figures[c] = figure
        
        export = ExportUpdateNetworkData(upNet)
        json_datas[c] = export.to_json()
    
    # Return the figures and JSON data for each attention value as a dict
    return figures, json_datas

def average_predict(models, X_new):
    # Make predictions with all models
    predictions = np.array([model.predict(X_new) for model in models])
    # Average the predictions across all models
    return np.mean(predictions, axis=0)    

# Shapes the frame to a square by cropping the longer side
def shape_frame(frame):
    if np.shape(frame)[0] > np.shape(frame)[1]:
        onset = np.shape(frame)[0] - np.shape(frame)[1]
        return frame[int(onset/2):-int(onset/2), :]
    elif np.shape(frame)[0] < np.shape(frame)[1]:
        onset = np.shape(frame)[1] - np.shape(frame)[0]
        return frame[:, int(onset/2):-int(onset/2), :]
    return frame
