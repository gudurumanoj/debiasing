""" 
    file to plot the data from the txt file and save the plot as a png file
    takes all arguments as cmd inputs
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os




def plot_loss(df, numcols,save_path):
    """
    Function to plot the data from the csv file and save the plot as a png file
    :param file_name: name of the dataframe
    :param save_path: path to save the plot
    csv file format: epoch, step, learning rate, loss
    """
    ## plots loss for across each step
    # df = pd.read_csv(file_name)
    plt.figure()
    # plt.
    plt.plot(df['loss'], label='Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Steps')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'steploss.png'))
    # plt.close()
    
    ## plots loss for across each epoch
    ## each epoch has 6 steps, so may be take average of across 6 steps and plot
    loss = df['loss'].values
    loss = loss.reshape(-1, numcols)  ## reshape to numcols columns, if there are numcols steps
    loss = np.mean(loss, axis=1)
    plt.figure()
    plt.plot(loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'epochloss.png'))

def plot_mAP(df, save_path):
    """
    Function to plot the mAP for training data
    :param file_name: name of the dataframe
    :param save_path: path to save the plot
    csv file format: mAP_regular, mAP_ema
    """
    ## plots mAP for regular and ema
    # df = pd.read_csv(file_name)
    plt.figure()
    plt.plot(df['epoch'], df['mAP_regular'], label='mAP score regular')
    plt.plot(df['epoch'], df['mAP_ema'], label='mAP score ema')
    plt.xlabel('Epoch')
    plt.ylabel('mAP score')
    plt.title('mAP score vs Epoch')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'mAP.png'))
    plt.close()

def convert_to_df(file_name):
    """
    Function to convert the txt file to a dataframe
    :param file_name: name of the txt file
    :return: dataframe
    text file has the following format:
        - Epoch [0/80], Step [000/642], LR 4.0e-06, Loss: 4001.5 (has 6 lines of this format for each epoch and step)
        - mAP score regular 0.0000, map score ema 0.0000
    """
    ## convert txt to csv
    with open(file_name, 'r') as f:
        lines = f.readlines()
    data = []
    for i in range(len(lines)):
        if 'mAP' in lines[i]:
            data.append(lines[i].split(' ')[-2:])
        elif 'Epoch' in lines[i]:
            data.append(lines[i].split(' ')[-1])
    return data

## argument parsing
parser = argparse.ArgumentParser(description='Plot the data from the txt file and save the plot as a png file')
parser.add_argument('--txt-path', type=str, help='path of the txt files')
parser.add_argument('--csv-path', type=str, help='path of the csv files')
parser.add_argument('--save-path', type=str, help='path to save the plots')
args = parser.parse_args()

# convert txt to csv


## loading the csv file
## joining path of the directory with train.csv and val.csv to get the full path and then loading the csv file

# train_file = os.path.join(args.file_path, 'train.csv')
val_file = os.path.join(args.file_path, 'val.csv')
# train_df = pd.read_csv(train_file, names=['epoch', 'step', 'loss'])
# train_df.dropna(inplace=True)
val_df = pd.read_csv(val_file, names=['epoch', 'step', 'loss'])
val_df.dropna(inplace=True)
## creating the save path by joining the save_path with train.csv and val.csv
# trainpng = os.path.join(args.save_path, 'train')
valpng = os.path.join(args.save_path, 'val')

## calling the function to plot the data
# plot_loss(train_df, 7, trainpng)
plot_loss(val_df, 4, valpng)
