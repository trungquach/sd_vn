import glob
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from datetime import datetime as dt
from datetime import timedelta
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties

# merage vm2012 log format data
# return a [samples,features] array
def merageVm2012Data(expression):
    Accs = []
    for f in sorted(glob.glob(expression), key=numericalSort):
        df = pd.read_csv(f, engine='python')
        accsPerFile = readVm2012Data(df)
        Accs.append(accsPerFile)
    Accs = np.array(Accs)
    Accs = Accs.reshape(Accs.shape[0] * Accs.shape[1])
    return Accs

# read vm2012 log format data
# read acceleration, log export datetime, log data span, frequency
def readVm2012Data(df):
    # get data
    accs = np.array(list(map(np.float32, df.ix[22:, 0])))

    return accs

def numericalSort(value):

    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])

    return parts


# global val
dataset_path = 'dataset_2018_04_26/'
normal_position1_data_path = dataset_path + 'normal_position1/'
normal_position2_data_path = dataset_path + 'normal_position2/'
bearing_position1_data_path = dataset_path + 'bearing_position1/'
bearing_position2_data_path = dataset_path + 'bearing_position2/'
gear_position1_data_path = dataset_path + 'gear_position1/'
gear_position2_data_path = dataset_path + 'gear_position2/'
csv_file_regular_expression = '*.wdat'
freq = 100000

if __name__ == "__main__":
    # set Japanese Font
    fp = FontProperties(fname=r'/System/Library/Fonts/ヒラギノ明朝 ProN.ttc', size=9)

    accs = merageVm2012Data(normal_position1_data_path + '/' + csv_file_regular_expression)
    print("Normal position 1 : " , accs.size)

    accs = merageVm2012Data(bearing_position1_data_path + '/' + csv_file_regular_expression)
    print("Bearing position 1 : " , accs.size)

    accs = merageVm2012Data(gear_position1_data_path + '/' + csv_file_regular_expression)
    print("Gear position 1 : " , accs.size)

    accs = merageVm2012Data(normal_position2_data_path + '/' + csv_file_regular_expression)
    print("Normal position 2 : " , accs.size)

    accs = merageVm2012Data(bearing_position2_data_path + '/' + csv_file_regular_expression)
    print("Bearing position 2 : " , accs.size)

    accs = merageVm2012Data(gear_position2_data_path + '/' + csv_file_regular_expression)
    print("Gear position 2 : " , accs.size)
