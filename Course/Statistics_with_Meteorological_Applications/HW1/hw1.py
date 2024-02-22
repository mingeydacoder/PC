import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('TS.txt', unpack='true')

def plot_histogram_1(data):
    bins = np.arange(min(data), max(data)+1 , 1.48)
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel('T')
    plt.ylabel('# of amounts')
    plt.title('Histogram of TS Data')
    plt.xticks(np.arange(0, max(data) + 5, 5))
    plt.yticks(np.arange(0, 251, 50))
    plt.show()

def plot_histogram_2(data):
    bins = np.arange(min(data), max(data)+1 , 0.28)
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel('T')
    plt.ylabel('# of amounts')
    plt.title('Histogram of TS Data')
    plt.xticks(np.arange(0, max(data) + 5, 5))
    plt.yticks(np.arange(0, 81, 10))
    plt.show()

plot_histogram_1(data)
plot_histogram_2(data)

# Calculate seasonal cycle of TS and tile the array of seasonal cycle
TS_SeasonalCycle = np.tile(np.nanmean(data.reshape((-1, 12)), axis=0), data.shape[0]//12)

# Calculate seasonal anomaly
TS_Anomaly = data - TS_SeasonalCycle

def plot_histogram_3(data):
    bins = np.arange(min(data), max(data) , 0.049)
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel('T')
    plt.ylabel('# of amounts')
    plt.title('Histogram of TS Data')
    plt.xticks(np.arange(-1.5, 1.5+0.5, 0.5))
    plt.yticks(np.arange(0, 101, 20))
    plt.show()

plot_histogram_3(TS_Anomaly)