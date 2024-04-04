import numpy
import math
import pandas
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import os
#from tqdm import tqdm #TODO uninstalled


if torch.cuda.is_available():
    print('yes')

