import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as plt

def prepare_data(df):
    X= df.drop("y", axis=1)
    y= df["y"]
    return X,y