import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
path=os.getcwd()
os.chdir(path)
exec(open('DOTS.py').read())
exec(open('BO.py').read())
exec(open('TuRBO5.py').read())

