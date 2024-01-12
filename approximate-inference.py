import os
import os.path
import matplotlib.pyplot as plt
import wget
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.io
import scipy.stats
import torch
import random
from torch.distributions.normal import Normal
from functools import partial
import matplotlib.pyplot as plt

wget.download("https://erdogdu.github.io/sta414/hw/hw2/chess_games.csv")
games = pd.read_csv("chess_games.csv")[["winner_index", "loser_index"]].to_numpy()
wget.download("https://erdogdu.github.io/sta414/hw/hw2/chess_players.csv")
names = pd.read_csv("chess_players.csv")[["index", "player_name"]].to_numpy()

games = torch.IntTensor(games)

