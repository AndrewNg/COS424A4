import pandas as pd
import os
import scrapy as sc


# The data's already all cleaned up!
data = pd.read_csv("../data/data.csv")

# we only want the comments that exceed a nice threshold
data = data.loc[data['score'] >= 20]

print(data.loc[data['link'] == 'http://reddit.com/r/nba/comments/4ciu5u/c/d1iwox1']['score'].values[0])