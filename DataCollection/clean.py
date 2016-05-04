import pandas as pd
import os
import scrapy as sc


# The data's already all cleaned up!
data = pd.read_csv("../data/data.csv")

print(data['link'][0])

# Second step is visiting the links and getting the submission title and (if it's a self-post) body content
class RedditSpider(sc.Spider):
    name = 'reddit'
    start_urls = data['link'][0]

    def parse_title(self, response):
      for

    def parse_score(self, response):


