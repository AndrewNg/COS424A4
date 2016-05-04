import pandas as pd
import os
import scrapy as sc
from bs4 import BeautifulSoup


# The data's already all cleaned up!
data = pd.read_csv("../data/data.csv")

# we only want the comments that exceed a nice threshold
data = data.loc[data['score'] >= 20]

# Second step is visiting the links and getting the submission title and (if it's a self-post) body content
class RedditSpider(sc.Spider):
    name = 'reddit'
    start_urls = [data['link']]
    title = ""
    score = ""

    def parse(self, response):
      sel1 = sc.Selector(text = response.css('a.title').extract()[0], type="html")
      for node in sel1.css('a *::text'):
        title = node.extract()

      sel2 = sc.Selector(text = response.css('div.score.unvoted').extract()[0], type="html")
      for node in sel2.css('div *::text'):
        score = node.extract()

      yield {
        'title': title,
        'score': score
      }


