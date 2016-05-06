import pandas as pd
import os
import scrapy as sc
from html.parser import HTMLParser


# The data's already all cleaned up!
data = pd.read_csv("../data/data.csv")

# we only want the comments that exceed a nice threshold
data = data.loc[data['score'] >= 20]

def find_between( s, first, last ):
  try:
      start = s.index( first ) + len( first )
      end = s.index( last, start )
      return s[start:end]
  except ValueError:
      return ""


# Second step is visiting the links and getting the submission title and (if it's a self-post) body content
class RedditSpider(sc.Spider):
    name = 'reddit'
    start_urls = data['link']
    title = ""
    score = ""
    comment = ""
    body = ""
    objects = {}

    def parse(self, response):
      parser = HTMLParser()
      sel1 = sc.Selector(text = response.css('a.title').extract()[0], type="html")
      for node in sel1.css('a *::text'):
        title = node.extract()

      sel2 = sc.Selector(text = response.css('div.score.unvoted').extract()[0], type="html")
      for node in sel2.css('div *::text'):
        score = node.extract()

      sel3 = sc.Selector(text = response.css('div.usertext-body.may-blank-within.md-container').extract()[1], type="html")
      for node in sel3.css('p *::text'):
        comment = node.extract()
      comment = parser.unescape(comment)

      url = response.meta.get('redirect_urls', [response.url])[0]

      objects =  {
        'title': title,
        'score': score,
        'comment': comment,
        'comment_score': data.loc[data['link'] == url]['score'].values[0],
        'subreddit': find_between(url, "http://reddit.com/r/", "/comments/"),
        'link': url
      }

      self_url = url.split('/c/', 1)[0]
      request = sc.Request(self_url, callback = self.parse_self)
      request.meta['objects'] = objects

      yield request

    def parse_self(self, response):
      parser = HTMLParser()
      body = ""
      objects = response.meta['objects']
      objects['self_url'] = response.url
      if (response.css('.expando div div').extract()):
        sel4 = sc.Selector(text = response.css('.expando div div').extract()[0], type="html")
        for node in sel4.css('p *::text'):
          body += node.extract()

        body = parser.unescape(body)
      else:
        body = ""

      objects['body'] = body

      yield objects


