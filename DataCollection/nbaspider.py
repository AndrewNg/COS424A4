import pandas as pd
import scrapy as sc
from html.parser import HTMLParser
from datetime import datetime
from datetime import date

# We first import all of our data
data = pd.read_csv("../data/data.csv")

# We're only going to be taking data from /r/nba
data = data.loc[data['link'].str.contains('http://reddit.com/r/nba')]

def find_between( s, first, last ):
  try:
      start = s.index( first ) + len( first )
      end = s.index( last, start )
      return s[start:end]
  except ValueError:
      return ""

# We need to intelligently select which links we want to parse.
# We want to scrape once for each submission and propogate all of the titles
submission_ids = []

for link in data['link']:
  submission_ids.append(find_between(link, 'http://reddit.com/r/nba/comments/', '/c/'))

data['submission_id'] = submission_ids

# Only take a single instance of each submission_id
data = data.drop_duplicates(['submission_id'], take_last=True)

# Visit the links and fetch the following pieces of data
# submission_utc, title, body if exists
# bring over data from the original data data set (comment, comment time, comment score, etc.)
class RedditSpider(sc.Spider):
  name= 'reddit'
  start_urls = data['link']
  title = ""
  score = ""
  submission_utc = ""
  objects = {}

  def parse(self, response):
    parser = HTMLParser()
    sel1 = sc.Selector(text = response.css('a.title').extract()[0], type="html")
    for node in sel1.css('a *::text'):
      title = node.extract()

    sel2 = sc.Selector(text = response.css('div.score.unvoted').extract()[0], type="html")
    for node in sel2.css('div *::text'):
      score = node.extract()

    sel3 = response.css('time::attr(datetime)').extract()[2]
    sel3 = sel3.replace("T", " ")
    sel3 = sel3.replace("+00:00", "")
    stringdate = datetime.strptime(sel3, '%Y-%m-%d %H:%M:%S')
    timestamp = (stringdate.toordinal() - date(1970, 1, 1).toordinal()) * 24*60*60

    submission_utc = timestamp

    url = response.meta.get('redirect_urls', [response.url])[0]

    objects = {
      'title': title,
      'score': score,
      'submission_utc': submission_utc,
      'comment': data.loc[data['link'] == url]['body'].values[0],
      'comment_utc': data.loc[data['link'] == url]['created_utc'].values[0],
      'comment_score': data.loc[data['link'] == url]['score'].values[0],
      'subreddit': 'nba',
      'link': url,
      'submission_id': data.loc[data['link'] == url]['submission_id'].values[0],
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