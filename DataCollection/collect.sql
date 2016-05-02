SELECT body, created_utc, score, 'http://reddit.com/r/'+subreddit+'/comments/'+REGEXP_REPLACE(link_id, 't[0-9]_','')+'/c/'+id as link
  FROM [fh-bigquery:reddit_comments.2016_03]
  WHERE subreddit IN ('nba', 'aww', 'worldnews', 'GlobalOffensive', 'circlejerk', 'funny', 'pics')
  AND author NOT IN (SELECT author FROM [fh-bigquery:reddit_comments.bots_201505])
  AND body IS NOT NULL and body <> '[deleted]'