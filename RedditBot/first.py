import praw

user_agent = ("COS424 0.1")

r = praw.Reddit(user_agent)

subreddit = r.get_subreddit("learnpython")

for submission in subreddit.get_hot(limit = 5):
    print("Title: " + submission.title)
    print("Text: " + submission.selftext)
    print("Score: " + str(submission.score))
    print("---------------------------------\n")