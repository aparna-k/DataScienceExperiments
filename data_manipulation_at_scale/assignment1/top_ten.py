import sys
import json

tweet_file = open(sys.argv[1])

hashtags_count = {}
for line in tweet_file:
    tweet = json.loads(line)
    if 'text' in tweet:
        hashtags = tweet['entities']['hashtags']
        for entry in hashtags:
            hashtag = entry['text']
            hashtags_count[hashtag] = hashtags_count.get(hashtag, 0) + 1

highest_scorers = sorted(hashtags_count, key=hashtags_count.get, reverse=True)[:10]
for item in highest_scorers:
    print item, hashtags_count[item]