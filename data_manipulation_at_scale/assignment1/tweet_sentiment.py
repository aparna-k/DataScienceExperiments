import sys
import json

def hw():
    print 'Hello, world!'


def lines(fp):
    print str(len(fp.readlines()))


def main():
    afinnfile = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    scores = {}  # initialize an empty dictionary
    for line in afinnfile:
        # The file is tab-delimited. "\t" means "tab character"
        term, score = line.split("\t")
        scores[term] = int(score)  # Convert the score to an integer.
    tweets_score = []

    for line in tweet_file:
        tweet = json.loads(line)
        if 'text' in tweet:
            words = tweet['text'].split(' ')
            score = 0
            for word in words:
                score += scores.get(word.lower(), 0)
            print score
            # tweets_score.append(score)
        else:
            print '0'
            # tweets_score.append(0)



if __name__ == '__main__':
    main()
