import sys
import json

def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])

    scores = {}  # initialize an empty dictionary
    for line in sent_file:
        # The file is tab-delimited. "\t" means "tab character"
        term, score = line.split("\t")
        scores[term] = int(score)  # Convert the score to an integer.

    # new_terms_scores = {}
    for line in tweet_file:
        tweet = json.loads(line)
        if 'text' in tweet:
            words = tweet['text'].split(' ')
            pos_words = 1
            neg_words = 1
            new_terms = []
            new_terms_score = 0
            for word in words:
                if scores.has_key(word):
                    if scores[word] > 0:
                        pos_words += 1
                    else:
                        neg_words += 1
                else:
                    new_terms.append(word)
                ratio = float(pos_words) / float(neg_words)
                if(ratio >= 1):
                    new_terms_score = pos_words
                else:
                    new_terms_score = neg_words * -1
                for new_term in new_terms:
                    print new_term, new_terms_score
                    # new_terms_scores[new_term] = new_terms_score  


if __name__ == '__main__':
    main()
