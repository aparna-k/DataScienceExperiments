import sys
import json

tweet_file = open(sys.argv[1])

term_freq_dict = {}

for line in tweet_file:
    tweet = json.loads(line)
    if 'text' in tweet:
        words = tweet['text'].split()
        for word in words:
            term_freq_dict[word] = term_freq_dict.get(word, 0) + 1

num_occurences_all_terms = sum(term_freq_dict.values())

# term_freq_hist_dict = {}

for key in term_freq_dict.keys():
    term_hist = term_freq_dict[key] / float(num_occurences_all_terms)
    print key, term_hist
    # term_freq_hist_dict[key] = term_freq_dict[key] / float(num_occurences_all_terms) 