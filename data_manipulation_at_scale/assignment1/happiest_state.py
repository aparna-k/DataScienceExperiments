import sys
import json


us_states_abr_to_expansion = {'ak': 'alaska',
'al': 'alabama',
'ar': 'arkansas',
'as': 'american samoa',
'az': 'arizona',
'ca': 'california',
'co': 'colorado',
'ct': 'connecticut',
'dc': 'district of columbia',
'de': 'delaware',
'fl': 'florida',
'ga': 'georgia',
'gu': 'guam',
'hi': 'hawaii',
'ia': 'iowa',
'id': 'idaho',
'il': 'illinois',
'in': 'indiana',
'ks': 'kansas',
'ky': 'kentucky',
'la': 'louisiana',
'ma': 'massachusetts',
'md': 'maryland',
'me': 'maine',
'mi': 'michigan',
'mn': 'minnesota',
'mo': 'missouri',
'mp': 'northern mariana islands',
'ms': 'mississippi',
'mt': 'montana',
'na': 'national',
'nc': 'north carolina',
'nd': 'north dakota',
'ne': 'nebraska',
'nh': 'new hampshire',
'nj': 'new jersey',
'nm': 'new mexico',
'nv': 'nevada',
'ny': 'new york',
'oh': 'ohio',
'ok': 'oklahoma',
'or': 'oregon',
'pa': 'pennsylvania',
'pr': 'puerto rico',
'ri': 'rhode island',
'sc': 'south carolina',
'sd': 'south dakota',
'tn': 'tennessee',
'tx': 'texas',
'ut': 'utah',
'va': 'virginia',
'vi': 'virgin islands',
'vt': 'vermont',
'wa': 'washington',
'wi': 'wisconsin',
'wv': 'west virginia',
'wy': 'wyoming'}

us_states_expansion_to_abr = {v: k for k, v in us_states_abr_to_expansion.iteritems()}

sent_file = open(sys.argv[1])
tweet_file = open(sys.argv[2])

scores = {}  # initialize an empty dictionary
for line in sent_file:
    # The file is tab-delimited. "\t" means "tab character"
    term, score = line.split("\t")
    scores[term] = int(score)  # Convert the score to an integer.

# new_terms_scores = {}

tweets_state_score = {}
for line in tweet_file:
    tweet = json.loads(line)
    if 'text' in tweet:
        words = tweet['text'].split(' ')
        score = 0
        for word in words:
            score += scores.get(word.lower(), 0)
        
        user_location = tweet['user']['location']
        if user_location != None:
            words_in_location = user_location.split()
            for loc in words_in_location:
                loc = loc.lower()
                if us_states_abr_to_expansion.get(loc, False):
                    tweets_state_score[loc] = tweets_state_score.get(loc, 0) + score
                elif us_states_expansion_to_abr.get(loc, False):
                    abbr_state = us_states_expansion_to_abr[loc]
                    tweets_state_score[abbr_state] = tweets_state_score.get(abbr_state, 0) + score

print max(tweets_state_score, key=tweets_state_score.get).upper()