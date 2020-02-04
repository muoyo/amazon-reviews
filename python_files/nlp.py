"""
##### NATURAL LANGUAGE PROCESSING #####

This module contains functions for dealing with Natural Language Processing (NLP)

"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Return dictionary of VADER sentiment scores for given sentence. Keys: 'neg', 'neu', 'pos', 'compound'
def sentiment_analyzer_scores(sentence, verbose=False):
    score = analyzer.polarity_scores(sentence)
    if verbose: print("{}".format(str(score)))
    
    return score
    
def sentiment_analyzer_scores_neg(sentence):
    return sentiment_analyzer_scores(sentence)['neg']

def sentiment_analyzer_scores_neu(sentence):
    return sentiment_analyzer_scores(sentence)['neu']

def sentiment_analyzer_scores_pos(sentence):
    return sentiment_analyzer_scores(sentence)['pos']

def sentiment_analyzer_scores_compound(sentence):
    return sentiment_analyzer_scores(sentence)['compound']

def append_sentiment_scores(df):
    df['neg'] = df['review_fulltext'].apply(sentiment_analyzer_scores_neg)
    df['neu'] = df['review_fulltext'].apply(sentiment_analyzer_scores_neu)
    df['pos'] = df['review_fulltext'].apply(sentiment_analyzer_scores_pos)
    df['compound'] = df['review_fulltext'].apply(sentiment_analyzer_scores_compound)
    
    return df


