# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:49:44 2022

@author: Zining
"""
import pandas as pd
import numpy as np
import pandas as pd
import nltk
nltk.download('vader_lexicon')

df = pd.read_csv('youtube_comment_raw.csv')
df['Comment'][3077] = str(df['Comment'][3077])

from nltk.sentiment import SentimentIntensityAnalyzer
import operator

sia = SentimentIntensityAnalyzer()
df["sentiment_score"] = df["Comment"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],
                           ['neg', 'neu', 'pos'])

df.to_csv('Sentiment Analysis NLTK.csv')