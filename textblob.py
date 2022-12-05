# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:41:20 2022

@author: Zining
"""

import pandas as pd
import numpy as np
import pandas as pd
import nltk

df = pd.read_csv('youtube_comment_raw.csv')
df['Comment'][3077] = str(df['Comment'][3077])

from textblob import TextBlob
df["sentiment_score"] = df["Comment"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],
                           ['neg', 'neu', 'pos'])

df.to_csv('Sentiment Analysis textblob.csv')