import pandas as pd
import re
def preprocess(df):    
    
    # convert to lowercase
    df['tweet'] = df['tweet'].astype(str).str.lower()

    # remove urls
    df['tweet'] = df['tweet'].str.replace(r'http\S+|www\S+', '', regex=True)

    # remove mentions and hashtags
    df['tweet'] = df['tweet'].str.replace(r'(@|#)\w+', '', regex=True)

    # remove RT
    df['tweet'] = df['tweet'].str.replace(r'\brt\b[:]*', '', regex=True)

    # remove html entities
    df['tweet'] = df['tweet'].str.replace(r'&\w+;', '', regex=True)

    # remove punctuation
    df['tweet'] = df['tweet'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    # remove extra spaces
    df['tweet'] = df['tweet'].str.replace(r'\s+', ' ', regex=True)

    # strip spaces
    df['tweet'] = df['tweet'].str.strip()
    return df