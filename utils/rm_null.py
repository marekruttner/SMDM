import pandas as pd

df = pd.read_csv('translated_merged_tweets.csv')

clean_df = df.dropna(subset=['Content'])

clean_df = clean_df.to_csv('nullrm_dataset.csv', index=False)