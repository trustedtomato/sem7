from time import sleep

import httpx
import pandas as pd
from googletrans import Translator
from tqdm import tqdm

# Load ptb-xl database
path = 'data/ptb-xl/'
df = pd.read_csv(path+'ptbxl_database.csv')
timeout = httpx.Timeout(20)
translator = Translator(timeout=timeout, service_urls=['translate.google.com'])
error_indices = []
# Create empty dataframe for translated reports
df_translated = df.copy()

# Iterate over rows in df
for index, row in tqdm(df.iterrows()):
    # Translate report
    report = row['report']
    if report is not None:
        try:
            report_translated = translator.translate(report, src='auto', dest='en')
            df_translated.at[index, 'report'] = report_translated.text
            if index % 100 == 0:
                print(index, report, report_translated.text)
        except Exception as e:
            print(e)
            print("Error translating report for index: ", index)
            error_indices.append(index)
    else:
        print("No report for index: ", index)
    # Sleep to avoid getting blocked
    sleep(1)

# Save translated reports
df_translated.to_csv(path+'ptbxl_database_translated.csv', index=False)

# Save errors
with open(path+'errors.txt', 'w') as f:
    for error_index in error_indices:
        f.write(f'{error_index}\n')