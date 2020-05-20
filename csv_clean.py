import pandas as pd

df = pd.read_csv('owid-covid-data.csv')
a = df[df.iso_code == 'OWID_WRL']
a.to_csv('new_csv.csv', index=False)

df = pd.read_csv('new_csv.csv')
a = df.drop('date', axis=1)
a.to_csv('new_covid.csv', index_label='Day')