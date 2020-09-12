import pandas as pd

data = [1, 2, 3]

dataframe = pd.DataFrame({'data': data})

dataframe.to_csv('test.csv', index=False, sep=',')