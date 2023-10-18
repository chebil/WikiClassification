import pandas as pd
#read from pkl
df2 = pd.read_pickle("from0.pkl")
#write to excel
df2.to_excel("from0.xlsx")