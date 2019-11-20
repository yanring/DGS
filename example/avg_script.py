import pandas as pd

df = pd.read_csv('sys_analysis.log', sep='\t')
print(df.mean())
