import pandas as pd
import numpy as np
from datetime import datetime

#input_csv = "PMI_24_Sep_2020/Store1_24_Sep_2020_Complete(3423).csv"     #input csv file path
#input_csv = "PMI_24_Sep_2020/Store2_24_Sep_2020_Complete(13879).csv"
input_csv = "PMI_24_Sep_2020/Store3_24_Sep_2020_Complete(6238).csv"
output_csv = "tally_3.csv"
df = pd.read_csv(input_csv)


df = df.fillna(0)


values = list(df.values)

values.sort(key = lambda date: datetime.strptime(":".join(date[0].split("_")[-4:]), '%H:%M:%S:%f'))
values = np.array(values)[:, 1:]

x = np.zeros(15)
for i in range(values.shape[0]-1):
    tally = np.multiply(values[i]- values[i+1], (values[i]- values[i+1])<0)
    if not [i>3 for i in tally].count(1)>=4:
        x = np.add(x, tally)
x = x*-1
new_df = pd.DataFrame(np.array([df.columns[1:].values, x]))
new_df.to_csv(output_csv, index=False)

