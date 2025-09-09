import pandas as pd

file_path = "titanic/train.csv"

df = pd.read_csv(file_path)

print(df.head())
print(df.columns) 
print(df.shape)      

data_array = df.to_numpy() #Convert the dataset to Numpy array
print(data_array.shape)