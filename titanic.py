import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "titanic/train.csv"

df = pd.read_csv(file_path)

#print(df.head())
#print(df.columns) 
#print(df.shape)      

data_array = df.to_numpy() #Convert the dataset to Numpy array
print(data_array.shape)
#print(data_array[:3])

#ubset = df[["Survived", "Age", "Pclass" ]]
#print(subset.head(5))

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
plt.savefig("correlation_heatmap.png") 

corr = df.corr(numeric_only=True)["Survived"].sort_values(ascending=False)
print(corr)

print(df.groupby("Sex")["Survived"].mean())
print(df.groupby("Pclass")["Survived"].mean()) 
print(df.groupby("Embarked")["Survived"].mean())