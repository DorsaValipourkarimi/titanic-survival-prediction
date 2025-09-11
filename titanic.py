import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "titanic/train.csv"

df = pd.read_csv(file_path)

#print(df.head())
#print(df.columns) 
#print(df.shape)      

# Convert to numpy array just to check shape
data_array = df.to_numpy() #Convert the dataset to Numpy array
print(data_array.shape)
#print(data_array[:3])

#ubset = df[["Survived", "Age", "Pclass" ]]
#print(subset.head(5))

# Heatmap for numeric correlations
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
plt.savefig("correlation_heatmap.png") 

# Correlation with Survived
corr = df.corr(numeric_only=True)["Survived"].sort_values(ascending=False)
print(corr)

#Group survival rates
print(df.groupby("Sex")["Survived"].mean())
print(df.groupby("Pclass")["Survived"].mean()) 
print(df.groupby("Embarked")["Survived"].mean())

#Sex-based model:
df["SPrediction"] = df["Sex"].apply(lambda x:1 if x == "female" else 0)

#Accuracy:
Saccuracy = (df["SPrediction"] == df["Survived"]).mean()
print("Sex-Based Model Accuracy level:", Saccuracy)

#Class-based model:
df["CPrediction"] = df["Pclass"].apply(lambda x:1 if x == 1 else 0)

#Accuracy:
Caccuracy = (df["CPrediction"] == df["Survived"]).mean()
print("Class-Based Model Accuracy level:", Caccuracy)

#Age-based model:
df["APrediction"] = df["Age"].apply(lambda x:1 if x < 16 else 0)

#Accuracy:
Aaccuracy = (df["APrediction"] == df["Survived"]).mean()
print("Age-Based Model Accuracy level:", Aaccuracy)