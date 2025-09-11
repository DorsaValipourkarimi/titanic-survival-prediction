import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

##Sex-based model:
df["SPrediction"] = df["Sex"].apply(lambda x:1 if x == "female" else 0)

#Accuracy:
Saccuracy = (df["SPrediction"] == df["Survived"]).mean()
print("Sex-Based Model Accuracy level:", Saccuracy)

##Class-based model:
df["CPrediction"] = df["Pclass"].apply(lambda x:1 if x == 1 else 0)

#Accuracy:
Caccuracy = (df["CPrediction"] == df["Survived"]).mean()
print("Class-Based Model Accuracy level:", Caccuracy)

##Age-based model:
df["APrediction"] = df["Age"].apply(lambda x:1 if x < 16 else 0)

#Accuracy:
Aaccuracy = (df["APrediction"] == df["Survived"]).mean()
print("Age-Based Model Accuracy level:", Aaccuracy)

##Fare-based model:
df["Fare"] = df["Fare"].fillna(df["Fare"].mean()) #handle missing values:
fare_threshold = df["Fare"].mean()
df["FPrediction"] = df["Fare"].apply(lambda x:1 if x >= fare_threshold else 0)

#Accuracy:
Faccuracy = (df["FPrediction"] == df["Survived"]).mean()
print("Fare-Based Model Accuracy level:", Faccuracy)

############################################
# #Sex & Class Based model:
df2 = df.copy()
#Features
df2["sex_female"] = (df2["Sex"] == "female").astype(int)  #1 if female, 0 if male
df2["class_n"] = (4 - df2["Pclass"])/3.0  # 1st=1.0, 2nd=0.667, 3rd=0.333

X = df2[["sex_female" ,"class_n"]].to_numpy()
y = df2["Survived"].to_numpy()

def Predict(weights, X, threshold=0.5):
    w = np.array(weights, dtype=float)
    w = w / (w.sum() + 1e-9) #to normalize weights to add up to 1
    survival_score = (X * w).sum(axis=1)
    return (survival_score >= threshold).astype(int) #return whether they survive (1) or not (0)

weights_ex = [3,3]
predict_ex = Predict(weights_ex, X)
accuracy_ex = (predict_ex == y).mean()
print("Accuracy of this model with given weights", weights_ex, ":", accuracy_ex)

'''
#weights where class dominates
for w in ([1,3], [1,4], [1,5]):
    preds = Predict(w, X)  # uses your normalized scoring & 0.5 cutoff
    acc = (preds == y).mean()
    print(w, acc)
'''
#When Class is given  more weight than Sex, the model degrads!

################################
#Sex & Fare Based model:
df2 = df.copy()

#Features
df2["sex_female"] = (df2["Sex"] == "female").astype(int)  #1 if female, 0 if male
df2["Fare"] = df2["Fare"].fillna(df2["Fare"].median()) #Handle scale and nulls

#Outlier clipping
fare_min = df2["Fare"].quantile(0.01)
fare_max = df2["Fare"].quantile(0.99)
fare_clip = df2["Fare"].clip(fare_min, fare_max)
df2["fare_n"] = (fare_clip - fare_min) / (fare_max - fare_min +1e-9) #normalize min max to [0,1]


X = df2[["sex_female" ,"fare_n"]].to_numpy()
y = df2["Survived"].to_numpy()

def Predict(weights, X, threshold=0.5):
    w = np.array(weights, dtype=float)
    survival_score = (X * w).sum(axis=1)
    return (survival_score >= threshold).astype(int) #return whether they survive (1) or not (0)

for w in ([5,1], [4,2], [3,2], [1,5], [2,1]):
    preds = Predict(w, X, threshold=0.5)
    acc = (preds == y).mean()
    print(f"Sex+Fare weights {w} -> acc={acc:.4f}")
    
#still heavily rely on sex, focusing on fare still degrade the model