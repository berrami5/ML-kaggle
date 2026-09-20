import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


df = pd.read_csv("train.csv")
kaggle_test = pd.read_csv("test.csv")

def editFeatures (df):
    df['Sex'] = df['Sex'].map({'male':1,"female":0})
    df['Embarked'] = df['Embarked'].map({'C':0,'S':1,'Q':2}) 

editFeatures(df)
editFeatures(kaggle_test)


model = LinearRegression()

def MessingData (col, df):
    imputer = SimpleImputer(strategy="most_frequent")
    df[col] = imputer.fit_transform(df[col])

    meduim_Embarked = df['Embarked'].median()
    df['Embarked'] = df['Embarked'].fillna(meduim_Embarked)
    return df


def sets(inp):
    train_val, test_df = train_test_split(df[inp], test_size=0.2, random_state=42)

    train_df, val_df = train_test_split(train_val, test_size=0.25, random_state=42)

    feature = [x for x in inp if x != 'Survived']

    imputer = SimpleImputer(strategy='mean')

    X_train = imputer.fit_transform(train_df[feature])
    X_val = imputer.transform(val_df[feature])
    
    model.fit(X_train, train_df['Survived'])

    val_preds = model.predict(X_val)

    return val_preds, imputer, feature

MessingData(['Age','Cabin'],df)
MessingData(['Age', 'Cabin'],kaggle_test)



colVal = np.array(df.select_dtypes(include=np.number).columns)

val_preds, imputer, feature = sets(colVal)

X_kaggle = imputer.transform(kaggle_test[feature])

test_preds = model.predict(X_kaggle)
