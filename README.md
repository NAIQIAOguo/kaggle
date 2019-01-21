# QUESTION DESCRIPTION 

* On E-commerce sites, the quality of product listing is crucial for improving search relevance and gaining customer attention. 
In this competition, you are provided a set of product names, description, and attributes,
as well as the quality score of their listing as rated by real customers.
You are challenged to build a model to automatically predict the quality of another set of product listings.

# CLEAN DATA

Using function in pandas read csv file. Using replace()function with regular expression to clean data.
```javascript
def load(file_name):
    df_test = pd.read_csv(file_name)
    sLength = len(df_test['id'])
    df_test.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_test = df_test.fillna("miss")
```
 And combining all the column in the csv file except price. 
```javascript
import pandas as pd
X_train =loadX('train_data.csv')
X_test = load('test_data.csv')
df_train =pd.DataFrame(data=X_train['lvl1']+X_train['lvl2']+X_train['lvl3']+X_train['type']+X_train['name']+X_train['descrption'],
columns=['train'])
X_train_text = df_train.train
```
Because I combine all the data from different columns, So I can get all word from the file , which means I can use more word to form dictionary to tokenize data in every place.

# FEATURE DATA
