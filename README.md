# QUESTION DESCRIPTION 

* On E-commerce sites, the quality of product listing is crucial for improving search relevance and gaining customer attention. 
In this competition, you are provided a set of product names, description, and attributes,
as well as the quality score of their listing as rated by real customers.
You are challenged to build a model to automatically predict the quality of another set of product listings.

# report

Using pandas read csv file. Using replace()function with regular expression to clean data.
```javascript
def load(file_name):
    df_test = pd.read_csv(file_name)
    sLength = len(df_test['id'])
    df_test.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_test = df_test.fillna("miss")
```
