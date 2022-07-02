# A gentle introduction of building Machine Learning Pipeline

## Overview

In this project, we are going to put our machine learning knowledge together and structure a machine learning project to model a Natural Language Processing (NLP) using **Random Forest** in a more readable and maintabe way. This tutorial focus on addressing the structure of a ETL and model pipeline with a real-world dataset. This includes following questions to address:

- What is ETL?
- How to construct a model pipeline to improve the readability of your model?
- How to encode grid search within your model pipeline to improve model accuracy?



## Introduction

In this tutorial, we are using a **Corporate Message** dataset from [**data.world**](https://data.world/crowdflower/corporate-messaging) to study a classification problem. Given a post on social media, how can we identify the post contains corporate information, calling for a vote or just reply to certain user? We are going to cover how text can be used in training a machine learning model. For a detailed explaination of the workthrough, please refer to the notebook.



First, let's have a look of what data is stored inside the Coroporate Message dataset. 

![Screenshot 2022-07-02 at 18.45.44](/Users/austincheang/Library/Application Support/typora-user-images/Screenshot 2022-07-02 at 18.45.44.png)

This dataset contains 3118 rows and 11 attributes. The dataset contain a lot of attributes and not sure which attributes are useful for our model. Since we only focus on predicting the category of a post, we would only focus on `text` and `cateogry` columns when building our model.`text` stores each post posted by the company in `screenname`. `category` provides the label for the category label for the post. 



## What is ETL?

ETL stands for Extract, Tranform and Load. When working with a real-world problem, sometimes we have to pull the dataset from different sources. This usual includes:

- Pull data from multiple tables in database
- Requests from different API
- Work with multiple files such as CSV, Json or even XML

In the Extract phrase, we gather all the data from different sources together and identify what attributes are useful to complete our table.



In the Transform phase, this requires a lot of data enigneering skills such as 

- Data Cleaning
- Remove duplicated data and outliers
- Impute data for missing values
- Create dummy variable or One-hot encoding
- Perform Scaling and Standarlisation for continuous data

Performing transformation on the data is crucial especailly for unstructured data. This makes our data is more unified between each rows and boost the machine learning model accuracy. 



In Load phrase, this requires data to store into a proper data warehouse such as database for modelling use. By automating the Extract and Transform phrase, we can have new data directly go through the previous steps from unstructured and sparse data into structured data. New data will be load/add into the database and ready to use.



In this project, we are working with only one data source and the dataset is well structured. We can simply filter the data which we actually want to include when training a model. `category:confidence` shows the confidence level of category where `1.0` is very confident to `0.0` depicts not confident at all. We are interested in predicting `Information`, `Action` and `Dialogue` with `1.0` confidence level. Therefore, rows which do not meet this requirement are excluded in our data frame. Since this project is small and simply, we assume the data is loaded from a database.



## How to construct a model pipeline to improve the readability of your model?

After loading the dataset, we might want to perform other transfomration steps to process how the data is going to be used in the model. `text` contains plain text that cannot direcly feed into the machine learning model. We have to apply traditional text preprocessing steps:

1. Normalisation - Convert data into case-insensitive form

2. Tokenisation - Break the sentence into words

3. Remove stop words (Optional) - Remove common meaningless words (E.g. at, the, be)

4. Stemming/lemmatisation - Convert word into root form



Next, there are basic NLP concepts such as **Bags of words** and **TF-IDF** to convert the text data to work in various model (Please refer to the notebook for a detailed explanation). Sklearn library also provides the above transfomer object to save user time. 

```Python
vect = CountVectorizer(tokenizer=preprocessing)
tfidf = TfidfTransformer()
classifier = RandomForestClassifier()

# Train classifier
# X_train_counts = vect.fit_transform(X_train)
# X_train_tfidf = tfidf.fit_transform(X_train_counts)
# clf.fit(X_train_tfidf, y_train)
```



Without knowing what is a model pipeline, user declares each transformer objects and call each of them when training the classifier. However, Sklearn provides `Pipeline` object to make the code more clean and maintable. We are wrap the above transformer objects into a `Pipeline` object and the model will automatically run the objects linearly.

```Python
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=preprocessing)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier()),
])
```

Compared the two versions of transfomer objects, user can easily understand what are the transfomation steps and reduce the redundant `fit_transform` and `fit` for every object. Adding additional transformation step becomes as simple as adding one line of code in a centralised place.



## How to encode grid search within your model pipeline to improve model accuracy?

To improve the model accuracy, trying out different models and parameters are evitable. Usually, user declared a dictionary of the parameters for all the possible combinations using `GridSearchCV` from Sklearn.

```Python
    parameters = {
        "vect__ngram_range": ((1,1), (1,2)),
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__max_features": (None, 100, 200),
        "tfidf__use_idf": (True, False),
        "classifier__n_estimators": [50, 100, 200]
    }
```

We can wrap the `parameters` and `pipeline` into a method such that all the things are executed at once and grouping samiliar items into same group.

```Python
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=preprocessing)),
        ('tfidf', TfidfTransformer()),
        ('classifier', RandomForestClassifier()),
    ])

    parameters = {
        "vect__ngram_range": ((1,1), (1,2)),
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__max_features": (None, 100, 200),
        "tfidf__use_idf": (True, False),
        "classifier__n_estimators": [50, 100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv
```

When every place is ready, we can simply perform the `fit` method of the classifier like what we used to do.

```Python
model = build_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```



------

Congratulations, you used learnt how to structure the a model pipeline in a organised way with Grid Search 

along side. By using Random Forest with different text transfomer objects, we obtained a 94.80% after trying out different parameters in Grid Search. The result is promising and now we have a model to automatically identify posts category for us.

![output](/Users/austincheang/Desktop/Corporate_Message/output.png)





## Summary

You have learnt what is ETL and why ETL is important before creating the machine learning model. Companies would have ETL pipeline streaming on cloud service such as Amazon Web Services. Furthermore, Model Pipeline makes transformation easy to organise and waive redundant code. Therefore, try using `Pipeline` in your next machine learning project!