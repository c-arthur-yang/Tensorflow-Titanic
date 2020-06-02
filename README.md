# Titanic-Tensorflow

Titanic: Machine Learning from Disaster is a typical and classical dataset on Kaggle.

I will briefly introduce the datasets we have, for more details please visit: https://www.kaggle.com/c/titanic/data

We got two datasets as train.csv and test.csv which can be used to train our model and test our model respectively, finally we need to use our model to predict weather the passanger was survived or not with a given Id in the test.csv.

In the train.csv, some of the features are missing. It forces us to fill the missing values and deal with NaN carefully.

In the test.csv, the feature "Survived" is hidden, we will predict their status based on the features given in the file.
