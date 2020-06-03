# Tensorflow-Titanic

Titanic: Machine Learning from Disaster is a typical and classical dataset on Kaggle.

I will briefly introduce the datasets we have, for more details please visit: https://www.kaggle.com/c/titanic/data

We got two datasets as train.csv and test.csv which can be used to train our model and test our model respectively, finally we need to use our model to predict weather the passanger was survived or not with a given Id in the test.csv.

In the train.csv, some of the features are missing. It forces us to fill the missing values and deal with NaN carefully.

In the test.csv, the feature "Survived" is hidden, we will predict their status based on the features given in the file.


I use two schemes to make the prediction.

In decision_tree.py, I use Decision Tree to predict the survived passangers, which got a reasonable result.

I also use TensorFlow to build a 3-layers NN to make the decision.
