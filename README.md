# A Closer Look on Gender Stereotypes in Movie Recommender Systems and Their Implications with Privacy

## Setup:

You must have the MovieLens (https://grouplens.org/datasets/movielens/) & yahoo movie data (https://webscope.sandbox.yahoo.com/) downloaded in your project.

## Gender Inference

The basic gender inference attack can be executed by running the Classifier_Gender.py. In the file, you can define what exactly should be executed. I.e.,
* To load the rating and gender data of MovieLens, set the value of the 'dataset' variable as dataset = 'ml1m' and for Yahoo!Movie dataset as dataset = 'yahoo'
* To execute the gender classifier on rating with gender stereotypes data, set the value of the 'FristFoldRQ2' variable as FristFoldRQ2 = 'Rating with GS'
* To execute the gender classifier on only rating data, set the value of the 'FristFoldRQ2' variable as FristFoldRQ2 = 'Only Rating' 
* To run logistic regression in cross-validation function, set the value of the 'SecondFoldRQ2' variable as SecondFoldRQ2 = 'LR', and for support vector machine, set the value of the same variable as SecondFoldRQ2 = 'SVM' 



