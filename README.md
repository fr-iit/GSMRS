# A Closer Look on Gender Stereotypes in Movie Recommender Systems and Their Implications with Privacy

## Abstract

The movie recommender system typically leverages user feedback to provide personalized recommendations that align with user preferences and increase business revenue. This study investigates the impact of gender stereotypes on such systems through a specific attack scenario. In this scenario, an attacker determines users' gender, a private attribute, by exploiting gender stereotypes about movie preferences and analyzing users' feedback data, which is either publicly available or observed within the system. The study consists of two phases. In the first phase, a user study involving 630 participants identified gender stereotypes associated with movie genres, which often influence viewing choices. In the second phase, four inference algorithms were applied to detect gender stereotypes by combining the findings from the first phase with users' feedback data. Results showed that these algorithms performed more effectively than relying solely on feedback data for gender inference. Additionally, we quantified the extent of gender stereotypes to evaluate their broader impact on digital computational science. The latter part of the study utilized two major movie recommender datasets: MovieLens 1M and Yahoo!Movie. Detailed experimental information is available on our GitHub repository: https://github.com/fr-iit/GSMRS.


## Setup

You must have the MovieLens (https://grouplens.org/datasets/movielens/) & yahoo movie data (https://webscope.sandbox.yahoo.com/) downloaded in your project.

## Gender Inference

The basic gender inference attack can be executed by running the Classifier_Gender.py. In the file, you can define what exactly should be executed. I.e.,
* To load the rating and gender data of MovieLens, set the value of the 'dataset' variable as dataset = 'ml1m' and for Yahoo!Movie dataset as dataset = 'yahoo'
* To execute the gender classifier on rating with gender stereotypes data, set the value of the 'FristFoldRQ2' variable as FristFoldRQ2 = 'Rating with GS'
* To execute the gender classifier on only rating data, set the value of the 'FristFoldRQ2' variable as FristFoldRQ2 = 'Only Rating' 
* To run logistic regression in cross-validation function, set the value of the 'SecondFoldRQ2' variable as SecondFoldRQ2 = 'LR', and for support vector machine, set the value of the same variable as SecondFoldRQ2 = 'SVM' 



