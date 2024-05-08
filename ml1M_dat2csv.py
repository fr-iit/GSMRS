import pandas as pd

# Read dat file & convert the data in comma seperated csv format
ratings=pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, names=['UserID','MovieID','Rating','Timestamp'],engine='python')
#ratings.to_csv('ml-1m/ratings.csv', index=False)

rat = pd.read_csv('ml-1m/ratings.csv')
print(rat.head())

# reading movie.dat
#movies = pd.read_csv("ml-1m/movies.dat", sep=",", header= None, encoding='latin-1', engine='python')
#print(movies.head())
#movies.to_csv("ml-1m/movies_mod.dat", sep=':', index=False)

# reading data from movies.dat
movies= pd.read_csv('ml-1m/movies.csv')
#print(movie.head())

# read user.dat
users=pd.read_csv('ml-1m/users.dat', sep='::', header=None, names=['UserID','Gender','Age','Occupation', 'Zip-code'],engine='python')
#users.to_csv('ml-1m/users.csv', index=False)

user= pd.read_csv('ml-1m/users.dat', sep='::', header=None, names=['UserID','Gender','Age','Occupation', 'Zip-code'],engine='python')
#print(user.head())
#user = pd.read_csv('ml-1m/users.csv')
merge_rating = pd.merge(ratings[['UserID','MovieID','Rating']], movies[['MovieID', 'Genres']], on='MovieID')
merge_rating = pd.merge(merge_rating, user[['UserID','Gender']], on='UserID')
#print(merge_rating.head())
print(len(merge_rating))
#merge_rating.to_csv('ml-1m/merge_rating.dat', sep=':', index=False, header=None)