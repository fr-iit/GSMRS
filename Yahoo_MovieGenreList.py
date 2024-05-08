
# Read the file and extract genres
with open("yahoo_movie_dataset/movies.csv", "r") as file:
    lines = file.readlines()

genres = set()
for line in lines[1:]:  # Skip header
    genre = line.strip().split(",")[-1]  # Extract Genre column
    genres.update(genre.split("|"))  # Split genres by "|" and add to set
    #genres.remove("\\N")

# Print unique genre values
print("Unique genres:")
print(len(genres))
for genre in genres:
    print(genre)