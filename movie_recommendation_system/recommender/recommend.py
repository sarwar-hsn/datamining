import os 
import ast
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def convert(obj):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(obj)])
    except ValueError:
        return ""

def convert_cast(obj):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(obj)[:3]])
    except ValueError:
        return ""

def get_director(obj):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director'])
    except ValueError:
        return ""

# Load datasets
current_dir = os.path.dirname(os.path.abspath(__file__))

def train_and_save():
    progress_bar = st.progress(0,text="Loading dataset")
    credit_file = os.path.join(current_dir, 'tmdb_5000_credits.csv')
    movies_file = os.path.join(current_dir, 'tmdb_5000_movies.csv')

    #creating pandas data frame
    init_movies = pd.read_csv(movies_file)
    init_credit = pd.read_csv(credit_file)

    movies = init_movies.merge(init_credit,left_on='id',right_on='movie_id')
    movies = movies.rename(columns={'title_x': 'title'})

    progress_bar.progress(10,text="Feature Selection")
    #feature selection
    movies = movies[['id','title','overview','genres','keywords','cast','crew']]

    #Saving this state
    movies.to_pickle(os.path.join(current_dir, 'movies.pkl'))

    progress_bar.progress(15,text="Feature generation")
    #feature generation
    movies.dropna(inplace=True)
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: " ".join(x.split()))

    # Concatenating features into tags
    movies['tags'] = movies['title'] + " " + movies['overview'] + " " + movies['genres'] + " " + movies['keywords'] + " " + movies['cast'] + " " + movies['crew']
    movies['title'] = movies['title'].apply(lambda x: x.lower())
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())


    progress_bar.progress(50,text="TF-IDF")
    # Create DataFrame for TF-IDF
    df = movies[['id', 'title', 'tags']]
    # Compute TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["tags"].fillna(''))
    #print(tfidf_matrix)
    
    progress_bar.progress(75,text="Computing Cosine Similarity")
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    #print(cosine_sim)

    progress_bar.progress(90,text="Saving")
    #Save dataframe
    df.to_pickle(os.path.join(current_dir, 'df.pkl'))
    # Save the matrix to a file for later use
    np.save(os.path.join(current_dir, 'cosine_sim_matrix.npy'), cosine_sim)
    progress_bar.progress(100,text="Done.")

def load_model():
    try:
        df = pd.read_pickle(os.path.join(current_dir, 'df.pkl'))
        cosine_sim = np.load(os.path.join(current_dir, 'cosine_sim_matrix.npy'), allow_pickle=True)
        return df, cosine_sim
    except Exception as e:
        print(f"Error at loading: {e}")
        return None, None

    

def get_movie_index(title, df):
    all_titles = df['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return df[df.title == closest_match[0]].index[0]


def make_recommendations(movie_title,top_n=10):
    movie_title=movie_title.lower()
    # Loading models
    df,cosine_sim = load_model()
    if df is not None and cosine_sim is not None:
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        # Fuzzy matching to correct misspelled titles
        title = process.extractOne(movie_title, df['title'])[0]
        idx = indices[title]
        #if multiple indices returned and select the first one if so
        if type(idx) == pd.Series:
            idx = idx.iloc[0]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx].flatten())) # Flatten array here
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:11]] #returning first 10 movies
        return df[['title', 'id']].iloc[movie_indices].to_dict(orient='records')
    else:
        return None
    


#statistics
def plot_genre_distribution():
    movies = pd.read_pickle(os.path.join(current_dir, 'movies.pkl'))
    # Drop NA values and reset index
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    # Function to parse and extract genre names
    def extract_genre_names(genres_str):
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list if 'name' in genre]
    # Apply parsing function to genres column
    movies['genres'] = movies['genres'].apply(extract_genre_names)
    # Explode genres into separate rows
    df = movies.explode('genres')
    # Count the occurrences of each genre
    genre_counts = df['genres'].value_counts()
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(genre_counts.index, genre_counts.values)
    plt.xlabel('Genres')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.title('Genre Distribution')
    st.pyplot(plt)


def plot_cosine_similarity_distribution(movie_title):
    df, cosine_sim = load_model()
    
    # Check if the movie is in the dataset
    if movie_title.lower() in df['title'].values:
        idx = df.index[df['title'] == movie_title.lower()].tolist()[0]

        # Extract similarity scores for the movie
        sim_scores = cosine_sim[idx]

        # Exclude the movie itself from the comparison
        sim_scores = np.delete(sim_scores, idx)

        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sim_scores, bins=30, edgecolor='black')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Cosine Similarity Scores for "{movie_title}"')
        st.pyplot(plt)
    else:
        st.error('Movie not found in dataset.')


def show_top_tfidf_terms(n_terms=20):
    df, _ = load_model()
    df = df.copy()
    
    # Combine all texts into one large string
    combined_text = " ".join(df['tags'].fillna(""))

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([combined_text])
    feature_array = np.array(tfidf.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray().flatten())[::-1]

    top_terms = feature_array[tfidf_sorting][:n_terms]
    top_scores = np.sort(tfidf_matrix.toarray().flatten())[::-1][:n_terms]

    # Create DataFrame and display it as a table
    top_terms_df = pd.DataFrame({
        "Term": top_terms,
        "TF-IDF Score": top_scores
    })

    st.table(top_terms_df)
    

def plot_tfidf_scores_distribution():
    
    df, _ = load_model()
    df = df.copy()
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'].fillna(''))
    scores = tfidf_matrix.toarray().flatten()
    scores = scores[scores > 0]

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, edgecolor='black')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of TF-IDF Scores Across All Movies')
    st.pyplot(plt)
