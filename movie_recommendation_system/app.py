import os
import streamlit as st
from recommender import *
import pandas as pd
import requests

# TMDB API KEY
api_key = '5300f87bceddc5aa6cc3df7622ef68f4'

def get_poster(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US")
    data = response.json()
    try:
        return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
    except:
        return None

def add_posters(movies):
    for movie in movies:
        movie['poster'] = get_poster(movie.get('id'))

# Initialize session states
if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = None
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = None

# Load movie list
current_dir = os.path.dirname(os.path.abspath(__file__))
movie_list = pd.read_pickle(os.path.join(current_dir, './recommender/df.pkl'))
movie_list = movie_list['title'].values


# App title
st.title("GTU Movie Recommender")

# Movie selection dropdown
movie_choice = st.selectbox("Enter A Movie To Get Recommendation", movie_list)

# Update the selected movie in session state
if movie_choice:
    st.session_state['selected_movie_name'] = movie_choice

# Column layout for buttons
recommend,stat, reload_model,  = st.columns(3)

# Button definitions
def set_view_recommend():
    st.session_state['current_view'] = 'recommend'

def set_view_reload_model():
    st.session_state['current_view'] = 'reload_model'

def set_view_stat():
    st.session_state['current_view'] = 'stat'

# Button layout
with recommend:
    st.button('Recommend', on_click=set_view_recommend)

with stat:
    st.button('Statistics', on_click=set_view_stat)

with reload_model:
    st.button('Reload Model', on_click=set_view_reload_model)


if st.session_state['current_view'] == 'recommend':
    # Display recommendations
    if st.session_state['selected_movie_name']:
        movies = make_recommendations(st.session_state['selected_movie_name'])
        add_posters(movies)
        one,two,three,four,five= st.columns(5)
        with one:
            if movies[0]['poster']:
                st.text(movies[0]['title'].capitalize())
                st.image(movies[0]['poster'])
        with two:
            if movies[1]['poster']:
                st.text(movies[1]['title'].capitalize())
                st.image(movies[1]['poster'])
        with three:
            if movies[2]['poster']:
                st.text(movies[2]['title'].capitalize())
                st.image(movies[2]['poster'])
        with four:
            if movies[3]['poster']:
                st.text(movies[3]['title'].capitalize())
                st.image(movies[3]['poster'])
        with five:
            if movies[4]['poster']:
                st.text(movies[4]['title'].capitalize())
                st.image(movies[4]['poster'])
        one,two,three,four,five= st.columns(5)
        with one:
            if movies[5]['poster']:
                st.text(movies[5]['title'].capitalize())
                st.image(movies[5]['poster'])
        with two:
            if movies[6]['poster']:
                st.text(movies[6]['title'].capitalize())
                st.image(movies[6]['poster'])
        with three:
            if movies[7]['poster']:
                st.text(movies[7]['title'].capitalize())
                st.image(movies[7]['poster'])
        with four:
            if movies[8]['poster']:
                st.text(movies[8]['title'].capitalize())
                st.image(movies[8]['poster'])
        with five:
            if movies[9]['poster']:
                st.text(movies[9]['title'].capitalize())
                st.image(movies[9]['poster'])

    st.session_state['selected_movie_name'] = None
    st.session_state['current_view'] = None
    


elif st.session_state['current_view'] == 'reload_model':
    with st.spinner('Reloading model...'):
        train_and_save()


elif st.session_state['current_view'] == 'stat':
    if st.session_state['selected_movie_name']:
        movie_title = st.session_state['selected_movie_name']
        plot_cosine_similarity_distribution(movie_title)

    plot_genre_distribution()
    show_top_tfidf_terms(n_terms=20)

    with st.spinner('Loading TF-IDF Distribution...'):
        plot_tfidf_scores_distribution()

    st.session_state['selected_movie_name'] = None
    st.session_state['current_view'] = None

