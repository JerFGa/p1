from django.shortcuts import render
from django.http import HttpResponse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Create your views here.

from .models import Movie

def home(request):
    #return HttpResponse("<h1>Welcome to the Home page</h1>")
    #return render(request, 'home.html', {'name': 'Jeremias Figueroa Garcia'})
    searchTerm = request.GET.get('searchMovie')
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm': searchTerm, 'movies': movies})

def about(request):
    return render(request, 'about.html')

def statistics(request):
    movies = Movie.objects.all()

    # Movies per year
    movie_counts_by_year = {}
    for movie in movies:
        if movie.year:
            movie_counts_by_year[movie.year] = movie_counts_by_year.get(movie.year, 0) + 1

    years = sorted(movie_counts_by_year.keys())
    year_counts = [movie_counts_by_year[year] for year in years]

    plt.figure(figsize=(10, 5))
    plt.bar([str(y) for y in years], year_counts)
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graphic_year = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Movies per genre (first genre of each movie)
    movie_counts_by_genre = {}
    for movie in movies:
        if movie.genre:
            genre = movie.genre.split(',')[0].strip()
            movie_counts_by_genre[genre] = movie_counts_by_genre.get(genre, 0) + 1

    genres = sorted(movie_counts_by_genre.keys(), key=lambda g: movie_counts_by_genre[g], reverse=True)
    genre_counts = [movie_counts_by_genre[genre] for genre in genres]

    plt.figure(figsize=(10, 5))
    plt.bar(genres, genre_counts)
    plt.title('Movies per genre')
    plt.xlabel('Genre')
    plt.ylabel('Number of movies')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graphic_genre = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return render(request, 'statistics.html', {
        'graphic_year': graphic_year,
        'graphic_genre': graphic_genre,
    })

def recommendations(request):
    import os
    import numpy as np
    from huggingface_hub import InferenceClient
    from dotenv import load_dotenv

    recommended_movie = None
    similarity_score = None
    error_message = None
    prompt = request.GET.get('prompt', '').strip()

    if prompt:
        load_dotenv('.env')
        load_dotenv('../openAI.env')
        token = os.environ.get('hf_apikey')

        def cosine_similarity(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        try:
            client = InferenceClient(token=token)
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            raw_emb = client.feature_extraction(prompt, model=model_name)
            prompt_emb = np.array(raw_emb, dtype=np.float32)

            best_movie = None
            max_similarity = -1.0

            for movie in Movie.objects.all():
                if movie.emb:
                    movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                    if len(prompt_emb) == len(movie_emb):
                        similarity = cosine_similarity(prompt_emb, movie_emb)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_movie = movie

            recommended_movie = best_movie
            similarity_score = round(max_similarity, 4)

        except Exception as e:
            error_message = str(e)

    return render(request, 'recommendations.html', {
        'prompt': prompt,
        'recommended_movie': recommended_movie,
        'similarity_score': similarity_score,
        'error_message': error_message
    })