import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Compare two movies and optionally a prompt using Hugging Face embeddings"

    def handle(self, *args, **kwargs):
        load_dotenv('.env')
        load_dotenv('../openAI.env')
        token = os.environ.get('hf_apikey')

        client = InferenceClient(token=token)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Películas seleccionadas de la base de datos
        movie1 = Movie.objects.get(title="Carmencita")
        movie2 = Movie.objects.get(title="Frankenstein")

        def get_embedding(text):
            emb = client.feature_extraction(text, model=model_name)
            return np.array(emb, dtype=np.float32)

        def cosine_similarity(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        emb1 = get_embedding(f"{movie1.title}. {movie1.description}")
        emb2 = get_embedding(f"{movie2.title}. {movie2.description}")

        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(self.style.SUCCESS(f"🎬 Similaridad entre '{movie1.title}' y '{movie2.title}': {similarity:.4f}"))

        prompt = "película de baile flamenco y superación personal"
        prompt_emb = get_embedding(prompt)

        sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"📝 Similitud prompt ('{prompt}') vs '{movie1.title}': {sim_prompt_movie1:.4f}")
        self.stdout.write(f"📝 Similitud prompt ('{prompt}') vs '{movie2.title}': {sim_prompt_movie2:.4f}")
