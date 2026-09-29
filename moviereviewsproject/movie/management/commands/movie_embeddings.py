import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate and store movie embeddings using Hugging Face Inference API"

    def handle(self, *args, **kwargs):
        load_dotenv('.env')
        load_dotenv('../openAI.env')
        token = os.environ.get('hf_apikey')

        if not token:
            self.stderr.write("No hf_apikey found in environment.")
            return

        client = InferenceClient(token=token)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        movies = list(Movie.objects.all())
        self.stdout.write(f"🚀 Generando embeddings con Hugging Face ({model_name}) para {len(movies)} películas...")

        for idx, movie in enumerate(movies, start=1):
            text_to_embed = f"{movie.title}. {movie.description}"
            try:
                emb = client.feature_extraction(text_to_embed, model=model_name)
                emb_array = np.array(emb, dtype=np.float32)
                movie.emb = emb_array.tobytes()
                movie.save(update_fields=['emb'])
                self.stdout.write(self.style.SUCCESS(f"[{idx}/{len(movies)}] 👌 Embedding guardado para: {movie.title} (dim: {len(emb_array)})"))
            except Exception as e:
                self.stderr.write(f"[{idx}/{len(movies)}] ❌ Error en {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS("🎯 ¡Embeddings de Hugging Face generados para todas las películas!"))
