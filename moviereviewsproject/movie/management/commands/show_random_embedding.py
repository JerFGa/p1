import random
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Display the embeddings of a randomly selected movie"

    def handle(self, *args, **kwargs):
        movies = list(Movie.objects.all())
        if not movies:
            self.stderr.write("No movies found in the database.")
            return

        movie = random.choice(movies)
        embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)

        self.stdout.write(self.style.SUCCESS(f"🎬 Película seleccionada al azar: '{movie.title}'"))
        self.stdout.write(f"📏 Dimensión del vector de embedding: {len(embedding_vector)}")
        self.stdout.write(f"✨ Elementos no nulos: {np.count_nonzero(embedding_vector)} / {len(embedding_vector)} (Norma L2: {np.linalg.norm(embedding_vector):.4f})")
        non_zeros = embedding_vector[embedding_vector != 0]
        self.stdout.write(f"🔢 Muestra de valores no nulos: {non_zeros[:10]}")
        self.stdout.write("\n🔢 Vector completo (resumen):")
        self.stdout.write(str(embedding_vector))
