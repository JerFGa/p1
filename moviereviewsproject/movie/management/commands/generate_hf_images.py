import os
import time
from django.core.management.base import BaseCommand
from movie.models import Movie
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate AI movie posters for movies using Hugging Face FLUX model"

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=0, help='Number of movies to generate (0 = all)')
        parser.add_argument('--delay', type=float, default=1.0, help='Delay in seconds between requests')

    def handle(self, *args, **options):
        load_dotenv('.env')
        load_dotenv('../openAI.env')
        token = os.environ.get('hf_apikey')

        if not token:
            self.stderr.write("No hf_apikey found in environment.")
            return

        client = InferenceClient(token=token)
        limit = options['limit']
        delay = options['delay']

        images_folder = os.path.join('media', 'movie', 'images')
        os.makedirs(images_folder, exist_ok=True)

        movies = list(Movie.objects.all())
        if limit > 0:
            movies = movies[:limit]

        self.stdout.write(f"🚀 Starting Hugging Face image generation for {len(movies)} movies...")

        model_name = "black-forest-labs/FLUX.1-schnell"
        generated_count = 0

        for idx, movie in enumerate(movies, start=1):
            safe_name = "".join(c for c in movie.title if c.isalnum() or c in (" ", "_", "-")).strip()
            filename = f"m_{safe_name}.png"
            filepath = os.path.join(images_folder, filename)

            # Si ya tiene una imagen grande de HF (> 500 KB), la saltamos para ahorrar cuota
            if os.path.exists(filepath) and os.path.getsize(filepath) > 500 * 1024:
                self.stdout.write(f"[{idx}/{len(movies)}] ⏩ Ya tiene imagen HF: {movie.title}")
                continue

            self.stdout.write(f"[{idx}/{len(movies)}] 🎨 Generando con HF ({model_name}) para '{movie.title}'...")

            prompt = f"Vintage classic movie poster of '{movie.title}', retro cinema illustration, detailed, high resolution"

            try:
                img = client.text_to_image(prompt, model=model_name)
                img.save(filepath)

                rel_path = f"movie/images/{filename}"
                movie.image = rel_path
                movie.save(update_fields=['image'])

                generated_count += 1
                self.stdout.write(self.style.SUCCESS(f"[{idx}/{len(movies)}] ✅ Guardada y actualizada: {movie.title}"))

                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                self.stderr.write(f"[{idx}/{len(movies)}] ❌ Error en {movie.title}: {e}")
                # Si es un límite de tasa (Rate limit / 429), pausar un poco
                if "429" in str(e) or "rate" in str(e).lower():
                    self.stdout.write(self.style.WARNING("⏳ Esperando 10 segundos por límite de peticiones de Hugging Face..."))
                    time.sleep(10)

        self.stdout.write(self.style.SUCCESS(f"🎉 Finalizado. Se generaron {generated_count} imágenes nuevas con Hugging Face."))
