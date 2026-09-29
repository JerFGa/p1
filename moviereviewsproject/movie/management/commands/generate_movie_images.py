import os
import urllib.parse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Generate AI movie posters using AI image generation API and update database"

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=0, help='Limit number of movies to process (0 = all)')
        parser.add_argument('--workers', type=int, default=6, help='Concurrent worker threads')

    def handle(self, *args, **options):
        limit = options['limit']
        workers = options['workers']

        images_folder = os.path.join('media', 'movie', 'images')
        os.makedirs(images_folder, exist_ok=True)

        movies = list(Movie.objects.all())
        if limit > 0:
            movies = movies[:limit]

        self.stdout.write(f"🎨 Generating AI images for {len(movies)} movies using {workers} concurrent workers...")

        def process_movie(m):
            safe_name = "".join(c for c in m.title if c.isalnum() or c in (" ", "_", "-")).strip()
            filename = f"m_{safe_name}.png"
            full_path = os.path.join(images_folder, filename)

            if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
                rel_path = f"movie/images/{filename}"
                if m.image.name != rel_path:
                    m.image = rel_path
                    m.save(update_fields=['image'])
                return f"✅ Already exists: {m.title}"

            prompt = urllib.parse.quote(f"Vintage movie poster of {m.title}, cinematic, retro illustration")
            url = f"https://image.pollinations.ai/prompt/{prompt}?width=256&height=256&nologo=true"

            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    with open(full_path, 'wb') as f:
                        f.write(resp.content)
                    m.image = f"movie/images/{filename}"
                    m.save(update_fields=['image'])
                    return f"🎨 Generated & updated: {m.title}"
                else:
                    return f"⚠️ API returned status {resp.status_code} for {m.title}"
            except Exception as e:
                return f"❌ Failed for {m.title}: {e}"

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_movie = {executor.submit(process_movie, m): m for m in movies}
            completed = 0
            for future in as_completed(future_to_movie):
                completed += 1
                result = future.result()
                self.stdout.write(f"[{completed}/{len(movies)}] {result}")

        self.stdout.write(self.style.SUCCESS("🎉 Finished generating AI movie posters!"))
