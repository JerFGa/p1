import os
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Update movie images from media/movie/images folder"

    def handle(self, *args, **kwargs):
        images_folder = os.path.join('media', 'movie', 'images')
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies in database")

        updated_count = 0
        for movie in movies:
            # Expected format: m_NOMBRE_PELICULA.png
            image_filename = f"m_{movie.title}.png"
            full_path = os.path.join(images_folder, image_filename)

            if os.path.exists(full_path):
                movie.image = os.path.join('movie', 'images', image_filename)
                movie.save()
                updated_count += 1
                self.stdout.write(self.style.SUCCESS(f"Updated image for: {movie.title}"))
            else:
                self.stdout.write(self.style.WARNING(f"Image not found for '{movie.title}': looking for {image_filename}"))

        self.stdout.write(self.style.SUCCESS(f"Finished processing. Total updated: {updated_count} movies."))
