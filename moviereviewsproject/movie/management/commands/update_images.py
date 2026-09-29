import os
import requests
from openai import OpenAI
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate images with OpenAI and update movie image field"

    def handle(self, *args, **kwargs):
        # ✅ Load environment variables from the .env file
        load_dotenv('../openAI.env')

        # ✅ Initialize the OpenAI client with the API key
        client = OpenAI(
            api_key=os.environ.get('openai_apikey'),
        )
        # ✅ Folder to save images
        images_folder = 'media/movie/images/'
        os.makedirs(images_folder, exist_ok=True)

        # ✅ Fetch all movies
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        for movie in movies:
            try:
                # ✅ Call the helper function
                image_relative_path = self.generate_and_download_image(client, movie.title, images_folder)

                # ✅ Update database
                movie.image = image_relative_path
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"Saved and updated image for: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {e}")

            # 🔎 Process just the first movie for demonstration
            break

        self.stdout.write(self.style.SUCCESS("Process finished (only first movie updated)."))

    def generate_and_download_image(self, client, movie_title, save_folder):
        """
        Generates an image using OpenAI's DALL·E model and downloads it.
        Returns the relative image path or raises an exception.
        """
        prompt = f"Movie poster of {movie_title}"

        # ✅ Generate image with OpenAI, or Hugging Face if OpenAI quota is exceeded
        try:
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="256x256",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            image_filename = f"m_{movie_title}.png"
            image_path_full = os.path.join(save_folder, image_filename)
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            with open(image_path_full, 'wb') as f:
                f.write(image_response.content)
            return os.path.join('movie/images', image_filename)
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"OpenAI error ({e}). Using Hugging Face FLUX API."))
            hf_token = os.environ.get('hf_apikey')
            if hf_token:
                from huggingface_hub import InferenceClient
                hf_client = InferenceClient(token=hf_token)
                img = hf_client.text_to_image(
                    f"Vintage classic movie poster of {movie_title}",
                    model="black-forest-labs/FLUX.1-schnell"
                )
                image_filename = f"m_{movie_title}.png"
                image_path_full = os.path.join(save_folder, image_filename)
                img.save(image_path_full)
                return os.path.join('movie/images', image_filename)
            raise e
