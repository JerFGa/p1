import os
from PIL import Image, ImageDraw, ImageFont
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Generate beautiful custom retro movie posters for all movies and update image fields"

    def handle(self, *args, **kwargs):
        images_dir = os.path.join('media', 'movie', 'images')
        os.makedirs(images_dir, exist_ok=True)

        movies = list(Movie.objects.all())
        self.stdout.write(f"🎨 Generating posters for {len(movies)} movies...")

        # Genre color schemes (Gradient top, Gradient bottom, Accent color)
        GENRE_THEMES = {
            'horror': ((25, 5, 10), (60, 10, 15), (220, 50, 50)),
            'drama': ((15, 20, 35), (35, 45, 75), (212, 175, 55)),
            'fantasy': ((20, 10, 35), (55, 20, 80), (180, 120, 240)),
            'comedy': ((35, 25, 10), (80, 55, 20), (255, 185, 45)),
            'documentary': ((25, 25, 25), (45, 45, 45), (200, 200, 200)),
            'animation': ((10, 30, 40), (25, 70, 90), (70, 200, 230)),
            'action': ((30, 15, 10), (70, 30, 15), (240, 90, 40)),
            'default': ((20, 22, 28), (40, 44, 55), (212, 175, 55))
        }

        def get_theme(genre_str):
            if not genre_str:
                return GENRE_THEMES['default']
            g = genre_str.lower()
            for key in GENRE_THEMES:
                if key in g:
                    return GENRE_THEMES[key]
            return GENRE_THEMES['default']

        W, H = 320, 480

        for movie in movies:
            safe_name = "".join(c for c in movie.title if c.isalnum() or c in (" ", "_", "-")).strip()
            filename = f"m_{safe_name}.png"
            filepath = os.path.join(images_dir, filename)

            top_col, bot_col, accent = get_theme(movie.genre)

            # Create vertical gradient image
            img = Image.new("RGB", (W, H))
            draw = ImageDraw.Draw(img)

            for y in range(H):
                ratio = y / float(H)
                r = int(top_col[0] * (1 - ratio) + bot_col[0] * ratio)
                g = int(top_col[1] * (1 - ratio) + bot_col[1] * ratio)
                b = int(top_col[2] * (1 - ratio) + bot_col[2] * ratio)
                draw.line([(0, y), (W, y)], fill=(r, g, b))

            # Draw decorative border
            border_pad = 14
            draw.rectangle(
                [border_pad, border_pad, W - border_pad, H - border_pad],
                outline=accent,
                width=2
            )
            inner_pad = border_pad + 4
            draw.rectangle(
                [inner_pad, inner_pad, W - inner_pad, H - inner_pad],
                outline=(accent[0]//2, accent[1]//2, accent[2]//2),
                width=1
            )

            # Corner ornaments
            for cx, cy in [(inner_pad, inner_pad), (W - inner_pad, inner_pad),
                           (inner_pad, H - inner_pad), (W - inner_pad, H - inner_pad)]:
                draw.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=accent)

            # Header Banner
            draw.text((W // 2, 35), "★ CLASSIC CINEMA ★", fill=accent, anchor="mm")
            draw.line([(W // 2 - 80, 48), (W // 2 + 80, 48)], fill=accent, width=1)

            # Central decorative box for title
            center_y = H // 2 - 20
            draw.rectangle(
                [border_pad + 15, center_y - 70, W - border_pad - 15, center_y + 70],
                outline=accent,
                fill=(0, 0, 0, 120),
                width=1
            )

            # Wrap movie title
            words = movie.title.split()
            lines = []
            curr = ""
            for w in words:
                test = f"{curr} {w}".strip()
                if len(test) <= 18:
                    curr = test
                else:
                    if curr:
                        lines.append(curr)
                    curr = w
            if curr:
                lines.append(curr)

            # Draw title lines
            total_lines_height = len(lines) * 26
            start_y = center_y - (total_lines_height // 2)
            for i, line in enumerate(lines):
                draw.text(
                    (W // 2, start_y + i * 26 + 10),
                    line.upper(),
                    fill=(255, 255, 255),
                    anchor="mm"
                )

            # Film Reel / Star Icon decoration below title
            draw.text((W // 2, center_y + 100), "✦  ❖  ✦", fill=accent, anchor="mm")

            # Year & Genre Footer
            year_text = f"YEAR: {movie.year}" if movie.year else "VINTAGE EDITION"
            genre_text = (movie.genre.split(',')[0].strip().upper()) if movie.genre else "FEATURE FILM"

            draw.text((W // 2, H - 75), genre_text, fill=(230, 230, 230), anchor="mm")
            draw.text((W // 2, H - 50), year_text, fill=accent, anchor="mm")
            draw.text((W // 2, H - 30), "UNIVERSAL ARCHIVE", fill=(150, 150, 150), anchor="mm")

            img.save(filepath, "PNG")

            # Update movie in DB
            rel_path = f"movie/images/{filename}"
            movie.image = rel_path
            movie.save(update_fields=['image'])

        self.stdout.write(self.style.SUCCESS(f"🎉 Successfully created and linked posters for all {len(movies)} movies!"))
