import pandas as pd

from google_images_download import google_images_download

df = pd.read_csv("pokemon.csv")

response = google_images_download.googleimagesdownload()

for pokemon in df["identifier"][:251]:
    absolute_image_paths = response.download(
        {
            "keywords": pokemon,
            "limit": 250,
            "chromedriver": "/usr/lib/chromium-browser/chromedriver"
        }
    )
