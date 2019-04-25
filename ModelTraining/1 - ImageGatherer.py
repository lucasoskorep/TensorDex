import pandas as pd
import multiprocessing

from google_images_download import google_images_download

df = pd.read_csv("pokemon.csv")

response = google_images_download.googleimagesdownload()


def get_images_for_pokemon(pokemon):
    response.download(
        {
            "keywords": pokemon,# + " pokemon",
            "limit": 250,
            "chromedriver": "chromedriver",
            "thumbnail":True
        #     Add chromedriver to your path or just point this var directly to your chromedriver
        }
    )

pool = multiprocessing.Pool(multiprocessing.cpu_count()*4)

pool.map(get_images_for_pokemon, df["identifier"][:490])

