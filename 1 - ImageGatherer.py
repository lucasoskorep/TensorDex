import pandas as pd
import multiprocessing

from google_images_download import google_images_download


def get_images_for_pokemon(pokemon):
    response = google_images_download.googleimagesdownload()
    response.download(
        {
            "keywords": pokemon + " pokemon",
            "limit": 250,
            "chromedriver": "chromedriver",
            "thumbnail": True
            #     Add chromedriver to your path or just point this var directly to your chromedriverv
        }
    )


    # freeze_support()
    df = pd.read_csv("pokemon.csv")

    pool = multiprocessing.Pool(multiprocessing.cpu_count()*3)
    fixes = []
    pool.map(get_images_for_pokemon, [fixes])#df["identifier"]

    # for pokemon in df["identifier"][:490]:
    #     get_images_for_pokemon(pokemon)