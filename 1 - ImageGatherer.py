import pandas as pd
import multiprocessing
import json

from pprint import pprint
from google_images_download import google_images_download


def create_forms_dict(df):
    poke_dict = {}
    banned_words = ["-small", "-large", "-super", "-cap", "-cosplay", "-pop-star", "-totem"]
    for index, row in df.iterrows():
        poke = row["identifier"]
        if any(word in poke for word in banned_words):
            continue
        if row["id"] > 807:
            name = poke.split("-")[0]
            if name in poke_dict:
                poke_dict[name].append(poke)
            else:
                if "-" in poke:
                    poke_dict[name] = [poke]
                else:
                    poke_dict[name] = []
        else:
            poke_dict[poke] = []

    with open('pokemon-forms.json', 'w') as fp:
        json.dump(poke_dict, fp)
    return poke_dict


def search_term(poke):
    return " ".join(poke.split("-"))


def process_pokemon_names(df):
    poke_dict = create_forms_dict(df)
    pprint(poke_dict)
    pokes_to_limits = []
    for pokemon, form_list in poke_dict.items():
        if len(form_list) == 0:
            print(pokemon)
            pokes_to_limits.append((pokemon, 200))

        elif len(form_list) == 1:
            pokes_to_limits.append((pokemon, 150))
            pokes_to_limits.append((search_term(form_list[0]), 50))

        elif len(form_list) == 2:
            pokes_to_limits.append((pokemon, 100))
            for form in form_list:
                pokes_to_limits.append((search_term(form), 50))

        elif len(form_list) >= 3:
            for form in form_list:
                pokes_to_limits.append((search_term(form), int(200 / len(form_list))))

    return pokes_to_limits



def get_images_for_pokemon(poke_to_limit):
    pokemon = poke_to_limit[0]
    limit = poke_to_limit[1]
    response = google_images_download.googleimagesdownload()
    response.download(
        {
            "keywords": pokemon + " pokemon",
            "limit": limit,
            "chromedriver": "chromedriver"
            # Add chromedriver to your path or just point this var directly to your chromedriverv
        }
    )


if __name__ == '__main__':
    df = pd.read_csv("pokemon.csv")

    pool = multiprocessing.Pool(multiprocessing.cpu_count() * 3)
    pokes_to_limits = process_pokemon_names(df)
    pool.map(get_images_for_pokemon, pokes_to_limits)
