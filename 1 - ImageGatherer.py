import pandas as pd
import multiprocessing
import json

from pprint import pprint
from google_images_download import google_images_download

total_per = 10
form_increment = 1


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
        print(pokemon)
        num_forms = len(form_list)
        if num_forms == 0:
            pokes_to_limits.append((pokemon, total_per))

        elif num_forms == 1:
            pokes_to_limits.append((pokemon, total_per - form_increment))
            pokes_to_limits.append((search_term(form_list[0]), form_increment))

        elif num_forms == 2:
            pokes_to_limits.append((pokemon, total_per - form_increment * num_forms))
            for form in form_list:
                pokes_to_limits.append((search_term(form), form_increment))

        elif num_forms >= 3:
            revised_increment = int(total_per / len(form_list))
            for form in form_list:
                pokes_to_limits.append((pokemon, total_per - revised_increment * num_forms))

                pokes_to_limits.append((search_term(form), revised_increment))

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
