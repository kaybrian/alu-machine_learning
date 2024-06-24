#!/usr/bin/env python3
'''
    Return the list of names of the home
    planets of all sentient species.
'''


import requests


def sentientPlanets():
    """
    Return the list of names of the home
    planets of all sentient species.
    """

    url = "https://swapi-api.alx-tools.com/api/species/"
    planets = []
    while url:
        response = requests.get(url)
        data = response.json()
        for planet in data["results"]:
            if planet["homeworld"] != "unknown":
                planets.append(planet["homeworld"])
        url = data["next"]
    return planets
