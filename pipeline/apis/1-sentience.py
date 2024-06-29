#!/usr/bin/env python3
"""
    Return the list of names of the home
    planets of all sentient species.
"""

import requests


def sentientPlanets():
    '''
    Return the list of names of the home
    planets of all sentient species.
    '''
    planets = []

    try:
        url = 'https://swapi-api.alx-tools.com/api/species/'

        while url:
            response = requests.get(url)
            data = response.json()
            species = data['results']

            for specie in species:
                if (
                    specie['classification'] == 'sentient' or
                    specie['designation'] == 'sentient'
                ):
                    homeworld_url = specie['homeworld']

                    if homeworld_url:
                        homeworld_response = requests.get(homeworld_url)
                        homeworld_response.raise_for_status()
                        homeworld_data = homeworld_response.json()
                        planets.append(homeworld_data['name'])

            url = data['next']

        return planets
    except requests.RequestException as e:
        print('An error occurred: {}'.format(e))
    except Exception as e:
        print('A generale occurred: {}'.format(e))
