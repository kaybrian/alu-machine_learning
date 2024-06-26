#!/usr/bin/env python3
"""
    script that prints the location of a specific user:
"""


import requests


def main(url):
    """
    - The user is passed as first argument of the script
    with the full API URL, example: ./2-user_location.py
    https://api.github.com/users/holbertonschool
    - If the user doesn’t exist, print Not found
    - If the status code is 403, print Reset in X min where X
    is the number of minutes from now and the value of
    X-Ratelimit-Reset
    - Your code should not be executed when the file is
    imported (you should use if __name__ == '__main__':)

    """

    response = requests.get(url)
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset = response.headers["X-Ratelimit-Reset"]
        print("Reset in {} min".format(reset))
    else:
        print(response.json()["location"])


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
