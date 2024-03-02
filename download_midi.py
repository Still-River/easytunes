import requests
from bs4 import BeautifulSoup
import os
import random

def get_midi_file_locations(url):
    try:
        payload = {
            'search_title': None,
            'search_genre': None,
            'search_composer': None,
            'search_submitter': None,
            'search_artist': None,
            'search_decade': None,
            'search_rating': -1,
            'search_difficulty': -1,
            'search.x': random.randint(1,80),
            'search.y': random.randint(1,16),
            'search_results': 5000
        }

        r=requests.post(url, data=payload)
        soup = BeautifulSoup(r.text, 'html.parser')

        links = soup.find_all('a')
        hrefs = []

        for link in links:
            href = f"http://www.ambrosepianotabs.com{link.get('href')}"
            if href and href.endswith('.mid'):
                hrefs.append(href)

        return hrefs

    except requests.RequestException as e:
        print(f"Error: {e}")

def download_file(url, folder_path):
    local_filename = os.path.join(folder_path, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"\tDownloaded {local_filename}")

def download_midi_files(hrefs, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    error_hrefs = []
    for href in hrefs:
        i = hrefs.index(href) + 1
        try:
            download_file(href, folder_path)
            print(f"{i}/{len(hrefs)} - Downloaded {href}")

        except requests.RequestException as e:
            print(f"Error: {e}")
            print(f"Could not download {href}")
            error_hrefs.append(href)

    print(f"Could not download {len(error_hrefs)} files")
    return error_hrefs

if __name__ == '__main__':
    url = 'http://www.ambrosepianotabs.com/page/library'
    folder_path = './ambrose_dataset/'
    hrefs = get_midi_file_locations(url)
    error_hrefs = download_midi_files(hrefs, folder_path)