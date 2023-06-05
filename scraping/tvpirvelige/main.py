import csv
import random
from datetime import datetime
from time import sleep
from urllib.parse import urljoin

import cloudscraper
from bs4 import BeautifulSoup


scraper = cloudscraper.create_scraper()
file_path = 'tvpirvelige.csv'

with open(file_path, "w", encoding="utf-8",  newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['date', 'title', 'content_url', 'content'])

def load_and_save(loader):
  new_row = list(loader.values())
  with open(file_path, "a", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(new_row)

def scrape_page(url):
  response = scraper.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  items = soup.find_all('div', class_='col-md-6')[:24]
  for item in items:
    loader = {}
    #scrape category page
    date_string = item.find('div', class_='tvpcard__date').find('span').text.strip()
    loader['date'] = datetime.strptime(date_string, "%d.%m.%y %H:%M")
    loader['title'] = item.find('h3', class_='tvpcard__title').text.strip()
    loader['content_url'] = urljoin(url, item.find('div').find('a')['href'])

    #scrape content page
    sleep(random.uniform(0.1, 0.6))
    response = scraper.get(loader['content_url'])
    soup = BeautifulSoup(response.text, 'html.parser')
    content_element = soup.find_all('div', class_='page__usercontent')
    if isinstance(content_element, list) and len(content_element) >= 2:
      loader['content'] = content_element[1].get_text().strip()
    else:
      loader['content'] = None

    #save
    load_and_save(loader)
    

urls=[
    ["https://tvpirveli.ge/ka/siaxleebi/politika?p=",910]
]

for url, n_pages in urls:
    for page in range(510, n_pages):
        full_url = url + str(page)
        scrape_page(full_url)
        print(full_url)