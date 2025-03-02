import requests
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep

# Base URL et range des années
url_base = "https://papers.nips.cc"
urls_index = [(1987, "https://papers.nips.cc/book/neural-information-processing-systems-1987")]
for i in range(1, 32):
    year = i + 1987
    urls_index.append((year, f"https://papers.nips.cc/book/advances-in-neural-information-processing-systems-{i}-{year}"))

# Fonction pour avoir les URLs des papiers
def get_paper_urls():
    urls_papers = []
    for year, url_page in urls_index:
        try:
            response = requests.get(url_page)
            response.raise_for_status()  # vérifier si la requete a réussi
            soup = BeautifulSoup(response.text, "html.parser")
            
            divs = soup.findAll('div', {'class': 'main-container'})
            for div in divs:
                uls = div.find_all('ul', attrs={'class': None})
                for ul in uls:
                    for li in ul.find_all('li'):
                        a = li.find('a')
                        if a and 'href' in a.attrs:
                            urls_papers.append((year, url_base + a['href']))
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url_page}: {e}")
        sleep(1)  # Eviter beaucoup de requetes consecutives
    return urls_papers

# Fonction pour extraire les paper details (title, abstract)
def get_paper_details(url_paper):
    try:
        response_paper = requests.get(url_paper)
        response_paper.raise_for_status()
        soup_paper = BeautifulSoup(response_paper.text, "html.parser")
        
        title = soup_paper.find('h2', attrs={'class': 'subtitle'})
        title = title.contents[0] if title else "No title found"

        abstract = soup_paper.find('p', attrs={'class': 'abstract'})
        abstract = abstract.contents[0] if abstract else "No abstract available"
        
        return title, abstract
    except requests.exceptions.RequestException as e:
        print(f"Error fetching paper details from {url_paper}: {e}")
        return None, None

# Fonction principale pour scraper et sauvegarder les données
def scrape_nips_papers():
    urls_papers = get_paper_urls()
    nips = []

    for year, url_paper in urls_papers:
        title, abstract = get_paper_details(url_paper)
        if title and abstract:
            nips.append((year, title, abstract))
        sleep(1)  

    # stocker les données sur nips.csv
    df = pd.DataFrame(nips, columns=["Year", "Title", "Abstract"])
    df.to_csv("nips.csv", index=False)
    print("Data saved to nips.csv")


if __name__ == "__main__":
    scrape_nips_papers()
