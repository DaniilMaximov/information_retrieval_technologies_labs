import urllib.request
from bs4 import BeautifulSoup
import csv
import time
import logging
import socket
from typing import Optional, Tuple, List, Set

class PersonScraper:
    BASE_URL: str = "https://itmo.ru/ru/personlist/{num}/letter_{num}.htm"
    RETRY_LIMIT: int = 3
    TIMEOUT: int = 3

    def __init__(self, start_range: int = 192, end_range: int = 224):
        self.start_range: int = start_range
        self.end_range: int = end_range
        self.unique_positions: Set[str] = set()
        self.csv_file: str = 'report.csv'
        self._setup_logging()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("scraper.log"),
                logging.StreamHandler()
            ]
        )

    def fetch_page(self, url: str) -> Optional[bytes]:
        for attempt in range(1, self.RETRY_LIMIT + 1):
            try:
                with urllib.request.urlopen(url, timeout=self.TIMEOUT) as response:
                    return response.read()
            except (Exception, socket.timeout) as e:
                logging.error(f"Error fetching {url}: {e}. Attempt {attempt} of {self.RETRY_LIMIT}")
                if attempt == self.RETRY_LIMIT:
                    return None
            self._sleep(0.2)

    def parse_person_list(self, html: bytes) -> Optional[BeautifulSoup]:
        soup = BeautifulSoup(html, 'html.parser')
        return soup.find('div', class_='rowFlex rowFlex--gutters phoneList')

    def parse_person_details(self, person_url: str) -> Tuple[Optional[str], Optional[bool], Optional[List[str]]]:
        person_html = self.fetch_page(person_url)
        if not person_html:
            return None, None, None

        person_soup = BeautifulSoup(person_html, 'html.parser')

        # Извлечение имени
        name_span = person_soup.find('span', class_='page-header-text')
        name = name_span.get_text(strip=True) if name_span else None

        # Извлечение деталей
        details_div = person_soup.find('div', class_='c-personCard-details')
        if not details_div:
            return name, None, None

        # Учёная степень
        academic_degree = details_div.find('dt', string='Ученая степень:') is not None

        # Позиция
        positions = self._extract_positions(details_div)
        return name, academic_degree, positions

    def _extract_positions(self, details_div: BeautifulSoup) -> List[str]:
        positions = []
        positions_dt = details_div.find('dt', string='Должность:')
        if positions_dt:
            positions_dd = positions_dt.find_next_sibling('dd')
            if positions_dd:
                for child in positions_dd.stripped_strings:
                    positions.append(child)
        return positions

    def _initialize_csv(self) -> None:
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['ФИО', 'Количество должностей', 'Учёная степень', 'Ссылка']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def _write_to_csv(self, row_data: dict) -> None:
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['ФИО', 'Количество должностей', 'Учёная степень', 'Ссылка'])
            writer.writerow(row_data)

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def scrape_persons(self) -> None:
        self._initialize_csv()

        for num in range(self.start_range, self.end_range):

            page_url = self.BASE_URL.format(num=num)
            logging.info(f'Processing page: {page_url}')

            html = self.fetch_page(page_url)
            if not html:
                continue

            personalities = self.parse_person_list(html)
            if not personalities:
                logging.warning(f'No person list found on {page_url}')
                continue

            person_links = personalities.find_all('a', href=True)
            for a_tag in person_links:
                person_href = a_tag['href']
                person_url = urllib.request.urljoin(page_url, person_href)
                logging.info(f'Processing person: {person_url}')

                name, academic_degree, positions = self.parse_person_details(person_url)
                if not name:
                    logging.warning(f'Name not found for {person_url}')
                    continue

                num_positions = len(positions) if positions else 0
                self.unique_positions.update(positions or [])

                logging.info(f'{num_positions} positions found for {person_url}')

                # Prepare row for CSV
                row_data = {
                    'ФИО': name,
                    'Количество должностей': num_positions,
                    'Учёная степень': 'True' if academic_degree else 'False',
                    'Ссылка': person_url
                }
                self._write_to_csv(row_data)

                self._sleep(0.1)

            self._sleep(0.2)


if __name__ == '__main__':
    scraper = PersonScraper()
    scraper.scrape_persons()
