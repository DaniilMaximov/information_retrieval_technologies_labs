import os
import base64
import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup


class XMLProcessor:
    def __init__(self, decoded_path):
        self.decoded_path = decoded_path

    @staticmethod
    def clean_html(raw_html):
        soup = BeautifulSoup(raw_html, 'html.parser')
        return soup.get_text()

    def process_single_file(self, file_path):
        documents = []
        tree = ET.parse(file_path)
        root = tree.getroot()

        for doc in root.findall('.//{http://www.romip.ru/data/common}document'):
            doc_id = doc.find('{http://www.romip.ru/data/common}docID').text
            try:
                decoded_data = {
                    'id': doc_id,
                    'url': base64.b64decode(doc.find('{http://www.romip.ru/data/common}docURL').text).decode('cp1251'),
                    'subject': base64.b64decode(doc.find('{http://www.romip.ru/data/common}subject').text).decode('cp1251'),
                    'content_xml': base64.b64decode(doc.find('{http://www.romip.ru/data/common}content').text).decode('cp1251'),
                }
                decoded_data['content'] = self.clean_html(decoded_data['content_xml'])
                documents.append(decoded_data)
            except Exception as e:
                print(f"Error decoding document {doc_id}: {e}")

        return documents

    def process_all_files(self):
        all_documents = []
        files = [f for f in os.listdir(self.decoded_path) if f.endswith('.xml')]

        for file_name in files:
            file_path = os.path.join(self.decoded_path, file_name)
            print(f"Processing file: {file_name}")
            all_documents.extend(self.process_single_file(file_path))

        return pd.DataFrame(all_documents)
