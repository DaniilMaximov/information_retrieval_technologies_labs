import bz2
import os
import re
import xml.etree.ElementTree as ET
import base64
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from pymystem3 import Mystem
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaModel

# Загрузка русских стоп-слов
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

# Инициализация инструментов
mystem = Mystem()
morph = MorphAnalyzer()


# -------------------------- #

def clean_html(raw_html):
    """Удаление HTML-тегов."""
    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup.get_text()


def remove_punctuation(text):
    """Удаление пунктуации и цифр."""
    return re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', '', text)


def lemmatize_with_mystem(text):
    """Лемматизация текста с использованием pymystem3."""
    lemmas = mystem.lemmatize(text)
    return ''.join(lemmas).strip()


def lemmatize_with_pymorphy(text):
    """Лемматизация текста с использованием pymorphy3."""
    return ' '.join([morph.parse(word)[0].normal_form for word in text.split()])


def calculate_stopwords(text):
    """Подсчет количества стоп-слов в тексте."""
    words = text.split()
    return sum(1 for word in words if word in russian_stopwords)


def latin_ratio(text):
    """Подсчет доли латинских слов в тексте."""
    latin_count = sum(1 for word in text.split() if re.match(r'[a-zA-Z]+', word))
    return latin_count / len(text.split()) if len(text.split()) > 0 else 0


# -------------------------- #

def decode_bz2_files(input_path, output_path):
    """Декодирование и разархивация файлов .bz2 в .xml."""
    os.makedirs(output_path, exist_ok=True)
    bz2_files = [file for file in os.listdir(input_path) if file.endswith('.bz2')]

    for bz2_file in bz2_files[:3]:
        bz2_file_path = os.path.join(input_path, bz2_file)

        with open(bz2_file_path, 'rb') as f:
            encoded_data = f.read()

        try:
            decompressed_data = bz2.decompress(encoded_data)
        except OSError:
            print(f"Ошибка разархивации {bz2_file}")
            continue

        # Декодирование содержимого
        try:
            xml_content = decompressed_data.decode('cp1251')
        except UnicodeDecodeError:
            print(f"Ошибка декодирования {bz2_file}")
            continue

        # Удаление описания коллекции
        xml_content = re.sub(
            r'<collection-description>.*?</collection-description>',
            '',
            xml_content,
            flags=re.DOTALL
        )

        # Сохранение декомпрессированного файла XML
        xml_file_name = bz2_file.replace('.bz2', '.xml')
        with open(os.path.join(output_path, xml_file_name), 'w', encoding='utf-8') as xml_file:
            xml_file.write(xml_content)
        print(f"Файл {xml_file_name} успешно разархивирован и декодирован.")


def process_xml_file(file_path):
    """Парсинг и обработка XML документов."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    documents = []

    for doc in root.findall('.//{http://www.romip.ru/data/common}document'):
        doc_id = doc.find('{http://www.romip.ru/data/common}docID').text
        encoded_url = doc.find('{http://www.romip.ru/data/common}docURL').text
        encoded_subject = doc.find('{http://www.romip.ru/data/common}subject').text
        encoded_content = doc.find('{http://www.romip.ru/data/common}content').text

        decoded_url = base64.b64decode(encoded_url).decode('cp1251')
        decoded_subject = base64.b64decode(encoded_subject).decode('cp1251')
        decoded_content = base64.b64decode(encoded_content).decode('cp1251')

        cleaned_content = clean_html(decoded_content)
        documents.append({
            'id': doc_id,
            'url': decoded_url,
            'subject': decoded_subject,
            'content_xml': decoded_content,
            'content': cleaned_content,
            'file_path': file_path
        })
    return documents


def process_decoded_files(decoded_path):
    """Обработка декомпрессированных файлов XML в DataFrame."""
    files = [f for f in os.listdir(decoded_path) if f.endswith('.xml')]
    all_documents = []

    for file in files:
        file_path = os.path.join(decoded_path, file)
        print(f"Обработка файла: {file}")
        all_documents.extend(process_xml_file(file_path))

    df = pd.DataFrame(all_documents)
    print(f"Обработано {len(all_documents)} документов.")
    return df


def analyze_documents(df):
    """Анализ DataFrame с документами."""
    df["full_doc"] = df['subject'] + " " + df['content']

    df['doc_lower'] = df['full_doc'].str.lower().apply(remove_punctuation)

    df['mystem'] = df['doc_lower'].apply(lemmatize_with_mystem)
    df['pymorphy'] = df['doc_lower'].apply(lemmatize_with_pymorphy)

    df['stopword_ratio'] = df['doc_lower'].apply(calculate_stopwords) / df['doc_lower'].str.split().apply(len)
    df['avg_word_len'] = df['doc_lower'].apply(
        lambda text: sum(len(word) for word in text.split()) / len(text.split()) if len(text.split()) > 0 else 0
    )
    df['latin_ratio'] = df['doc_lower'].apply(latin_ratio)

    df['doc_len_words'] = df['full_doc'].apply(lambda x: len(x.split()))
    df['doc_len_bytes'] = df['full_doc'].apply(lambda x: len(x.encode('utf-8')))

    print(f"Средняя длина документа в словах: {df['doc_len_words'].mean()}")
    print(f"Средняя длина документа в байтах: {df['doc_len_bytes'].mean()}")
    print(f"Средняя доля стоп-слов: {df['stopword_ratio'].mean()}")
    print(f"Средняя длина слова: {df['avg_word_len'].mean()}")
    print(f"Доля слов, написанных латиницей: {df['latin_ratio'].mean()}")

    analyze_document_lengths(df)


def analyze_document_lengths(df):
    """Визуализация распределения длины документов."""
    all_docs_len = df['doc_len_words'].tolist()
    all_docs_len_byte = df['doc_len_bytes'].tolist()
    clip_all_docs_len = [num if num <= 1000 else 1000 for num in all_docs_len]
    clip_all_docs_len_byte = [num if num <= 10000 else 10000 for num in all_docs_len_byte]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Распределение длин документов', fontsize=16)

    axes[0, 0].hist(all_docs_len, bins=100, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Длина документов (слова)')
    axes[0, 0].set_xlabel('Количество слов')
    axes[0, 0].set_ylabel('Частота')

    axes[0, 1].hist(all_docs_len_byte, bins=100, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Длина документов (байты)')
    axes[0, 1].set_xlabel('Количество байт')

    axes[1, 0].hist(clip_all_docs_len, bins=100, color='salmon', edgecolor='black')
    axes[1, 0].set_title('Длина документов (слова, с ограничением)')
    axes[1, 0].set_xlabel('Количество слов (с ограничением)')

    axes[1, 1].hist(clip_all_docs_len_byte, bins=100, color='gold', edgecolor='black')
    axes[1, 1].set_title('Длина документов (байты, с ограничением)')
    axes[1, 1].set_xlabel('Количество байт (с ограничением)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def build_tfidf_model(df):
    """Построение TF-IDF модели."""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['mystem'])

    def search_query(query, top_n=3):
        query = lemmatize_with_mystem(remove_punctuation(query.lower()))
        query_vec = tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[::-1][:top_n]
        for i in top_indices:
            print(f"Документ ID: {i}, Сходство: {cosine_similarities[i]}")
            print(df.iloc[i]['full_doc'])

    return search_query


def build_lda_model(df):
    """Построение LDA модели."""
    texts = df["mystem"].apply(str.split).tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=10)

    for idx, topic in lda_model.print_topics(-1):
        print(f"Тема {idx}: {topic}")

    return lda_model


# -------------------------- #

if __name__ == "__main__":
    # Пути к файлам
    input_path = './news2006'
    output_path = './news2006_decoded'

    # 1 Разархивация
    decode_bz2_files(input_path, output_path)

    # 2 Расшифровка
    df = process_decoded_files(output_path)

    # 3: Анализ документа
    analyze_documents(df)

    # 4 tfidf и lda
    search = build_tfidf_model(df)
    lda_model = build_lda_model(df)

    # Пример запросов
    print("\nПоиск по запросу 'выборы':")
    search("выборы")
