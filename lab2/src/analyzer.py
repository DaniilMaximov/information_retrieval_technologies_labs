import re
import matplotlib.pyplot as plt
from pymystem3 import Mystem
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class TextAnalyzer:
    def __init__(self):
        self.mystem = Mystem()
        self.morph = MorphAnalyzer()
        self.stopwords = set(stopwords.words('russian'))

    @staticmethod
    def remove_punctuation(text):
        return re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', '', text)

    def lemmatize_with_mystem(self, text):
        return ''.join(self.mystem.lemmatize(text)).strip()

    def lemmatize_with_pymorphy(self, text):
        return ' '.join([self.morph.parse(word)[0].normal_form for word in text.split()])

    def calculate_stopwords(self, text):
        words = text.split()
        return sum(1 for word in words if word in self.stopwords)

    @staticmethod
    def latin_ratio(text):
        latin_count = sum(1 for word in text.split() if re.match(r'[a-zA-Z]+', word))
        return latin_count / len(text.split()) if text.split() else 0

    def analyze_documents(self, df):
        df['doc_lower'] = df['content'].str.lower().apply(self.remove_punctuation)
        df['mystem'] = df['doc_lower'].apply(self.lemmatize_with_mystem)
        df['pymorphy'] = df['doc_lower'].apply(self.lemmatize_with_pymorphy)
        df['stopword_ratio'] = df['doc_lower'].apply(self.calculate_stopwords) / df['doc_lower'].str.split().apply(len)
        df['latin_ratio'] = df['doc_lower'].apply(self.latin_ratio)
        return df

    @staticmethod
    def plot_document_lengths(df):
        lengths = df['content'].str.split().apply(len)
        plt.hist(lengths, bins=50, color='blue', alpha=0.7)
        plt.title('Document Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.show()
