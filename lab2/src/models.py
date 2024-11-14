from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaModel


class TfidfModel:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, query, top_n=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        return top_indices, similarities[top_indices]


class LdaTopicModel:
    def __init__(self, documents, num_topics=10):
        texts = [doc.split() for doc in documents]
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.model = LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=10)

    def display_topics(self):
        for idx, topic in self.model.print_topics(-1):
            print(f"Topic {idx}: {topic}")
