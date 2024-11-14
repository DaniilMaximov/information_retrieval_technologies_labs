from decoder import BZ2Decoder
from processor import XMLProcessor
from analyzer import TextAnalyzer
from models import TfidfModel, LdaTopicModel


def main():
    input_path = './news2006'
    output_path = './news2006_decoded'

    # Step 1: Decode files
    decoder = BZ2Decoder(input_path, output_path)
    decoder.decode_files()

    # Step 2: Process XML
    processor = XMLProcessor(output_path)
    df = processor.process_all_files()

    # Step 3: Analyze text
    analyzer = TextAnalyzer()
    df = analyzer.analyze_documents(df)
    analyzer.plot_document_lengths(df)

    # Step 4: Build and use models
    tfidf_model = TfidfModel(df['mystem'])
    top_indices, similarities = tfidf_model.search("выборы")
    print(f"Top search results: {top_indices}")

    lda_model = LdaTopicModel(df['mystem'])
    lda_model.display_topics()


if __name__ == "__main__":
    main()
