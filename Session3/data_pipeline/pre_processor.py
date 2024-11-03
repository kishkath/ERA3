# pre-processing the input data
import spacy

# Load the English model
nlp = spacy.load('en_core_web_sm')


def text_processing(text_data):
    # Process text and tokenize using spaCy
    doc = nlp(text_data.lower())
    tokens = [token.text for token in doc]
    return tokens


if __name__ == "__main__":
    # Example text data
    text = """Semiconductors are materials that have electrical conductivity between that of a conductor (like copper) and an insulator (like glass), allowing them to control electric current in devices."""
    print(text_processing(text))
