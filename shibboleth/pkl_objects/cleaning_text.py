

def cleaning_text(text):
    import nltk
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.snowball import SnowballStemmer
    import re
    """
    Removes punctuation, converts all characters to lowercase, stems

    Args:
        a single string of text

    Returns:
        processed text string

    """
    tokens = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer('english')

    token = tokens.tokenize(text)
    filtered_words = [word for word in token]
    stems = [stemmer.stem(t) for t in filtered_words]
    stemmed_text = " ".join(stems)
    stemmed_text_list = stemmed_text.split()
    return stemmed_text_list