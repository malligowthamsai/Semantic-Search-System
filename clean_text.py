import re

def clean_text(text):

    text = text.lower()

    # remove common email header lines
    text = re.sub(r'^(from|subject|organization|lines|path|xref|newsgroups):.*$', '', text, flags=re.MULTILINE)

    # remove URLs
    text = re.sub(r'http\S+', '', text)

    # remove special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()