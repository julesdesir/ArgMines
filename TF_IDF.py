import numpy as np
import re



def train_tfidf(corpus):
    idf = {}
    total_documents = len(corpus)
    for document in corpus:
        unique_words = set(document.split())
        for word in unique_words:
            idf[word] = idf.get(word, 0) + 1

    for word in idf:
        idf[word] = np.log(total_documents / (idf[word] + 1))

    with open("idf.txt", "w") as f:
        for word in idf:
            if word.isalpha() and word.isascii():
                try:
                    f.write(f"{word} {idf[word]}\n")
                except:
                    print(f"Error with word {word} and value {idf[word]}")

    return idf



def load_idf():
    idf = {}
    with open("idf.txt", "r") as f:
        for line in f:
            word, value = line.split()
            idf[word] = float(value)
    return idf



def tfidf(document, idf):
    tfidf = dict()
    words = document.split()
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        tfidf[word] = tfidf.get(word, 0) + 1
    for word in tfidf:
        tfidf[word] = tfidf[word] / len(words) * idf.get(word, 0)
    return tfidf



def findKeyWords(query):
    idf = load_idf()
    query_tfidf = tfidf(query, idf)
    sorted_query_tfidf = sorted(query_tfidf.items(), key=lambda x: x[1], reverse=True)
    keywords = [(word, tfidf) for word, tfidf in sorted_query_tfidf if tfidf > 0]
    return keywords



if __name__ == "__main__":
    import os
    import fitz

    def read_data_folder(folder_path):
        corpus = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                with fitz.open(os.path.join(folder_path, filename)) as pdf_document:
                    text = ""
                    for page in pdf_document:
                        text += page.get_text()
                    corpus.append(text)
        return corpus
    
    corpus = read_data_folder("DataBase/")
    idf = load_idf()

    query = "TerrDOC released in summer from soils during high temperature has been shown to be mainly derived from oxidative breakdown of lignin, whereas TerrDOC released in winter and spring is dominated by carbohydrates; the reasons for this are not yet fully understood"
    print(findKeyWords(query))