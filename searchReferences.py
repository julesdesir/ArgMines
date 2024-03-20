import fitz  # PyMuPDF
import os
import re
from references import formateString
from TF_IDF import load_idf, tfidf, findKeyWords


def rechercher_mot_cle_dans_pdf(chemin_dossier, titre):
    # List to store the names of the files found
    fichiers_trouves = dict()

    keyWords = tfidf(titre, load_idf())
    print(keyWords)

    scores = dict()

    # Browse all the files in the folder
    for fichier in os.listdir(chemin_dossier):
        if fichier.endswith(".pdf"):
            scores[fichier] = 0
            chemin_complet = os.path.join(chemin_dossier, fichier)

            # Open the pdf file
            pdf_document = fitz.open(chemin_complet)

            # Check if all the key words are in the document
            for mot_cle in keyWords:


                # Search in the first page of the document
                page = pdf_document[0]
                texte_page = page.get_text()
                texte_page = formateString(texte_page)

                # Check if the key word is in the page
                if mot_cle in texte_page or mot_cle in fichier:
                    scores[fichier] += keyWords[mot_cle]
        pdf_document.close()

    mean = sum(scores.values())/len(scores)
    std = (sum([(x[1]-mean)**2 for x in scores.items()])/len(scores))**0.5
    argmax_score = max(scores, key=scores.get)
    max_score = scores[argmax_score]

    if max_score > mean + 2*std:
        return argmax_score

    else:
        return None



if __name__ == "__main__":
    txt = "Piedecausa, M.A., Aguado-Giménez, F., Cerezo Valverde, J., Hernández Llorente, M.D., García-García, B., 2012. Inﬂuence of ﬁsh food and faecal pellets on short-term oxygen"

    # Example of use
    chemin_dossier_pdf = "DataBase"

    resultat_recherche = rechercher_mot_cle_dans_pdf(chemin_dossier_pdf, txt)

    if resultat_recherche:
        print("Fichier trouvé :", resultat_recherche)
    else:
        print("Aucun fichier trouvé.")