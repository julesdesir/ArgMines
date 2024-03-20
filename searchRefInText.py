import fitz



def loadPDF(pdfPath):
    txt = ""
    with fitz.open(pdfPath) as pdf_document:
        for page in pdf_document:
            txt += page.get_text()
    return txt



def keyWordsSearch(keywords, sentences):
    scores = dict()
    for sentence in sentences.split('\n'):
        score = 0
        for word in keywords.keys():
            if word in sentence:
                score += keywords[word]
        scores[sentence] = score

    if len(scores) == 0:
            return None
    
    mean = sum(scores.values())/len(scores)
    std = (sum([(x[1] - mean)**2 for x in scores.items()])/len(scores))**0.5


    thresh = mean + 2*std

    sortedScores = {key: value for key, value in scores.items() if value > thresh}
    sortedScores = sorted(sortedScores.items(), key=lambda x: x[1], reverse=True)

    return sortedScores



if __name__ == "__main__":
    pdfPath = "DataBase/1-s2.0-S0146638001001620-main.pdf"
    txt = loadPDF(pdfPath)
    keywords = [('TerrDOC', 0.17679711868177392), ('oxidative', 0.051713744785368075), ('released', 0.049836646221072256), ('understood', 0.04681933757897415), ('soils', 0.04469710887191001), ('reasons', 0.04469710887191001), ('breakdown', 0.03925458323010001), ('fully', 0.028829119453833187), ('winter', 0.025838818914995815), ('spring', 0.023175421549996012), ('dominated', 0.023175421549996012), ('yet', 0.023175421549996012), ('summer', 0.022348554435955004), ('mainly', 0.013567999947384097), ('whereas', 0.012459161555268067), ('derived', 0.009889392880686937), ('temperature', 0.006269392755851004), ('shown', 0.005442525641809994), ('during', 0.0027511096133010803), ('high', 0.0016827529630963761), ('been', 0.0016827529630963761), ('has', 0.0006592359119004736)]
    print(keyWordsSearch(keywords, txt)[:10])