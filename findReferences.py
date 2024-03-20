import fitz
from PIL import ImageDraw, Image
import re
import numpy as np
from searchReferences import rechercher_mot_cle_dans_pdf



class mainArticle:

    def __init__(self, pdfPath):
        self.doc = fitz.open(pdfPath)
        self.names = self.doc.resolve_names()
        self.refPages = list(range(self.findPage("References")[-1].number, len(self.doc)))
        self.links = None
        self.ref = None
        

    def display(self, page_number, rectangles=[], dpi=300):
        
        page = self.doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Get the annotations (rectangles) on the page
        draw = ImageDraw.Draw(img)

        for i, rect in enumerate(rectangles):
            x0, y0, x1, y1 = rect
            x0 *= dpi / 72  # Convert from points to pixels
            y0 *= dpi / 72
            x1 *= dpi / 72
            y1 *= dpi / 72
            draw.rectangle([(x0, y0), (x1, y1)], outline="red")
            draw.text((x0, y0), str(i), fill="red")  # Display the number of the rectangle

        img.show()


    def extractLinks(self):
        links = []
        for page in self.doc:
            for link in page.get_links():
                links.append(Link(link, self, page.number))
        self.links = links
        return links
    

    def findPage(self, word):
        pages = []
        for page in self.doc:
            text = page.get_text()
            if word in text:
                pages.append(page)
        return pages
    

    def findReferencesLinks(self):
        references = []
        for link in self.links:
            if link.isRef(self, self.refPages):
                references.append(link)

        self.ref = references
        i = 0
        while i < len(references):
            if references[i].text[-2] == ")":
                references[i].text = references[i].text[:-1]
            elif references[i].text[-1] not in {";", ")"}:
                references[i].text += " " + references[i+1].text
                del references[i+1]
            else:
                i += 1
        return references
    

    def findReferences(self):
            """
            This method finds the references in the article by extracting the relevant text.
            
            Returns:
            - refs (list): A list of references found in the article.
            """
            words = []
            for page in article.refPages:
                words += article.doc[page].get_text("words") # Get the words of the page
            reference = True
            words = words[::-1]
            wordsRef = []
            i = 0
            while reference: # Check only the words after "References"
                if "References" in words[i]:
                    reference = False
                    break
                else:
                    wordsRef.append(words[i])
                    i += 1
            wordsRef = wordsRef[::-1]
            linesRef = [list(wordsRef[0])]
            for i, word in enumerate(wordsRef[1:]): # Bring together the words which are on the same line, based on the height
                if np.isclose(word[3], linesRef[-1][3], atol=0.5):
                    linesRef[-1][4] += " " + word[4]
                else:
                    linesRef[-1][2:4] = wordsRef[i][2:4]
                    linesRef.append(list(word))

            refs = [linesRef[0]] # Bring together the lines which are in the same paragraph, based on the indentation
            for i, line in enumerate(linesRef[1:]):
                if line[0] > refs[-1][0] + 1 and line[1] > linesRef[i][1] - 1:
                    refs[-1][4] += " " + line[4]
                else:
                    refs[-1][3] = linesRef[i][3]
                    refs.append(line)
            for ref in refs:
                ref[4] = re.sub(r'- ', '', ref[4])
            self.articleRef = refs
            return refs


    def findstr(self, str1, str2):
        if str1 not in str2:
            print(f"{str1} not in {str2}")
            return None
        else:
            for i in range(len(str2)):
                if str2[i:i+len(str1)] == str1:
                    return i


    def correspondanceLinkRef(self, link):
        txt = link.text
        txt = re.sub(r'[^a-zA-Z0-9\sÀ-ÿ-]', ' ', txt)
        txt = re.sub(r'\n', ' ', txt)
        txt = re.sub(r'et al', '', txt)
        txt = re.sub(r' and ', ' ', txt)
        txt = txt.split(" ")
        txt_copy = txt.copy()  # Create a copy of the set
        for elt in txt_copy:  # Iterate over the copy
            if len(elt) < 3:
                txt.remove(elt)
        potentialRefs = []
        for ref in self.articleRef:
            _ = True
            for elt in txt:
                if elt not in ref[4]:
                    _ = False
            if _:
                potentialRefs.append(ref)
        if len(potentialRefs) == 1:
            return potentialRefs[0]
        elif len(potentialRefs) == 0:
            return None
        return min(potentialRefs, key=lambda x: self.findstr(txt[0], x[4]))



class Link:
    def __init__(self, link, article, page):
        self.kind = link["kind"]
        if self.kind == 1:
            self.page = link["page"]
        elif self.kind == 4:
            self.nameddest = link["nameddest"]
        elif self.kind == 2:
            self.uri = link["uri"]
        self.originePage = page
        self.rect = link["from"]
        self.text = self.findLinkText(article)


    def isRef(self, article, refPages):
        if self.kind == 1:
            if self["page"] in refPages:
                return True
        elif self.kind == 4:
            names = article.names
            if names[self.nameddest]["page"] in refPages:
                return True
    

    def dealWithDest(self, dest: str):
        coordinate = dest.split(" ")[1:]
        for i in range(4):
            coordinate[i] = float(coordinate[i])
        return coordinate
    

    def findLinkText(self, article, marge=0.3):
        """find the text of the link, using the coordinates of the link"""

        rect = self.rect

        rect = (rect[0] - marge, rect[1]+0.3, rect[2] + marge, rect[3]-0.3)
        
        txt = article.doc[self.originePage].get_textbox(rect)

        return txt



if __name__ =="__main__":
    pdfPath = "DataBase\Biodegradable plastics can alter carbon and nitrogen cycles to a greater extent than conventional plastics in marine sediment.pdf"
    article = mainArticle(pdfPath)

    refs = article.findReferences()
    article.extractLinks()
    article.findReferencesLinks()

    rectangles = []
    for ref in article.articleRef[:34]:
        rectangles.append(ref[:4])

    article.display(5, rectangles=rectangles)