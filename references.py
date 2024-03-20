import fitz
import re

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
        self.context = self.findContext(article)


    def isRef(self, article, refPages):
        if self.kind == 1:
            if self["page"] in refPages:
                return True
        elif self.kind == 4:
            names = article.names
            if names[self.nameddest]["page"] in refPages:
                return True
            

    def isInternal(self):
        return self.kind in {1, 4}
    

    def dealWithDest(self, dest: str):
        coordinate = dest.split(" ")[1:]
        for i in range(4):
            coordinate[i] = float(coordinate[i])
        return coordinate
    

    def isInt(self):
        try:
            int(self.text)
            return True
        except:
            return False
    

    def findLinkText(self, article, marge=0.5):
        """Find the text of the link, using the coordinates of the link"""

        rect = self.rect

        rect = (rect[0] + marge, rect[1] + marge, rect[2] - marge, rect[3])
        
        txt = article[self.originePage].get_textbox(rect)

        return txt
    
    
    def findContext(self, article, marge=0.5, sizeContext=15):

        rect = self.rect
        rect = (rect[0] + marge - sizeContext, rect[1] + marge, rect[2] - marge + sizeContext, rect[3])

        txt = article[self.originePage].get_textbox(rect)
        if "\n" in txt:
            txt = txt.split("\n")
            for elt in txt:
                if self.text in elt:
                    return elt
            return txt[0]
        
        return txt
    


def formateString(string):
    string = string.strip()

    string = re.sub(r'ﬂ', "fl", string)
    string = re.sub(r"ﬂ", "fl", string)
    string = re.sub(r'ﬁ', "fi", string)
    string = re.sub(r'ﬀ', "ff", string)
    string = re.sub(r'ﬃ', "ffi", string)
    string = re.sub(r'ﬄ', "ffl", string)
    string = re.sub(r'ﬅ', "ft", string)
    string = re.sub(r'ﬆ', "st", string)
    string = re.sub(r'č', "c", string)
    string = re.sub(r'ć', "c", string)
    string = re.sub(r'š', "s", string)
    string = re.sub(r'ž', "z", string)
    string = re.sub(r'Š', "S", string)
    string = re.sub(r'Ž', "Z", string)
    string = re.sub(r'Č', "C", string)
    string = re.sub(r'Ć', "C", string)
    string = re.sub(r'Ÿ', "Y", string)
    string = re.sub(r'ÿ', "y", string)
    string = re.sub(r'Š', "S", string)
    string = re.sub(r'Ž', "Z", string)
    string = re.sub(r'Œ', "OE", string)
    string = re.sub(r'œ', "oe", string)
    string = re.sub(r'æ', "ae", string)
    string = re.sub(r'Æ', "AE", string)
    string = re.sub(r'ß', "ss", string)
    string = re.sub(r'ℎ', "h", string)
    string = re.sub(r'ℏ', "h", string)
    string = re.sub(r'∫', "integral", string)
    string = re.sub(r'∮', "integral", string)
    string = re.sub(r'∬', "integral", string)
    string = re.sub(r'∭', "integral", string)
    string = re.sub(r'∯', "integral", string)
    string = re.sub(r'∰', "integral", string)
    string = re.sub(r'∱', "integral", string)
    string = re.sub(r'∲', "integral", string)
    string = re.sub(r'∳', "integral", string)
    string = re.sub(r'∴', "therefore", string)
    string = re.sub(r'∵', "because", string)
    string = re.sub(r'⁄', "/", string)
    string = re.sub(r"‐|–|─|⎯|‑", "-", string)
    string = re.sub(r"‘|’", "'", string)
    string = re.sub(r"“|”", '"', string)
    string = re.sub(r"…", "...", string)
    string = re.sub(r"⁎", "*", string)
    string = re.sub(r"′|″", lambda m: "'" if m.group() == "′" else '"', string)
    string = re.sub(r"−", "-", string)
    string = re.sub(r"⋯", "...", string)
    string = re.sub(r"Δ", "Delta", string)
    string = re.sub(r"∂", "delta", string)
    string = re.sub(r"μ", "mu", string)
    string = re.sub(r"α", "alpha", string)
    string = re.sub(r"Σ", "Sigma", string)
    string = re.sub(r"ρ", "rho", string)
    string = re.sub(r"σ", "sigma", string)
    string = re.sub(r"∑", "Sigma", string)
    string = re.sub(r"τ", "tau", string)
    string = re.sub(r"ω", "omega", string)
    string = re.sub(r"∆", "Delta", string)
    string = re.sub(r"Γ", "Gamma", string)
    string = re.sub(r"γ", "gamma", string)
    string = re.sub(r"β", "beta", string)
    string = re.sub(r"η", "eta", string)
    string = re.sub(r"ξ", "xi", string)
    string = re.sub(r"ζ", "zeta", string)
    string = re.sub(r"Υ", "Upsilon", string)
    string = re.sub(r"ε", "epsilon", string)
    string = re.sub(r"ϕ", "phi", string)
    string = re.sub(r"Ε", "Epsilon", string)
    string = re.sub(r"Ω", "Omega", string)
    string = re.sub(r"𝜏", "tau", string)
    string = re.sub(r"Φ", "Phi", string)
    string = re.sub(r"φ", "phi", string)
    string = re.sub(r"ɵ", "theta", string)
    string = re.sub(r"∇", "nabla", string)
    string = re.sub(r"Ψ", "Psi", string)
    string = re.sub(r"ψ", "psi", string)
    string = re.sub(r"Π", "Pi", string)
    string = re.sub(r"π", "pi", string)
    string = re.sub(r"δ", "delta", string)
    string = re.sub(r"κ", "kappa", string)
    string = re.sub(r"λ", "lambda", string)
    string = re.sub(r"Λ", "Lambda", string)
    string = re.sub(r"Θ", "Theta", string)
    string = re.sub(r"θ", "theta", string)
    string = re.sub(r"∅", "empty", string)
    string = re.sub(r"υ", "upsilon", string)
    string = re.sub(r"∈", "in", string)
    string = re.sub(r"∩", "intersection", string)
    string = re.sub(r"Ł", "L", string)
    string = re.sub(r"∞", "infinity", string)
    string = re.sub(r" ", " ", string)
    string = re.sub(r"◦|˚", "°", string)
    string = re.sub(r"✉", "email", string)
    string = re.sub(r"∗", "*", string)
    string = re.sub(r"◀|▶|●|◭|◮|∙||￿||||", "", string)
    string = re.sub(r"ı", "i", string)
    string = re.sub(r"ź", "z", string)
    string = re.sub(r"ń", "n", string)
    string = re.sub(r"ł", "l", string)
    string = re.sub(r"ś", "s", string)
    string = re.sub(r"Ś", "S", string)
    string = re.sub(r"Ź", "Z", string)
    string = re.sub(r"Ń", "N", string)
    string = re.sub(r"Ł", "L", string)
    string = re.sub(r"Ą", "A", string)
    string = re.sub(r"ą", "a", string)
    string = re.sub(r"Ę", "E", string)
    string = re.sub(r"ę", "e", string)
    string = re.sub(r"Ó", "O", string)
    string = re.sub(r"ó", "o", string)
    string = re.sub(r"Ć", "C", string)
    string = re.sub(r"ć", "c", string)
    string = re.sub(r"Ż", "Z", string)
    string = re.sub(r"ż", "z", string)
    string = re.sub(r"Ź", "Z", string)
    string = re.sub(r"ź", "z", string)
    string = re.sub(r"≲", "<~", string)
    string = re.sub(r"≥", ">=", string)
    string = re.sub(r"≤", "<=", string)
    string = re.sub(r"∓", "-+", string)
    string = re.sub(r"⇒", "implies", string)
    string = re.sub(r"∼|≈", "~", string)
    string = re.sub(r"≠", "!=", string)
    string = re.sub(r"≡", "==", string)
    string = re.sub(r"∀", "for all", string)
    string = re.sub(r"I", "I", string)
    string = re.sub(r"∝", "proportional", string)
    string = re.sub(r"√", "sqrt", string)
    string = re.sub(r"˙", ".", string)
    string = re.sub(r"ϵ", "epsilon", string)
    string = re.sub(r"Ω", "Omega", string)
    string = re.sub(r"𝑎", "a", string)
    string = re.sub(r"𝑏", "b", string)
    string = re.sub(r"𝑐", "c", string)
    string = re.sub(r"𝑑", "d", string)
    string = re.sub(r"𝑒", "e", string)
    string = re.sub(r"𝑓", "f", string)
    string = re.sub(r"𝑔", "g", string)
    string = re.sub(r"𝑖", "i", string)
    string = re.sub(r"𝑗", "j", string)
    string = re.sub(r"𝑘", "k", string)
    string = re.sub(r"𝑙", "l", string)
    string = re.sub(r"𝑚", "m", string)
    string = re.sub(r"𝑛", "n", string)
    string = re.sub(r"𝑜", "o", string)
    string = re.sub(r"𝑝", "p", string)
    string = re.sub(r"𝑞", "q", string)
    string = re.sub(r"𝑟", "r", string)
    string = re.sub(r"𝑠", "s", string)
    string = re.sub(r"𝑡", "t", string)
    string = re.sub(r"𝑢", "u", string)
    string = re.sub(r"𝑣", "v", string)
    string = re.sub(r"𝑤", "w", string)
    string = re.sub(r"𝑥", "x", string)
    string = re.sub(r"𝑦", "y", string)
    string = re.sub(r"𝑧", "z", string)
    string = re.sub(r"𝐴", "A", string)
    string = re.sub(r"𝐵", "B", string)
    string = re.sub(r"𝐶", "C", string)
    string = re.sub(r"𝐷", "D", string)
    string = re.sub(r"𝐸", "E", string)
    string = re.sub(r"𝐹", "F", string)
    string = re.sub(r"𝐺", "G", string)
    string = re.sub(r"𝐻", "H", string)
    string = re.sub(r"𝐼", "I", string)
    string = re.sub(r"𝐽", "J", string)
    string = re.sub(r"𝐾", "K", string)
    string = re.sub(r"𝐿", "L", string)
    string = re.sub(r"𝑀", "M", string)
    string = re.sub(r"𝑁", "N", string)
    string = re.sub(r"𝑂", "O", string)
    string = re.sub(r"𝑃", "P", string)
    string = re.sub(r"𝑄", "Q", string)
    string = re.sub(r"𝑅", "R", string)
    string = re.sub(r"𝑆", "S", string)
    string = re.sub(r"𝑇", "T", string)
    string = re.sub(r"𝑈", "U", string)
    string = re.sub(r"𝑉", "V", string)
    string = re.sub(r"𝑊", "W", string)
    string = re.sub(r"𝑋", "X", string)
    string = re.sub(r"𝑌", "Y", string)
    string = re.sub(r"𝑍", "Z", string)
    string = re.sub(r'-\n', "", string)
    string = re.sub(r'\n', " ", string)
    string = re.sub(r'   ', " ", string)
    string = re.sub(r'  ', " ", string)

    return string



def findElementsBetweenParentheses(string):
    elements = []
    open = False
    element = ""

    for caractere in string:
        if caractere == "(":
            open = True
        elif caractere == ")":
            open = False
            elements.append(element)
            element = ""
        elif open:
            element += caractere

    return elements