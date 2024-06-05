class SentenceCleaner:
    def __init__(self):
        pass

    def clean_sentence(self, s):
        return s.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

    def parse_sentences(self, s):
        cleaned = [self.clean_sentence(sent) for sent in s]
        return "\n-----------------------------------------\n".join(cleaned)