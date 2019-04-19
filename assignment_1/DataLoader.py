from collections import defaultdict

class DataLoader:
    def __init__(self, english_data_path, french_data_path):
        self.english_data = open(english_data_path, "r").readlines()
        self.french_data = open(french_data_path, "r").readlines()
        self.n_english_vocab = self.vocabulary_length(self.english_data)

    def preprocess(self, sentence):
        output = sentence[:-2] # remove /n character
        output = output.lower()
        return output.split()
    
    def generate_sentence_pairs(self):
        for e,f in zip(self.english_data, self.french_data):
            e = self.preprocess(e)
            f = self.preprocess(f)
            f.insert(0, "NULL")
            yield (e,f)

    def vocabulary_length(self, languange_data):
        vocab = defaultdict(int)
        for sentence in languange_data:
            for word in self.preprocess(sentence):
                vocab[word] += 1
        return len(vocab)



if __name__ == "__main__":
    english_data_path = "./training/hansards.36.2.e"
    french_data_path = "./training/hansards.36.2.f"

    data = DataLoader(english_data_path, french_data_path)
    # max_idx = 10
    # for idx, pair in enumerate(data.generate_sentence_pairs()):
    #     if idx > max_idx:
    #         break
    #     print(pair)
    
