from collections import defaultdict


# In[33]:


class DataLoader:
    def __init__(self, source_data_path, target_data_path):
        self.source_data_path = source_data_path
        self.target_data_path = target_data_path
        self.source_data = open(source_data_path, "r").readlines()
        self.target_data = open(target_data_path, "r").readlines()
        self.n_source_vocab =  self.vocabulary(self.source_data)
        self.n_target_vocab = self.vocabulary(self.target_data)

    def preprocess(self, sentence):
        output = sentence
        # output = output.lower()
        return output.split()
    
    def generate_sentence_pairs(self):
        for s,t in zip(self.source_data, self.target_data):
            s = self.preprocess(s)
            s.insert(0, "NULL") #add NULL to source language
            t = self.preprocess(t)
            yield (s,t)

    def vocabulary(self, languange_data):
        vocab = defaultdict(int)
        for sentence in languange_data:
            for word in self.preprocess(sentence):
                vocab[word] += 1
        return len(vocab)


# In[34]:


if __name__ == "__main__":
    english_data_path = "./training/hansards.36.2.e"
    french_data_path = "./training/hansards.36.2.f"


    data = DataLoader(english_data_path, french_data_path)
    print(data.n_source_vocab)
    print(data.n_target_vocab)
