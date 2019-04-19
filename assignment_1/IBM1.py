from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
import copy
import dill
import os

from DataLoader import DataLoader

PARAMETERS_PATH = "./models/IBM1/"

class IBM1:
    def __init__(self, english_data_path, french_data_path):
        self.training_data = DataLoader(english_data_path, french_data_path)
        self.prob = self.uniform_initialisation()
    
    def uniform_initialisation(self):
        unif_value = 1/self.training_data.n_english_vocab
        return defaultdict(lambda: defaultdict(lambda: unif_value))

    def train(self, epsilon = 10e-5):
        for iteration in tqdm(range(1)):
            tcount = defaultdict(lambda: defaultdict(float))
            total = defaultdict(float)
            for e_sentence,f_sentence in tqdm(self.training_data.generate_sentence_pairs()):    
                for f_word in set(f_sentence):
                    denom_c = 0
                    for e_word in set(e_sentence):
                        denom_c += self.prob[e_word][f_word] * f_sentence.count(f_word)
                    for e_word in set(e_sentence):
                        weight = (self.prob[e_word][f_word] * f_sentence.count(f_word) * e_sentence.count(e_word)) / denom_c
                        tcount[e_word][f_word] += weight                                                
                        total[e_word] += weight
            for e_word in tqdm(tcount.keys()):
                for f_word in tcount[e_word].keys():
                    self.prob[e_word][f_word] = tcount[e_word][f_word] / total[e_word]
            self.save_checkpoint(iteration)

    def save_checkpoint(self, iteration):
        model_path = self.training_data.english_data_path.split("/")[2]
        with open(PARAMETERS_PATH + 'probs_{}_{}'.format(model_path, iteration) + '.pkl', 'wb') as f:
            dill.dump(self.prob, f)

    def load_checkpoint(self):
        all_checkpoints = os.listdir(PARAMETERS_PATH)
        latest_checkpoint = sorted(all_checkpoints)[-1]
        with open(PARAMETERS_PATH + latest_checkpoint, 'rb') as f:
            self.prob = dill.load(f)
    

if __name__ == "__main__":
    english_data_path = "./training/hansards.36.2.e"
    french_data_path = "./training/hansards.36.2.f"
    ibm1 = IBM1(english_data_path, french_data_path)
    ibm1.train()
    ibm1.load_checkpoint()