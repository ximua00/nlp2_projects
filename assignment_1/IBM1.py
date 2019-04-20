from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
import dill
import os

from DataLoader import DataLoader
from aer import test

PARAMETERS_PATH = "./models/IBM1/"
PREDICTIONS_PATH = "./predictions/IBM1/"

class IBM1:
    def __init__(self, english_train_path, french_train_path, english_eval_path, french_eval_path):
        self.training_data = DataLoader(english_train_path, french_train_path)
        self.evaluation_data = DataLoader(english_eval_path, french_eval_path)
        self.prob = self.uniform_initialisation()
    
    def uniform_initialisation(self):
        unif_value = 1/self.training_data.n_english_vocab
        return defaultdict(lambda: defaultdict(lambda: unif_value))

    def train(self, num_iterations = 10):
        for iteration in tqdm(range(num_iterations)):
            self.em_iteration()
            self.write_prediction(iteration)
            self.evaluate(iteration)
            self.save_checkpoint(iteration)

    def em_iteration(self):
        tcount = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        for e_sentence,f_sentence in self.training_data.generate_sentence_pairs():    
            for f_word in set(f_sentence):
                denom_c = 0
                for e_word in set(e_sentence):
                    denom_c += self.prob[e_word][f_word] * f_sentence.count(f_word)
                for e_word in set(e_sentence):
                    weight = (self.prob[e_word][f_word] * f_sentence.count(f_word) * e_sentence.count(e_word)) / denom_c
                    tcount[e_word][f_word] += weight                                                
                    total[e_word] += weight
        for e_word in tcount.keys():
            for f_word in tcount[e_word].keys():
                self.prob[e_word][f_word] = tcount[e_word][f_word] / total[e_word]

    def save_checkpoint(self, iteration):
        model_path = self.training_data.english_data_path.split("/")[2]
        with open(PARAMETERS_PATH + 'probs_{}_{}'.format(model_path, iteration) + '.pkl', 'wb') as f:
            dill.dump(self.prob, f)

    def load_checkpoint(self):
        all_checkpoints = os.listdir(PARAMETERS_PATH)
        latest_checkpoint = sorted(all_checkpoints)[-1]
        with open(PARAMETERS_PATH + latest_checkpoint, 'rb') as f:
            self.prob = dill.load(f)
    
    def write_prediction(self, iteration = 9): 
        f = open(PREDICTIONS_PATH + "eval_prediction_{}.txt".format(iteration), "w+")
        for sentence_idx, (e_sentence,f_sentence) in enumerate(self.evaluation_data.generate_sentence_pairs()):
            alignments = self.viterbi_alignment(e_sentence, f_sentence)            
            for e_align, f_align in alignments:
                f.write("{} {} {} {} \n".format(sentence_idx+1, e_align+1, f_align, "S")) 
                # TODO: wtf is S/P
        f.close()

    def evaluate(self, iteration):
        aer = test("./validation/dev.wa.nonullalign", PREDICTIONS_PATH + "eval_prediction_{}.txt".format(iteration))
        print(aer)
        return aer

    def viterbi_alignment(self, english_sentence, french_sentence):
        sentence_alignment = []
        for e_idx, e_word in enumerate(english_sentence):
            max_f_idx = 0
            max_f_prob = 0
            for f_idx, f_word in enumerate(french_sentence):
                probability = self.prob[e_word][f_word]
                if probability > max_f_prob:
                    max_f_prob = probability
                    max_f_idx = f_idx
            sentence_alignment.append((e_idx, max_f_idx))
        return sentence_alignment


if __name__ == "__main__":
    english_train_path = "./training/hansards.36.2.e"
    french_train_path = "./training/hansards.36.2.f"
    english_eval_path  = "./validation/dev.e"
    french_eval_path = "./validation/dev.f"
    ibm1 = IBM1(english_train_path, french_train_path, english_eval_path, french_eval_path)
    ibm1.train()
    # ibm1.load_checkpoint()
    # ibm1.write_prediction(9)
    # ibm1.evaluate(9)