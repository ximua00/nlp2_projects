from collections import defaultdict
import dill
import os

from pprint import pprint
from tqdm import tqdm
from math import log
from matplotlib import pyplot as plt

from DataLoader import DataLoader
from aer import test

PARAMETERS_PATH = "./models/IBM1/"
PREDICTIONS_PATH = "./predictions/IBM1/"
PLOTS_PATH = "./plots/IBM1/"

class IBM1:
    def __init__(self, source_train_path, target_train_path, source_eval_path, target_eval_path):
        self.training_data = DataLoader(source_train_path, target_train_path)
        self.evaluation_data = DataLoader(source_eval_path, target_eval_path)
        self.prob = self.uniform_initialisation()
    
    def uniform_initialisation(self):
        unif_value = 1/self.training_data.n_source_vocab
        return defaultdict(lambda: defaultdict(lambda: unif_value))

    def train(self, num_iterations = 10):
        aers = []
        log_likelihoods = []
        for iteration in tqdm(range(num_iterations)):
            self.em_iteration()
            self.save_checkpoint(iteration)
            self.write_prediction(iteration)
            aer, log_likelihood = self.evaluate(iteration)
            aers.append(aer)
            log_likelihoods.append(log_likelihood)
        self.plot_results(aers, log_likelihoods)

    def em_iteration(self):
        tcount = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        for s_sentence, t_sentence in self.training_data.generate_sentence_pairs():    
            for t_word in set(t_sentence):
                denom_c = 0
                for s_word in set(s_sentence):
                    denom_c += self.prob[s_word][t_word] * t_sentence.count(t_word)
                for s_word in set(s_sentence):
                    weight = (self.prob[s_word][t_word] * t_sentence.count(t_word) * s_sentence.count(s_word)) / denom_c
                    tcount[s_word][t_word] += weight                                                
                    total[s_word] += weight
        for s_word in tcount.keys():
            for t_word in tcount[s_word].keys():
                self.prob[s_word][t_word] = tcount[s_word][t_word] / total[s_word]

    def save_checkpoint(self, iteration):
        model_path = self.training_data.source_data_path.split("/")[2]
        with open(PARAMETERS_PATH + 'probs_{}_{}'.format(model_path, iteration) + '.pkl', 'wb') as f:
            dill.dump(self.prob, f)

    def load_checkpoint(self):
        all_checkpoints = os.listdir(PARAMETERS_PATH)
        latest_checkpoint = sorted(all_checkpoints)[-1]
        with open(PARAMETERS_PATH + latest_checkpoint, 'rb') as f:
            self.prob = dill.load(f)
    
    def write_prediction(self, iteration = 9): 
        f = open(PREDICTIONS_PATH + "eval_prediction_{}.txt".format(iteration), "w+")
        for sentence_idx, (s_sentence,t_sentence) in enumerate(self.evaluation_data.generate_sentence_pairs()):
            alignments = self.viterbi_alignment(s_sentence, t_sentence)            
            for s_align, t_align in alignments:
                f.write("{} {} {} {} \n".format(sentence_idx+1, t_align+1, s_align, "S")) 
                # TODO: wtf is S/P
        f.close()

    def evaluate(self, iteration):
        aer = test("./validation/dev.wa.nonullalign", PREDICTIONS_PATH + "eval_prediction_{}.txt".format(iteration))
        log_likelihood = 0
        for sentence_idx, (s_sentence,t_sentence) in enumerate(self.evaluation_data.generate_sentence_pairs()):
            log_likelihood += self.calculate_log_likelihood(s_sentence, t_sentence)
        print("AER:", aer)
        print("Log-Likelihood:", log_likelihood)
        return aer, log_likelihood

    def viterbi_alignment(self, source_sentence, target_sentence):
        sentence_alignment = []
        for t_idx, t_word in enumerate(target_sentence):
            max_s_idx = 0
            max_s_prob = 0
            for s_idx, s_word in enumerate(source_sentence):
                probability = self.prob[s_word][t_word]
                if probability > max_s_prob:
                    max_s_prob = probability
                    max_s_idx = s_idx
            sentence_alignment.append((max_s_idx, t_idx))
        return sentence_alignment

    def calculate_log_likelihood(self, source_sentence, target_sentence):
        alignments = self.viterbi_alignment(source_sentence, target_sentence)
        log_likelihood = 0
        for s_idx, t_idx in alignments:
            log_likelihood += log(self.prob[source_sentence[s_idx]][target_sentence[t_idx]])
        return log_likelihood

    def plot_results(self, aers, log_likelihoods):
        plt.plot(aers, label = "AER")
        plt.legend()
        plt.savefig(PLOTS_PATH + "AER.eps")
        plt.close()
        plt.plot(log_likelihoods, label = "log_likelihood")
        plt.legend()
        plt.savefig(PLOTS_PATH + "log_likelihood.eps")


if __name__ == "__main__":
    english_train_path = "./training/hansards.36.2.e"
    french_train_path = "./training/hansards.36.2.f"
    english_eval_path  = "./validation/dev.e"
    french_eval_path = "./validation/dev.f"
    ibm1 = IBM1(source_train_path = french_train_path,
                target_train_path= english_train_path,
                source_eval_path= french_eval_path,
                target_eval_path = english_eval_path)
    ibm1.train()
    # ibm1.load_checkpoint()
    # ibm1.write_prediction(9)
    # ibm1.evaluate(9)