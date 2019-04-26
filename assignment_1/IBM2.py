
# coding: utf-8

# In[ ]:


from collections import defaultdict
import dill
import os
from math import log
from matplotlib import pyplot as plt
import numpy as np
import math
import sys

from DataLoader import DataLoader
from aer import test


# In[ ]:


PARAMETERS_PATH = "./models/IBM2/"
TEMP_PATH = "./models/IBM1/"
PREDICTIONS_PATH = "./predictions/IBM2/"
TEST_PATH = "./testing/eval/"
PLOTS_PATH = "./plots/IBM2/"
EXPERIMENT_NAME = "Lowercase_E_F"


# In[ ]:


class IBM2:
    def __init__(self, source_train_path, target_train_path, source_eval_path, target_eval_path, source_test_path, target_test_path):
        
        self.training_data = DataLoader(source_train_path, target_train_path)
        self.evaluation_data = DataLoader(source_eval_path, target_eval_path)
        self.test_data = DataLoader(source_test_path, target_test_path)
        self.l = self.training_data.n_source_vocab
        self.m = self.training_data.n_target_vocab
        self.jump = np.zeros((0,0))
        self.jump_range = 100
        self.jump = 1. / (2 * self.jump_range) * np.ones((1, 2 * self.jump_range))
#         change it to different kinds of initialisation later        
#         self.load_checkpoint() #uncomment this IBM1 intialization
        self.prob = self.random_initialisation() 
#         self.prob = self.uniform_initialisation() #uncomment this uniform intialization
    
    def random_initialisation(self):
        return defaultdict(lambda: defaultdict(lambda: np.random.choice(np.random.dirichlet((1,2,3)))))
    
    def uniform_initialisation(self):
        unif_value = 1/(self.training_data.n_source_vocab * 2) 
        return defaultdict(lambda: defaultdict(lambda: unif_value))
    
    def ibm_initialisation(self):
        self.load_checkpoint()

    def train(self, num_iterations = 15):
        aers = []
        log_likelihoods = []
        for iteration in range(num_iterations):
            print("Starting iteration", iteration)
            self.em_iteration()
#             self.save_checkpoint(iteration)
#             print("Checkpoint created")
            self.write_prediction(data = self.evaluation_data, iteration = iteration)
            aer, log_likelihood = self.evaluate(iteration)
            aers.append(aer)
            log_likelihoods.append(log_likelihood)
        np.save(PARAMETERS_PATH + "_aers_ibm", aers)
        np.save(PARAMETERS_PATH + "_logl_ibm", log_likelihoods)
        self.plot_results(aers, log_likelihoods)

    def jump_func(self,i,j,l,m):
        
        #jump is being used as an index
        # [-jump_range, +jump range] -> [0, 2 * jump_range]
        
        jump = int(i - math.floor(j *l / m)) + self.jump_range
        if(jump >= 2* self.jump_range):
            index = self.jump_range - 1
        elif(jump < 0):
            index = 0 
        else:
            index = jump
        
        return index
        
        
    
    def em_iteration(self):
        tcount = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        jcount = np.zeros((1, 2 * self.jump_range))
        #for all sentence pairs
        for s_sentence, t_sentence in self.training_data.generate_sentence_pairs():  
            for i, t_word in enumerate(t_sentence):
                denom_c = 0
                for j, s_word in enumerate(s_sentence):
                    
                    index = self.jump_func(j, i, self.l, self.m) 
                    denom_c += self.prob[s_word][t_word]  * self.jump[0, index] #* t_sentence.count(t_word)
                for j, s_word in enumerate(s_sentence):
                    
                    index = self.jump_func(j, i, self.l, self.m)
                    weight = (self.prob[s_word][t_word]  * self.jump[0,index]) / denom_c  #* t_sentence.count(t_word) * s_sentence.count(s_word)
                    
                    tcount[s_word][t_word] += weight                                                
                    total[s_word] += weight
                    jcount[0, index] += weight
           
        for s_word in tcount.keys():
            for t_word in tcount[s_word].keys():
                self.prob[s_word][t_word] = tcount[s_word][t_word] / total[s_word]
        self.jump = 1. / float(np.sum(jcount)) *  jcount


    def save_checkpoint(self, iteration):
        model_path = self.training_data.source_data_path.split("/")[2]
        with open(PARAMETERS_PATH + 'probs_{}_{}_{}'.format(model_path, EXPERIMENT_NAME, iteration) + '.pkl', 'wb') as f:
            dill.dump(self.prob, f)

    def load_checkpoint(self):
        all_checkpoints = os.listdir(TEMP_PATH)
        latest_checkpoint = sorted(all_checkpoints)[-1]
        with open(TEMP_PATH + latest_checkpoint, 'rb') as f:
            print("loading", TEMP_PATH + latest_checkpoint)
            self.prob = dill.load(f)
    
    def write_prediction(self, data, path = PREDICTIONS_PATH, output_type = "eval", iteration = 9): 
        f = open(path + "prediction_{}_{}.txt".format(EXPERIMENT_NAME, iteration), "w+")
        for sentence_idx, (s_sentence,t_sentence) in enumerate(data.generate_sentence_pairs()):
            alignments = self.find_alignment(s_sentence, t_sentence)
            for s_align, t_align in alignments:
                if s_align != 0:
                    if output_type == "eval":
                        f.write("{} {} {} {} \n".format(sentence_idx+1, s_align, t_align+1, "S")) 
                    if output_type == "test":
                        f.write("{} {} {} {} \n".format(self.add_back_zeros(sentence_idx+1), s_align, t_align+1, "S"))
        f.close()

    def evaluate(self, iteration):
        aer = test("./validation/dev.wa.nonullalign", PREDICTIONS_PATH + "prediction_{}_{}.txt".format(EXPERIMENT_NAME, iteration))
        print("AER:", aer)
        log_likelihood = 0
        for s_sentence,t_sentence in self.training_data.generate_sentence_pairs():
            log_likelihood += self.calculate_log_likelihood(s_sentence, t_sentence)
        print("Log-Likelihood:", log_likelihood)
        return aer, log_likelihood

    def test(self):
        self.write_prediction(data = self.test_data, path = TEST_PATH, output_type = "test")

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

    def find_alignment(self, source_sentence, target_sentence):
        sentence_alignment = []
        j = 0
        for t_idx, t_word in enumerate(target_sentence):
            max_s_idx = 0
            max_s_prob = 0
            deno_c = 0.
            for s_idx, s_word in enumerate(source_sentence):
                index = self.jump_func(s_idx, t_idx, self.l, self.m)
                probability = self.prob[s_word][t_word] * self.jump[0, index]                
                if probability > max_s_prob:
                    max_s_prob = probability
                    max_s_idx = s_idx
            sentence_alignment.append((max_s_idx, t_idx))
        return sentence_alignment
        
    def calculate_log_likelihood(self, source_sentence, target_sentence):
        
        alignments = self.find_alignment(source_sentence, target_sentence)
        log_likelihood = 0
        for s_idx, t_idx in alignments:
            index = self.jump_func(alignments[t_idx][0], t_idx ,self.l, self.m)
            log_likelihood += log(self.prob[source_sentence[s_idx]][target_sentence[t_idx]]) + log(self.jump[0,index])
        return log_likelihood

    @staticmethod
    def add_back_zeros(number, number_of_back_zeros=4):
        number = str(number)
        while len(number) < number_of_back_zeros:
            number = "0" + number
        return number

    def plot_results(self, aers, log_likelihoods):
        plt.plot(aers, label = "AER")
        plt.legend()
        plt.savefig(PLOTS_PATH + EXPERIMENT_NAME + "AER.eps")
        plt.close()
        plt.plot(log_likelihoods, label = "log_likelihood")
        plt.legend()
        plt.savefig(PLOTS_PATH + EXPERIMENT_NAME + "log_likelihood.eps")
        


# In[ ]:


if __name__ == "__main__":
    english_train_path = "./training/hansards.36.2.e"
    french_train_path = "./training/hansards.36.2.f"
    english_eval_path  = "./validation/dev.e"
    french_eval_path = "./validation/dev.f"
    english_test_path = "./testing/test/test.e"
    french_test_path = "./testing/test/test.f"
    ibm2 = IBM2(source_train_path = english_train_path,
                target_train_path= french_train_path,
                source_eval_path= english_eval_path,
                target_eval_path = french_eval_path,
                source_test_path = english_test_path,
                target_test_path = french_test_path)
    ibm2.train()
    ibm2.test()

