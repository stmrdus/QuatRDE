import config
from  models import *
import json
import os 
import numpy as np

def get_term_id(filename):
    entity2id = {}
    id2entity = {}
    with open(filename) as f:
        for line in f:
            if len(line.strip().split()) > 1:
                tmp = line.strip().split()
                entity2id[tmp[0]] = int(tmp[1])
                id2entity[int(tmp[1])] = tmp[0]
    return entity2id, id2entity

def get_init_embeddings(relinit, entinit):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstent.append(tmp)
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)

hidden_size = 50
dataset = "WN18RR"

init_entity_embs, init_relation_embs = get_init_embeddings(
            "./benchmarks/" + dataset + "/relation2vec"+str(hidden_size)+".init",
            "./benchmarks/" + dataset + "/entity2vec"+str(hidden_size)+".init")

e2id, id2e = get_term_id(filename="./benchmarks/" + dataset + "/entity2id.txt")
e2id50, id2e50 = get_term_id(filename="./benchmarks/" + dataset + "/entity2id_"+str(hidden_size)+"init.txt")
assert len(e2id) == len(e2id50)

entity_embs = np.empty([len(e2id), hidden_size]).astype(np.float32)
for i in range(len(e2id)):
    _word = id2e[i]
    id = e2id50[_word]
    entity_embs[i] = init_entity_embs[id]
    
r2id, id2r = get_term_id(filename="./benchmarks/" + dataset + "/relation2id.txt")
r2id50, id2r50 = get_term_id(filename="./benchmarks/" + dataset + "/relation2id_"+str(hidden_size)+"init.txt")
assert len(r2id) == len(r2id50)

rel_embs = np.empty([len(r2id), hidden_size]).astype(np.float32)
for i in range(len(r2id)):
    _rel = id2r[i]
    id = r2id50[_rel]
    rel_embs[i] = init_relation_embs[id]

con = config.Config()
con.set_init_embeddings(entity_embs, rel_embs)
con.set_in_path("./benchmarks/WN18RR/")
con.set_work_threads(8)
con.set_train_times(3000)
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_bern(1)
# Test ở các chiều: 100, 200, 300, 400
con.set_dimension(50)
# L2 regularization: 0.05, 0.1
con.set_lmbda(0.1)
con.set_lmbda_two(0.01)
con.set_margin(1.0)
# Number of negative triples for each triple: 1, 5, 10
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1)
con.set_valid_steps(1)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint_WN18RR")
con.set_result_dir("./result_WN18RR")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(QuatRDE)
con.train()

'''
Best config
- dimension:
- lambda:
- lambda_2:
- ent_neg_rate:
- optimizer:
'''