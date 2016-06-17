import scipy
import scipy.io
import random
import cPickle

from experiment import *


def parse_line(filename, line,i):
	line = line.strip().split("\t")
	sub = line[0]
	rel = line[1]
	obj = line[2]
	val = 1

	return sub,obj,rel,val


def load_triples_from_txt(filenames, entities_indexes = None, relations_indexes = None, add_sameas_rel = False, parse_line = parse_line):
	"""
	Take a list of file names and build the corresponding dictionnary of triples
	"""


	if entities_indexes == None:
		entities_indexes= dict()
		entities = set()
		next_ent = 0
	else:
		entities = set(entities_indexes)
		next_ent = max(entities_indexes.values()) + 1


	if relations_indexes == None:
		relations_indexes= dict()
		relations= set()
		next_rel = 0
	else:
		relations = set(relations_indexes)
		next_rel = max(relations_indexes.values()) + 1

	data = dict()



	for filename in filenames:
		with open(filename) as f:
			lines = f.readlines()

		for i,line in enumerate(lines):

			sub,obj,rel,val = parse_line(filename, line,i)


			if sub in entities:
				sub_ind = entities_indexes[sub]
			else:
				sub_ind = next_ent
				next_ent += 1
				entities_indexes[sub] = sub_ind
				entities.add(sub)
				
			if obj in entities:
				obj_ind = entities_indexes[obj]
			else:
				obj_ind = next_ent
				next_ent += 1
				entities_indexes[obj] = obj_ind
				entities.add(obj)
				
			if rel in relations:
				rel_ind = relations_indexes[rel]
			else:
				rel_ind = next_rel
				next_rel += 1
				relations_indexes[rel] = rel_ind
				relations.add(rel)

			data[ (sub_ind, rel_ind, obj_ind)] = val

	if add_sameas_rel :
		rel = "sameAs_"
		rel_ind = next_rel
		next_rel += 1
		relations_indexes[rel] = rel_ind
		relations.add(rel)
		for sub in entities_indexes:
			for obj in entities_indexes:
				if sub == obj:
					data[ (entities_indexes[sub], rel_ind, entities_indexes[obj])] = 1
				else:
					data[ (entities_indexes[sub], rel_ind, entities_indexes[obj])] = -1

	return data, entities_indexes, relations_indexes


def build_data(name, path = '/home/ttrouill/dbfactor/projects/relational_bench/datasets/'):




	folder = path + '/' + name + '/'


	train_triples, entities_indexes, relations_indexes = load_triples_from_txt([folder + 'train.txt'], 
					add_sameas_rel = False, parse_line = parse_line)


	valid_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'valid.txt'],
					entities_indexes = entities_indexes , relations_indexes = relations_indexes,
					add_sameas_rel = False, parse_line = parse_line)

	test_triples, entities_indexes, relations_indexes =  load_triples_from_txt([folder + 'test.txt'],
					entities_indexes = entities_indexes, relations_indexes = relations_indexes,
					add_sameas_rel = False, parse_line = parse_line)


	train = Triplets_set(np.array(train_triples.keys()), np.array(train_triples.values()))
	valid = Triplets_set(np.array(valid_triples.keys()), np.array(valid_triples.values()))
	test = Triplets_set(np.array(test_triples.keys()), np.array(test_triples.values()))


	return Experiment(name,train, valid, test, positives_only = True, compute_ranking_scores = True, entities_dict = entities_indexes, relations_dict = relations_indexes)




def load_mat_file(name, path, matname, load_zeros = False, prop_valid_set = .1, prop_test_set=0):

	x = scipy.io.loadmat(path + name)[matname]


	if sp.issparse(x): 
		if not load_zeros:
			idxs = x.nonzero()

			indexes = np.array(zip(idxs[0], np.zeros_like(idxs[0]), idxs[1]))
			np.random.shuffle(indexes)

			nb = indexes.shape[0]
			i_valid = int(nb - nb*prop_valid_set - nb * prop_test_set)
			i_test = i_valid + int( nb*prop_valid_set)

			train = Triplets_set(indexes[:i_valid,:], np.ones(i_valid))
			valid = Triplets_set(indexes[i_valid:i_test,:], np.ones(i_test - i_valid))
			test = Triplets_set(indexes[i_test:,:], np.ones(nb - i_test))


	return Experiment(name,train, valid, test, positives_only = True, compute_ranking_scores = True)
	




