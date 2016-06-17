import uuid
import time
import subprocess
import xml.dom.minidom
import cPickle as pickle
import numpy as np


from tools import *
from evaluation import *
import models


class Experiment(object):

	def __init__(self, name, train, valid, test, positives_only = False,  compute_ranking_scores = False, entities_dict = None, relations_dict =None) :
		"""
		An experiment is defined by its train and test set, which are two Triplets_set objects.
		"""

		self.name = name
		self.train = train
		self.valid = valid
		self.test = test
		self.train_tensor = None
		self.train_mask = None
		self.positives_only = positives_only
		self.entities_dict = entities_dict
		self.relations_dict = relations_dict

		if valid != None:
			self.n_entities = len(np.unique(np.concatenate((train.indexes[:,0], train.indexes[:,2], valid.indexes[:,0], valid.indexes[:,2], test.indexes[:,0], test.indexes[:,2]))))
			self.n_relations = len(np.unique(np.concatenate((train.indexes[:,1], valid.indexes[:,1], test.indexes[:,1]))))
		else:
			self.n_entities = len(np.unique(np.concatenate((train.indexes[:,0], train.indexes[:,2], test.indexes[:,0], test.indexes[:,2]))))
			self.n_relations = len(np.unique(np.concatenate((train.indexes[:,1], test.indexes[:,1]))))


		logger.info("Nb entities: " + str(self.n_entities))
		logger.info( "Nb relations: " + str(self.n_relations))
		logger.info( "Nb obs triples: " + str(train.indexes.shape[0]))
		
		self.scorer = Scorer(train, valid, test, compute_ranking_scores)
		
		#The trained models are stored indexed by name
		self.models = {}
		#The test Results are stored indexed by model name
		self.valid_results = CV_Results()
		self.results = CV_Results()






	def grid_search_on_all_models(self, params, embedding_size_grid = [1,2,3,4,5,6,7,8,9,10], lmbda_grid = [0.1], nb_runs = 10):
		"""
		Here params is a dictionnary of Parameters, indexed by the names of each model, that
		must match with the model class names
		"""

		#Clear previous results:
		self.results = CV_Results()
		self.valid_results = CV_Results()

		for model_s in params:
			logger.info("Starting grid search on: " + model_s)

			#Getting train and test function using model string id:
			cur_params = params[model_s]
			for embedding_size in embedding_size_grid:
				for lmbda in lmbda_grid:
					cur_params.embedding_size = embedding_size
					cur_params.lmbda = lmbda

					for run in range(nb_runs):
						self.run_model(model_s,cur_params)
						self.test_model(model_s)
								
		logger.info("Grid search finished")



	def run_model(self,model_s,params):
		"""
		Generic training for any model, model_s is the class name of the model class defined in module models
		"""
		
		#Reuse ancient model if already exist:
		if model_s in self.models:
			model = self.models[model_s][0]
		else: #Else construct it:
			model = vars(models)[model_s]()

		self.models[model_s] = (model, params)

		model.fit(self.train, self.valid, params, self.n_entities, self.n_relations, self.n_entities, self.scorer)

	def test_model(self, model_s):
		"""
		Generic testing for any model, model_s is the class name of the model class defined in module models
		"""
		model, params = self.models[model_s]

		if self.valid != None:
			res = self.scorer.compute_scores(model, model_s, params, self.valid)
			self.valid_results.add_res(res, model_s, params.embedding_size, params.lmbda, model.nb_params)

		res = self.scorer.compute_scores(model, model_s, params, self.test)
		self.results.add_res(res, model_s, params.embedding_size, params.lmbda, model.nb_params)


	def print_best_MRR_and_hits(self):
		"""
		Print best results on validation set, and corresponding scores (with same hyper params) on test set
		"""
		logger.info( "Validation metrics:")
		metrics = self.valid_results.print_MRR_and_hits()
		logger.info( "Corresponding Test metrics:")
		for model_s, (best_rank, best_lambda, _,_,_,_,_) in metrics.items():
			self.results.print_MRR_and_hits_given_params(model_s, best_rank, best_lambda)


	def print_best_MRR_and_hits_per_rel(self):
		"""
		Print best results on validation set, and corresponding scores (with same hyper params) on test set
		"""
		
		logger.info( "Overall relations validation metrics:")
		metrics = self.valid_results.print_MRR_and_hits()

		logger.info( "Corresponding per relation Test metrics:" )
		for rel_name, rel_idx in self.relations_dict.items():

			logger.info( rel_name )
			this_rel_row_idxs = self.test.indexes[:,1] == rel_idx
			this_rel_test_indexes = self.test.indexes[ this_rel_row_idxs ,:]
			this_rel_test_values = self.test.values[ this_rel_row_idxs ]

			this_rel_set = Triplets_set(this_rel_test_indexes,this_rel_test_values)

			for model_s, (best_rank, best_lambda, _,_,_,_,_) in metrics.items():
				rel_cv_results = self.results.extract_sub_scores( this_rel_row_idxs)
				rel_cv_results.print_MRR_and_hits_given_params(model_s, best_rank, best_lambda)



