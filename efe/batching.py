from .tools import *


class Batch_Loader(object):
	def __init__(self, train_triples, n_entities, batch_size=100, neg_ratio = 0.0, contiguous_sampling = False):
		self.train_triples = train_triples
		self.batch_size = batch_size
		self.n_entities = n_entities
		self.contiguous_sampling = contiguous_sampling
		self.neg_ratio = int(neg_ratio)
		self.idx = 0

		self.new_triples_indexes = np.empty((self.batch_size * (self.neg_ratio + 1) , 3), dtype=np.int64)
		self.new_triples_values = np.empty((self.batch_size * (self.neg_ratio + 1 )), dtype=np.float32)

	def __call__(self):


		if self.contiguous_sampling:
			if self.idx >= len(self.train_triples.values):
				self.idx = 0

			b = self.idx
			e = self.idx + self.batch_size
			this_batch_size = len(self.train_triples.values[b:e]) #Manage shorter batches (last ones)
			self.new_triples_indexes[:this_batch_size,:] = self.train_triples.indexes[b:e]
			self.new_triples_values[:this_batch_size] = self.train_triples.values[b:e]

			self.idx += this_batch_size

			last_idx = this_batch_size
		else:
			idxs = np.random.randint(0,len(self.train_triples.values),self.batch_size)
			self.new_triples_indexes[:self.batch_size,:] = self.train_triples.indexes[idxs,:]
			self.new_triples_values[:self.batch_size] = self.train_triples.values[idxs]

			last_idx = self.batch_size


		if self.neg_ratio > 0:

			#Pre-sample everything, faster
			rdm_entities = np.random.randint(0, self.n_entities, last_idx * self.neg_ratio)
			rdm_choices = np.random.random(last_idx * self.neg_ratio) < 0.5
			#Pre copying everyting
			self.new_triples_indexes[last_idx:(last_idx*(self.neg_ratio+1)),:] = np.tile(self.new_triples_indexes[:last_idx,:],(self.neg_ratio,1))
			self.new_triples_values[last_idx:(last_idx*(self.neg_ratio+1))] = np.tile(self.new_triples_values[:last_idx], self.neg_ratio)

			for i in range(last_idx):
				for j in range(self.neg_ratio):
					cur_idx = i* self.neg_ratio + j
					#Sample a random subject or object 
					if rdm_choices[cur_idx]:
						self.new_triples_indexes[last_idx + cur_idx,0] = rdm_entities[cur_idx]
					else:
						self.new_triples_indexes[last_idx + cur_idx,2] = rdm_entities[cur_idx]

					self.new_triples_values[last_idx + cur_idx] = -1

			last_idx += cur_idx + 1

		train = [self.new_triples_values[:last_idx], self.new_triples_indexes[:last_idx,0], self.new_triples_indexes[:last_idx,1], self.new_triples_indexes[:last_idx,2]]


		return train



class TransE_Batch_Loader(Batch_Loader):
	#Hacky trick to normalize embeddings at each update
	def __init__(self, model, train_triples, n_entities, batch_size=100, neg_ratio = 0.0, contiguous_sampling = False):
		super(TransE_Batch_Loader, self).__init__(train_triples, n_entities, batch_size, neg_ratio, contiguous_sampling)

		self.model = model

	def __call__(self):
		train = super(TransE_Batch_Loader, self).__call__()
		train = train[1:]

		#Projection on L2 sphere before each batch
		self.model.e.set_value(L2_proj(self.model.e.get_value(borrow = True)), borrow = True)

		return train
