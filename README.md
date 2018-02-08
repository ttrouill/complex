# Complex Embeddings for Simple Link Prediction

This repository contains the code of the main experiments presented in the papers:

[Complex Embeddings for Simple Link Prediction](http://jmlr.org/proceedings/papers/v48/trouillon16.pdf),
Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard, ICML 2016.

[Knowledge Graph Completion via Complex Tensor Factorization](http://www.jmlr.org/papers/volume18/16-563/16-563.pdf),
Théo Trouillon, Christopher R. Dance, Éric Gaussier, Johannes Welbl, Sebastian Riedel and Guillaume Bouchard, JMLR 2017.

## Install 

First clone the repository:
```
git clone https://github.com/ttrouill/complex.git
```

The code depends on [downhill](https://github.com/lmjohns3/downhill), a theano-based Stochastic Gradient Descent implementation.

Install it, along with other dependencies with:
```
pip install -r requirements.txt
```

The code is compatible with Python 2 and 3.

## Run the experiments

To run the experiments, unpack the datasets first:
```
unzip datasets/fb15k.zip -d datasets/
unzip datasets/wn18.zip -d datasets/
```

And run the corresponding python scripts, for Freebase (FB15K):
```
python fb15k_run.py
```

And for Wordnet (WN18):
```
python wn18_run.py
```

By default, it runs the ComplEx (Complex Embeddings) model, edit the files and uncomment the corresponding lines to run DistMult, TransE, RESCAL or CP models. The given hyper-parameters for each model are the best validated ones by the grid-search described in the paper.

To run on GPU (approx 5x faster), simply add the following theano flag before the python call:
```
THEANO_FLAGS='device=gpu' python fb15k_run.py
```

## Export the produced embeddings

Simply uncomment the last lines in `fb15k_run.py` and `wn18_run.py` (and the `import scipy.io` line (requires scipy module)), this will save the embeddings of the ComplEx model in the common matlab `.mat` format.
If you want to save the embeddings of other models, just edit the embedding variable names corresponding to the desired model (see `models.py`).


## Run on your own data

Create a subfolder in the `datasets` folder, and put your data in three files `train.txt`, `valid.txt` and `test.txt`. Each line is a triple, in the format: 
```
subject_entity_id	relation_id	object_entity_id
```
separated with tabs. Then modify `fb15k_run.py` for example, by changing the `name` argument in the `build_data` function call to your data set folder name:
```
fb15kexp = build_data(name = 'your_dataset_folder_name',path = tools.cur_path + '/datasets/')
```


## Implement your own model


Models are defined as classes in `models.py`, that all inherit the class `Abstract_Model` defined in the same file. The `Abstract_Model` class handles all the common stuff (training functions, ...), and child classes (the actual models) just need to define their embeddings shape and initialization, and their scoring and loss function.

To properly understand the following, one must be comfortable with [Theano basics](http://deeplearning.net/software/theano/library/tensor/basic.html).

The `Abstract_Model` class contains the symbolic 1D tensor variables `self.rows`, `self.cols`, `self.tubes` and `self.ys` that will instantiate at runtime the corresponding: subject entity indexes, relation indexes, object entity indexes and truth values (1 or -1) respectively, of the triples of the current batch. It also contains the number of subject entities, relations and object entities of the dataset in `self.n`, `self.m`, `self.l` respectively, as well as the current embedding size in `self.k`.

Two functions must be overridden in the child classes to define a proper model: `get_init_params(self)` and `define_loss(self)`.

Let's have a look at the `DistMult_Model` class and its `get_init_params(self)` function:
```
def get_init_params(self):
	params = { 'e' : randn(max(self.n,self.l),self.k),
			   'r' : randn(self.m,self.k)}
	return params
```
This function both defines the embedding-matrix shapes (number of entities * rank for `e`, number of relations * rank for `r`), and their initial value (`randn` is `numpy.random.randn`), by returning a dictionnary where the key names correspond to the class attribute names. From this dict the mother class will create shared tensor variables initialized with the given values, and assigned to the corresponding attribute names (`self.e` and `self.r`).

Now the `define_loss(self)` function must define three Theano expressions: the scoring function, the loss, and the regularization.
Here is the `DistMult_Model` one:
```
def define_loss(self):
	self.pred_func = TT.sum(self.e[self.rows,:] * self.r[self.cols,:] * self.e[self.tubes,:], 1)

	self.loss = TT.sqr(self.ys - self.pred_func).mean()

	self.regul_func = TT.sqr(self.e[self.rows,:]).mean() \
					+ TT.sqr(self.r[self.cols,:]).mean() \
					+ TT.sqr(self.e[self.tubes,:]).mean()
```
The corresponding expressions must be written in their batched form, i.e. to compute the scores of multiple triples at once. For a given batch, the corresponding embeddings are retrieved with `self.e[self.rows,:]`, `self.r[self.cols,:]` and `self.e[self.tubes,:]`.

In the case of the DistMult model, the trilinear product between these embeddings is computed, here by doing first two element-wise multiplications and then a sum over the columns in the `self.pred_func` expression. The `self.pred_func` expression must yield a vector of the size of the batch (the size of `self.rows`, `self.cols`, ...).
The loss defined in `self.loss` is the squared-loss here (see the `DistMult_Logistic_Model` class for the logistic loss), and is averaged over the batch, as the `self.loss` expression must yield a scalar value.
The regularization defined here is the L2 regularization over the corresponding embeddings of the batch, and must also yield a scalar value.

That's all you need to implement your own tensor factorization model! All gradient computation is handled by Theano auto-differentiation, and all the training functions by the [downhill](https://github.com/lmjohns3/downhill) module and the `Abstract_Model` class.




## Cite ComplEx

If you use this package for published work, please cite either or both papers, here is the BibTeX:
```
@inproceedings{trouillon2016complex,
	title = {{Complex embeddings for simple link prediction}},
	author = {Trouillon, Th\'eo and Welbl, Johannes and Riedel, Sebastian and Gaussier, \'Eric and Bouchard, Guillaume},
	booktitle = {International Conference on Machine Learning (ICML)},
	volume={48},
	pages={2071--2080},
	year = {2016}
}
@article{trouillon2017knowledge,
	title={Knowledge graph completion via complex tensor factorization},
	author={Trouillon, Th{\'e}o and Dance, Christopher R and Gaussier, {\'E}ric and Welbl, Johannes and Riedel, Sebastian and Bouchard, Guillaume},
	journal={Journal of Machine Learning Research (JMLR)},
	volume={18},
	number={130},
	pages={1--38},
	year={2017}
}

```

## License

This software comes under a non-commercial use license, please see the LICENSE file.
