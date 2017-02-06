# Complex Embeddings for Simple Link Prediction

This repository contains the code for experiments in the paper:

[Complex Embeddings for Simple Link Prediction](http://jmlr.org/proceedings/papers/v48/trouillon16.pdf),
Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard, ICML 2016.

## Install 

First clone the repository:
```
git clone https://github.com/ttrouill/complex.git
```

The code dependends on [downhill](https://github.com/lmjohns3/downhill),
a theano based Stochastic Gradient Descent implementation.

Install it, along with other dependencies with:
```
pip install -r requirements.txt
```

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

By default, it runs the ComplEx (Complex Embeddings) model, edit the files and uncomment the corresponding lines to run DistMult, TransE or CP models. The given hyper-parameters for each model are the best validated ones by the grid-search described in the paper.

To run on GPU (approx 5x faster), simply add the following theano flag before the python call:
```
THEANO_FLAGS='device=gpu' python fb15k_run.py
```

## Citing ComplEx

If you use this package for published work, please cite the paper. Here is the BibTeX:
```
@inproceedings{trouillon2016complex,
	author = {Trouillon, Th\'eo and Welbl, Johannes and Riedel, Sebastian and Gaussier, \'Eric and Bouchard, Guillaume},
	booktitle = {International Conference on Machine Learning (ICML)},
	title = {{Complex embeddings for simple link prediction}},
	volume={48},
	pages={2071--2080},
	year = {2016}
}
```

## License

This software comes under a non-commercial use license, please see the LICENSE file.
