import os

DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.join(os.path.dirname(DATA_DIR), 'mnist')

MAX_EPOCH       = 20
EXTRACT_EPOCHS  = list(range(MAX_EPOCH + 1))
N_PER_CLASS     = 100
TSNE_PERPLEXITY = 30
TSNE_ITER       = 1000
PCA_DIM         = 50
