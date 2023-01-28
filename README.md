# Hierarchical-Attention-Network-with-BERT-Embeddings-for-Sentiment-Analysis-
Repository contains code for building a Hierarchical Attention Network with BERT Embeddings for Sentiment Analysis on IMDB dataset as described in the paper:

Yang, Zichao, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and EduardbHovy. 2016. Hierarchical attention networks for document classification. Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1480-1489.(Link: https://aclanthology.org/N16-1174.pdf). 

Refer to the report on how to run the codes. The repository contains code to perform sentiment analysis using the following:

1.  Hierarchical Attention Network with BERT Embeddings - Accuracy obtained: 86.25% for batch size = 16 and number of iterations = 10 
2.  Hierarchical Attention Network with GLOVE Embeddings - Accuracy obtained: 83.8% for batch size = 16 and number of iterations = 10 
3.  Hierarchical Attention Network with One-hot-Encodings - Accuracy obtained: 58.25% for batch size = 16 and number of iterations = 10 
4.  Naive Bayes - Accuracy obtained: 80.85%
5.  Perceptron  - Accuracy obtained: 77.10% for 1000 iterations

# Experiment: How removal of various attention layers affect the accuracy:
For 10 iterations and batch size 16 and with Glove Embeddings:

Original model (Hierarchical Attention Network) accuracy: 0.838000 

Model accuracy with attention at sentence level removed: 0.825500 

Model accuracy with attention at word level removed: 0.812500 

Model accuracy with attention at both word and sentence level removed: 0.792000

So, there is a decrease in accuracy from the original (full) model when attention at word level or sentence level is removed. The accuracy is the lowest when attention from both the levels is removed. This is expected because with attention mechanisms at word level the model could understand which words are more important and need more attention whereas with attention mechanism at sentence level the model could understand which sentences in the passage are more important and need more attention. Without either or both the features, the model loses the corresponding capabilities hence resulting in a decrease in the accuracy of the original model.
