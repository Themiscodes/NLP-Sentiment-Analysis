# NLP Sentiment Analysis

This project focuses on the development of deep learning models for sentiment analysis tasks, exploring the techniques of logistic regression, Feedforward Neural Networks (FNN), Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) with attention mechanisms, and Bidirectional Encoder Representations from Transformers (BERT). 

The primary objective is to build sentiment classifiers capable of evaluating the polarity, positive or negative, of a given piece of text. The IMDb movie reviews dataset is employed for training, and to boost the models' performance, an in-depth exploration of preprocessing, tokenizing, and feature extraction techniques is conducted.

## Overview

![lstmcell](https://github.com/Themiscodes/NLP-Sentiment-Analysis/assets/73662635/db808618-fc20-4105-b85d-dd7a6eda475b)

The [experiments](experiments/) directory showcases the construction of the RNN architectures with stacked LSTM and GRU cells, integrating attention mechanisms to capture important patterns in the input sequences. Additionally, gradient clipping, and skip connections are extensively explored to improve convergence and generalization. Gradient clipping prevents exploding gradients by capping them to a predefined threshold, stabilizing model weight updates. Skip connections enable better gradient flow, addressing the vanishing gradient problem, and enhancing learning of long-range dependencies.

The [notebooks](notebooks/) directory contains the implementations of the top-performing models for each architecture, accompanied by an extensive analysis of preprocessing techniques and hyperparameter tuning, utilizing the Optuna framework. For preprocessing, both lemmatization and stemming methods were trialed. Logistic regression [^1] utilized the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer, whereas the Feedforward network used Stanford's GloVe embeddings [^2]. The RNN models incorporated multiheaded attention, inspired by the "Attention is All You Need" paper [^3]. The final model explored was BERT-based, obtained from Hugging Face [^4].

### Performance Comparison

|   |  Logistic Regression | FNN  | LSTM  |  GRU  | BERT|
|---|---|---|---|---|---|
| Accuracy | 89.91%  | 85.41%  | 90.79%  |  90.99% | 92.00%|
| Macro F-1|  90.18% |  85.31% |  90.78% |  90.99% |92.00% | 
| Precision  | 90.03%  | 85.40%  |  91.32% | 90.09%  |91.48% |
| Recall  | 90.32%  |  85.26% | 90.12%  |  92.08% | 92.50% |

The BERT-based model, with its larger parameter size, demonstrated the best performance across all metrics. Similarly, the RNN architectures, incorporating stacked bidirectional LSTM and GRU cells with attention mechanisms, achieved comparable results. This architecture enabled the network to capture the overall sentiment context of a review, surpassing the limitations of individual words or sentences.

Logistic regression exhibited strong performance; however, its effectiveness is significantly influenced by the quality of the features and the nature of the NLP task. In contrast, the feedforward network underperformed due to the absence of a memory component, limiting their ability to process sequential data like text. For further analysis, and evaluation metrics, refer to the corresponding notebooks.

## References

[^1]: [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James H. Martin.

[^2]: [Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) Jeffrey Pennington, Richard Socher and Christopher D. Manning

[^3]: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

[^4]: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.
