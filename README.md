# Enhancing Disaster Tweet Prediction with Recurrent Neural Networks

Introduction
This report examines tweet sentiment analysis using a dataset of tweets classified as (0) Not-disaster or (1) Disaster. We utilize three models: a Shallow RNN, a Unidirectional LSTM, and a Bidirectional LSTM, to evaluate their effectiveness in identifying disaster-related sentiments. By training and assessing these models, we explore their performance in terms of accuracy, precision, recall, and F1 score. The results highlight the strengths and weaknesses of each architecture, providing insights into their applicability for sentiment analysis in disaster contexts.


Data Description
Our dataset consists of the text in the tweet and the event label for each tweet.
1. Tweet: This is the sequence of sentences for each tweet.
2. Label(event): This is to determine if it is a disaster or not a disaster tweet.

Dataset: https://www.kaggle.com/competitions/nlp-getting-started/code 

Data-preprocessing
At first, we extract the data from the CSV file and split it into testing and training sets with 10% and 90%. From this 90%, the  data will later use 10% for validation. The  variable is set to 1000, which means we randomly pick the data from the dataset while splitting to ensure there is no connection in the order of the data.

Tokenizing text sequences to word sequences
We begin by initializing the  and fitting it on the training data to create a word index based on the frequency of unique words. The tweets are then converted into sequences of integers using , enabling numerical representation for model training. The vocabulary size is calculated by obtaining the length of the word index, with an additional unit to account for a reserved index. Finally, the code outputs a sample tweet, its corresponding integer sequence, the vocabulary size, and the maximum sequence lengths for both training and testing datasets, preparing the text data for neural network processing .

Embedding with GloVe
The  function initializes an embedding matrix with dimensions based on the vocabulary size and the specified embedding dimension . The vocabulary size is calculated as the length of the  plus one to account for a reserved index. The function reads the GloVe embeddings from the specified , where each line contains a word followed by its corresponding vector representation. For each word in the , its index is retrieved, and the corresponding vector is assigned to the appropriate row in the embedding matrix. After creating the embedding matrix, its shape is printed. Additionally, the number of non-zero elements in the matrix is calculated and divided by the vocabulary size, indicating that approximately 56.2\% of the vocabulary has corresponding embeddings in the matrix.


Conclusion
In this analysis, we evaluated the performance of three models based on key metrics such as accuracy, precision, recall, and F1 score across both training and testing phases. Model 03 exhibited the most consistent and best overall performance, demonstrating superior generalization, with the highest testing accuracy (0.8163) and F1 score (0.7859). Model 02 showed better training metrics but a more significant drop in recall during testing, suggesting potential overfitting. Model 01 had the weakest performance, particularly in testing, where it struggled with both accuracy and F1 score, pointing to overfitting as well. To improve the overall generalization and performance consistency of these models, strategies such as further hyperparameter tuning, cross-validation, and regularization techniques could be applied. Additionally, focusing on improving recall, particularly in testing phases, would help ensure better identification of relevant cases across the board. Implementing these improvements may help reduce the gap between training and testing metrics, leading to more robust models.
