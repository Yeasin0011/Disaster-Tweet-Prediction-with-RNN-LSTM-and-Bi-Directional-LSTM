# DisasterTweetPrediciton
Disaster Tweet Prediction using 
n
002 This report examines tweet sentiment analysis us-
003 ing a dataset of tweets classified as (0) Not-disaster
004 or (1) Disaster. We utilize three models: a Shallow
005 RNN, a Unidirectional LSTM, and a Bidirectional
006 LSTM, to evaluate their effectiveness in identifying
007 disaster-related sentiments. By training and assess-
008 ing these models, we explore their performance in
009 terms of accuracy, precision, recall, and F1 score.
010 The results highlight the strengths and weaknesses
011 of each architecture, providing insights into their
012 applicability for sentiment analysis in disaster con-
013 texts.
014 2 Data Description
015 Our dataset consists of the text in the tweet and the
016 event label for each tweet.
017 1. Tweet: This is the sequence of sentences for
018 each tweet.
019 2. Label(event): This is to determine if it is a
020 disaster or not a disaster tweet.
021 2.1 Data-preprocessing
022 2.1.1 Splitting data
023 At first, we extract the data from the CSV file and
024 split it into testing and training sets with 10% and
025 90%. From this 90%, the Xtrain data will later use
026 10% for validation. The random_state variable is
027 set to 1000, which means we randomly pick the
028 data from the dataset while splitting to ensure there
029 is no connection in the order of the data.
030 2.1.2 Label Split
031 Graphical representation of the available tweets
032 distinguished by their labels. As shown in Figure 1,
033 the distribution of labels for the events is illustrated,
034 providing insights into the label counts.
Figure 1: Label count for events
2.2 Tokenizing text sequences to word 035
sequences 036
We begin by initializing the Tokenizer and fit- 037
ting it on the training data (tweet_train) to create 038
a word index based on the frequency of unique 039
words. The tweets are then converted into se- 040
quences of integers using texts_to_sequences, 041
enabling numerical representation for model train- 042
ing. The vocabulary size is calculated by obtaining 043
the length of the word index, with an additional 044
unit to account for a reserved index. Finally, the 045
code outputs a sample tweet, its corresponding in- 046
teger sequence, the vocabulary size, and the maxi- 047
mum sequence lengths for both training and testing 048
datasets, preparing the text data for neural network 049
processing .
Figure 2: Sequence distribution after tokenization
1
Enhancing Disaster Tweet Prediction with Recurrent Neural Networks050 2.3 Adding Padding to Sequence
051 The maximum sequence length (maxlen) is printed
052 to ensure proper padding of the tweet sequences.
053 The pad_sequences function is then applied to
054 both the training (X_train) and testing (X_test)
055 datasets. This function adds padding to the se-
056 quences such that they all have the same length,
057 determined by maxlen, and specifies that padding
058 should be added at the end of the sequences
059 (padding=’post’). After padding, a sample from
060 the training tensor is displayed, along with the
061 shapes of both the training and testing tensors to
062 verify their dimensions before model training.
063 2.4 Embedding with GloVe
064 The create_embedding_matrix function initial-
065 izes an embedding matrix with dimensions based
066 on the vocabulary size and the specified embedding
067 dimension (embedding_dim). The vocabulary size
068 is calculated as the length of the word_index plus
069 one to account for a reserved index. The function
070 reads the GloVe embeddings from the specified
071 filepath, where each line contains a word fol-
072 lowed by its corresponding vector representation.
073 For each word in the word_index, its index is re-
074 trieved, and the corresponding vector is assigned to
075 the appropriate row in the embedding matrix. After
076 creating the embedding matrix, its shape is printed.
077 Additionally, the number of non-zero elements in
078 the matrix is calculated and divided by the vocabu-
079 lary size, indicating that approximately 56.2% of
080 the vocabulary has corresponding embeddings in
081 the matrix.
082 3 Model Architectures
083 In this section, we present three different neural net-
084 work architectures designed for text classification
085 tasks. Each model leverages an embedding layer
086 for word representation and varies in the choice
087 of recurrent layers and structural components to
088 capture the sequence dynamics of the input data.
089 Below, provided a brief overview of each model’s
090 architecture and configuration.
091 3.1 Model 01: Simple Shallow Sequential
092 Model
093 Model 01 is a straightforward architecture that be-
094 gins with an embedding layer, which transforms the
095 input sequences into dense vector representations
Figure 3: Basic RNN Model
using pre-trained embeddings (GloVe). Following 096
the embedding layer, a Global Max Pooling layer 097
is utilized to extract the most significant features, 098
which is then followed by two dense layers, with 099
the final layer employing a sigmoid activation func- 100
tion to predict binary outcomes. This model is 101
optimized for efficiency and simplicity, making it 102
suitable for baseline performance evaluation. 103
3.2 Model 02: LSTM Model 104
Figure 4: LSTM Model.
Model 02 incorporates a Long Short-Term Mem- 105
ory (LSTM) layer to capture temporal dependen- 106
cies in the input sequences. The model starts with 107
an embedding layer, followed by an LSTM layer 108
that processes the sequential data. A dropout layer 109
is added to prevent overfitting, followed by a dense 110
output layer with a sigmoid activation. This archi- 111
tecture aims to enhance the model’s ability to learn 112
long-range dependencies in the text data, thereby 113
improving classification performance. 114
3.3 Model 03: Bidirectional LSTM Model 115
Model 03 enhances the LSTM architecture by uti- 116
lizing a Bidirectional LSTM layer, which processes 117
the input sequences in both forward and backward 118
directions. This allows the model to capture context 119
2Figure 5: Bi-directional LSTM Model.
120 from both past and future words in the sequences.
121 The structure includes an embedding layer, a bidi-
122 rectional LSTM layer, and a dropout layer for regu-
123 larization, culminating in a dense output layer with
124 a sigmoid activation. This model is designed to
125 provide a more comprehensive understanding of
126 the text by leveraging context from both directions,
127 which is expected to yield better classification re-
128 sults.
129 4 Training, Testing and Result analysis
130 4.1 Training and Testing Steps
131 For each of the three models, training begins with
132 the implementation of early stopping to monitor
133 validation loss and prevent overfitting. The mod-
134 els are trained for up to 20 epochs with a batch
135 size of 32, using a validation split of 10% to as-
136 sess performance during training. After training,
137 the models are evaluated on both the training and
138 testing datasets to calculate loss and accuracy. Pre-
139 dictions are made on both datasets, and various met-
140 rics—including accuracy, precision, recall, and F1
141 score—are computed to provide a comprehensive
142 evaluation of each model’s performance. These
143 metrics help in understanding the models’ ability
144 to generalize beyond the training data
