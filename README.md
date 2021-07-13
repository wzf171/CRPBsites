# CRPBsites
Code for my paper on"Prediction of RBP binding sites on circRNAs using a LSTM-based deep sequence learning architecture"

Prediction of RBP binding sites on circRNAs using a LSTM-based deep sequence learning architecture
===
Interaction with RNA-binding protein (RBP) to influence post-transcriptional regulation is considered to be an important pathway for circRNA function. We design a deep learning framework, CRPBsites, to predict the binding sites of RBPs on circRNAs. In this model, the sequences of variable-length binding sites are transformed into embedding vectors by word2vec model. Bidirectional LSTM is used to encode the embedding vectors of binding sites, and then they are fed into another LSTM decoder for decoding and classification tasks. 

Dependency:
===
Python 3.7.4<br>
Keras 2.3.1<br>
Gensim 3.8.3<br>
Numpy 1.19.0<br>
Pandas 1.0.1<br>
NumPy 1.19.0<br>
Sklearn.<br> 

Usage
===
**1.** Configure the hyper-parameters (e.g. BATCH_SIZE, EPOCHS, ENCODER_UNITS and DECODER_UNITS, etc.), the name of RBP binding site data set (e.g. "LIN28B") and the TIME_STEP of input sequence length.<br><br>
**2.** Run Word2vec_train.py, train the word2vec model and generate the word vectors of binding site sequence words. <br><br>
**3.** Run CRPBsites.py, train the CRPBsites predictor using the training set and use the test set for independent testing. The code for dividing the training set and the test set is provided in the program. The final evaluation metric is written to the text format, it can be found in the Auc directory (e.g. "CRPBsites/DEMO/Code/AUC.txt").<br><br>
**4.** In addition, a full-length circRNA sequence can be scanned with a fixed length using a trained model to predict possible binding sites for a given RBP.<br>
