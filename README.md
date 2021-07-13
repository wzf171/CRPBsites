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
Sklearn<br>

Usage
===
**1.** Configure the hyper-parameters (e.g. BATCH_SIZE, EPOCHS, ENCODER_UNITS and DECODER_UNITS, etc.), the name of RBP binding site data set (e.g. "LIN28B") and the TIME_STEP of input sequence length. Only the default values of the hyperparameters are provided in the program. If you need to migrate to a custom dataset, the parameters need to be adjusted according to the size of the data, such as the EPOCHS on IGF2BP1 dataset can be set to 5 to obtain better results. <br><br>
**2.** Run Word2vec, train the word2vec model and generate the word vectors of binding site sequence words. <br><br>
**3.** Run CRPBsites, train the CRPBsites predictor using the training set and use the test set for independent testing. The code for dividing the training set and the test set is provided in the program. The final evaluation metric is written to the text format, it can be found in the Auc directory (e.g. "CRPBsites/Demo/Code/AUC.txt").<br><br>
**4.** In addition, a full-length circRNA sequence can be scanned with a fixed length using a trained model to predict possible binding sites for a given RBP.<br>

Use case:
===
**step 1:** Prepare the data set required for the training model, such as the LIN28B data set provided by Demon. It can be found in the path: "CRPBsites/Demo/Data/LIN28B". 

**step 2:** Configure default hyper-parameters. "BATCH_SIZE": 512, "EPOCHS": 30, "ENCODER_UNITS": 32, "DECODER_UNITS": 16, "PROTEIN": "LIN28B", "TIME_STEP": 80 and etc.

**step 3:** Run circRB("CRPBsites/Demo/Code/CRPBsites.py"), The final evaluation metric is written to "AUC.txt" which is saved in the path: "CRPBsites/Demo/Code/AUC.txt". The trained circRB predictor could be found in the path: "CRPBsites/Demo/Code/Models/".
