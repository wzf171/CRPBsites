import numpy as np
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import  train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,roc_auc_score
import keras
from keras.models import Model
from keras.layers import Dense,Bidirectional,LSTM,Permute,Input,Flatten,multiply,Embedding,RepeatVector,TimeDistributed
from keras.utils import np_utils
from matplotlib import pyplot as plt
import seaborn as sns
import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def confustion_matrix(labels,predicions):
    LABELS=['POS','NEG']
    matrix=metrics.confusion_matrix(labels,predicions)
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix,cmap='coolwarm',linecolor='white',linewidths=1,xticklabels=LABELS,yticklabels=LABELS,annot=True,fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def attention(decoder_out):
    weight=Permute((2,1))(decoder_out)
    weight=Dense(TIME_STEP,activation='softmax')(weight)
    weight = Permute((2, 1))(weight)
    attention_output = multiply([decoder_out, weight])
    return attention_output

def seq2ngram(seq_path,k,s,model):
    with open(seq_path, "r") as fr:
        lines = fr.readlines()
    fr.close()
    list_full_text=[]
    for line in lines:
        if line.startswith(">hsa") or len(line) <= 1:
            continue
        else:
            line = line[:-1].upper()
            seq_len = len(line)
            list_line = []
            for index in range(0, seq_len, s):
                if index + k >= seq_len + 1:
                    break
                list_line.append(line[index:index+k])
            word_index=[]
            for word in list_line:
              if word in model.wv:
                  word_index.append(model.wv.vocab[word].index)
            list_full_text.append(word_index)
    return list_full_text

def SeqNet():
    word2vec_model = word2vec.Word2Vec.load("wv/"+PROTEIN+"_circ_seq_word_vec")
    pos_path = "../Data/"+PROTEIN+"/pos_seq.fa"
    pos_list = seq2ngram(pos_path, 3, 1, word2vec_model)
    neg_path = "../Data/"+PROTEIN+"/neg_seq.fa"
    neg_list = seq2ngram(neg_path, 3, 1, word2vec_model)
    seq_list = pos_list + neg_list
    feature = pad_sequences(seq_list, maxlen=TIME_STEP, padding="post",value=0)
    label = [1] * len(pos_list) + [0] * len(neg_list)
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=0)

    embedding_matrix = np.zeros((len(word2vec_model.wv.vocab), EMBEDDING_DIM))
    for i in range(len(word2vec_model.wv.vocab)):
        embedding_vector = word2vec_model.wv[word2vec_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    inputs = Input(shape=(TIME_STEP,))
    embeding=Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        trainable=True)(inputs)

    # Encoder layers
    encoder_out = Bidirectional(LSTM(ENCODER_UNITS, return_sequences=True))(embeding)
    encoder_out = Bidirectional(LSTM(ENCODER_UNITS, return_sequences=False))(encoder_out)
    encoder_out = RepeatVector(TIME_STEP)(encoder_out)
    # Decoder layers
    decoder_out = LSTM(DECODER_UNITS, return_sequences=True)(encoder_out)
    # Depending on the size of the data, double-layer decoding can be used
    # decoder_out = LSTM(DECODER_UNITS, return_sequences=True)(decoder_out)
    decoder_out = TimeDistributed(Dense(DECODER_UNITS // 2))(decoder_out)
    attention_out = attention(decoder_out)
    attention_out = Flatten()(attention_out)
    fc_cout = Dense(DECODER_UNITS)(attention_out)
    fc_cout = Dense(DECODER_UNITS // 2)(fc_cout)
    outputs = Dense(units=2, activation='softmax')(fc_cout)
    model=Model(input=[inputs],output=outputs)

    print(model.summary())

    print('\n****** Fit the model ******\n')
    # The EarlyStopping callback monitors training accuracy:
    # if it fails to improve for two consecutive epochs,training stops early
    time_callback=TimeHistory()
    callbacks_list = [time_callback,
        keras.callbacks.ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                        save_best_only=True), keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Small learning rate, slow convergence speed, suitable for small-scale data sets
    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.adam(lr=1e-04), metrics=['accuracy'])
    # Fit the model
    y_train = np_utils.to_categorical(y_train, NUMCLASSES)
    y_test = np_utils.to_categorical(y_test, NUMCLASSES)
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list,
                        validation_split=0.1, verbose=1)
    # model.save('CnnCircRBPS.h5')
    # EPOCH time
    print("Time per iteration:\n")
    print(time_callback.times)
    print("Total time:\n")
    # Total time
    print(time_callback.totaltime)

    print('\n****** Check on test set ******\n')
    # evaluate the model on the test set
    score = model.evaluate(x_test, y_test, verbose=1)
    print('\nLoss on test set: %0.4f' % score[0])
    print('\nAccuracy on test set: %0.4f' % score[1])

    print('\n****** Confusion matrix on test set')
    # prediction
    y_pred_prob = model.predict(x_test)
    y_pred_label = np.argmax(y_pred_prob, axis=1)
    # take the class with the highest probability on the test set prediction
    y_test_label = np.argmax(y_test, axis=1)
    confustion_matrix(y_test_label, y_pred_label)

    print('\n****** Classification report on test set ******\n')
    print(classification_report(y_test_label, y_pred_label,digits=4))


    print('\n****** calculate the AUC ******\n')
    auc_ = roc_auc_score(y_test_label, y_pred_prob[:, 1])
    print('\nThe AUC on the test set is: %0.4f' % auc_)
    f = open(str(PROTEIN), 'w')
    print(str(auc_), file=f)
    f.close()
    return auc_

if __name__ == "__main__":
    # hyper-parameter
    EMBEDDING_DIM=32
    PROTEIN="LIN28B"
    TIME_STEP = 80
    NUMCLASSES = 2
    BATCH_SIZE = 512
    EPOCHS = 30
    ENCODER_UNITS=32
    DECODER_UNITS=16
    # run the model
    auc_ = SeqNet()
    line = str(PROTEIN) + ":" + str(auc_)
    file_handle = open('AUC.txt', mode='a+')
    file_handle.write(line + "\n")
    file_handle.close()