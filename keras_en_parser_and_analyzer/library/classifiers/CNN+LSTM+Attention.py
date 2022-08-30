from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.models import model_from_json, Sequential
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from attention import Attention



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from sklearn.metrics import classification_report

class WordVecCnnLstm(object):
    model_name = 'wordvec_cnn_lstm'
    
    

    
    def __init__(self):
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecCnnLstm.model_name + '_architecture.json'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecCnnLstm.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecCnnLstm.model_name + '_config.npy'

    def load_model(self, model_dir_path):
        json = open(self.get_architecture_file_path(model_dir_path), 'r').read()
        self.model = model_from_json(json)
        self.model.load_weights(self.get_weight_file_path(model_dir_path))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        config_file_path = self.get_config_file_path(model_dir_path)

        self.config = np.load(config_file_path).item()

        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']

    def create_model(self):
        lstm_output_size = 70
        embedding_size = 100
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, input_length=self.max_len, output_dim=embedding_size))
        self.model.add(SpatialDropout1D(0.5))
        self.model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(LSTM(lstm_output_size,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
        self.model.add(Attention(64))
        self.model.add(Dense(units=len(self.labels), activation='softmax'))

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def fit(self, text_data_model, text_label_pairs, model_dir_path, batch_size=None, epochs=None,
            test_size=None, random_state=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 20
        if test_size is None:
            test_size = 0.3
        if random_state is None:
            random_state = 42

        self.config = text_data_model
        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']
        import numpy as np
        np.save(self.get_config_file_path(model_dir_path), self.config)

        self.create_model()
        json = self.model.to_json()
        open(self.get_architecture_file_path(model_dir_path), 'w').write(json)

        xs = []
        ys = []
        for text, label in text_label_pairs:
            tokens = [x.lower() for x in word_tokenize(text)]
            wid_list = list()
            for w in tokens:
                wid = 0
                if w in self.word2idx:
                    wid = self.word2idx[w]
                wid_list.append(wid)
            xs.append(wid_list)
            ys.append(self.labels[label])

        X = pad_sequences(xs, maxlen=self.max_len)
        Y = np_utils.to_categorical(ys, len(self.labels))

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)
        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=[x_test, y_test], callbacks=[checkpoint],
                                 verbose=1)
        self.model.save_weights(weight_file_path)
        model=self.model

        
        
        y_pred=model.predict(x_test, batch_size=batch_size)
        
        for i in range(len(y_pred)):
            max_value=max(y_pred[i])
            for j in range(len(y_pred[i])):
                if max_value==y_pred[i][j]:
                    y_pred[i][j]=1
                else:
                    y_pred[i][j]=0
        print(classification_report(y_test, y_pred,digits=4))
        
        # 混淆矩阵定义
        from sklearn.metrics import confusion_matrix

        import numpy as np
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        def plot_confusion_matrix(cm,
                                target_names,
                                title='Confusion matrix',
                                cmap=plt.cm.Greys,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                                normalize=True):
        
        
            accuracy = np.trace(cm) / float(np.sum(cm))
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            plt.figure(figsize=(5, 4))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            #plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")


            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
            #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
            #plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
            plt.show()
        # 显示混淆矩阵
        def plot_confuse(model, x_val, y_val):
            predictions = model.predict_classes(x_val,batch_size=batch_size)
            truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
            conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
            plt.figure()
            plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')
        #=========================================================================================
        #最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
        #labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
        #比如这里我的labels列表
        labels=['experience','knowledge','education','project','others']

        plot_confuse(model, x_test,y_test)
        
        np.save(model_dir_path + '/' + WordVecCnnLstm.model_name + '-history.npy', history.history)

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print('score: ', score[0])
        print('accuracy: ', score[1])

        return history

    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else len(self.word2idx) for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.max_len)
        output = self.model.predict(x)
        return output[0]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

    def test_run(self, sentence):
        print(self.predict(sentence))


def main():
    app = WordVecCnnLstm()
    app.test_run('i liked the Da Vinci Code a lot.')


if __name__ == '__main__':
    main()
