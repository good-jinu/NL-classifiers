from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


class NLClassifier:
    def __init__(self, max_words=8000, pad_len=50, embedding_size=100, btsize = 128,
                 d_out=0.5, neuron_num=64, bidir=True, layer2=True, gru=False, output_dim=1):
        self.max_words = max_words
        self.pad_len = pad_len
        self.btsize=btsize
        self.lbe = LabelEncoder()
        self.m = Sequential()
        self.tk = None
        neuron_num = 2 if neuron_num < 2 else neuron_num
        lstmorgru = layers.GRU if gru else layers.LSTM
        self.m.add(layers.Embedding(self.max_words, embedding_size, input_length=pad_len))
        if bidir:
            self.m.add(layers.Bidirectional(lstmorgru(neuron_num, return_sequences=layer2)))
            if layer2:
                self.m.add(layers.Bidirectional(lstmorgru(neuron_num // 2)))
            self.m.add(layers.Dropout(d_out))
        else:
            self.m.add(lstmorgru(neuron_num, dropout=d_out, return_sequences=layer2))
            if layer2:
                self.m.add(lstmorgru(neuron_num // 2, dropout=d_out))
        self.m.add(layers.Dense(output_dim, activation='softmax' if output_dim > 2 else 'sigmoid'))
        self.m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    def preprocessing(self, df):
        X = df['X']
        y = None
        if 'y' in df.columns:
            y = self.lbe.transform(df['y'])
        return X, y

    def fit(self, x_data, y_data, pre_prc=True):
        X, y = x_data, y_data
        if pre_prc:
            df = pd.DataFrame()
            df['X'] = x_data
            df['y'] = y_data
            self.lbe.fit(df['y'])
            X, y = self.preprocessing(df)
        e_st = EarlyStopping(patience=2, restore_best_weights=True)
        return self.m.fit(X, y, epochs=20, validation_split=0.2, batch_size=self.btsize, callbacks=[e_st])

    def evaluate(self, x_data, y_data, pre_prc=True):
        X, y = x_data, y_data
        if pre_prc:
            df = pd.DataFrame()
            df['X'] = x_data
            df['y'] = y_data
            self.lbe.fit(df['y'])
            X, y = self.preprocessing(df)
        return self.m.evaluate(X, y)

    def predict(self, x_data, pre_prc=True):
        X = x_data
        if pre_prc:
            df = pd.DataFrame()
            df['X'] = x_data
            X, a = self.preprocessing(df)
        return self.m.predict(X)

class KorClassifier(NLClassifier):
    def preprocessing(self, df):
        df = df.dropna(how='any')
        df = df.drop_duplicates(subset=['X'])  # X??? ?????? ???????????? ??? ??????
        df['X'] = df.X.map(lambda x: re.sub('[^???-??????-??????-??? ]', '', x))  # ?????? ????????? ?????? ??????
        df['X'] = df.X.map(lambda x: re.sub('^ +', '', x))  # ?????? ?????? ??????
        df = df.loc[df.X != '']  # ???????????? ???????????? nan?????? ??????
        df = df.dropna(how='any')
        s_w = set(['???', '???', '???', '???', '???', '???', '??????', '???', '???', '???', '??????', '???', '??????', '??????',
                  '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '??????', '??????', '??????'])
        # s_w.add(????????? ???????????????)
        okt = Okt()
        X_data = []
        for i in tqdm(df['X']):
            tk_d = okt.morphs(i, stem=True)  # clean_X??? ????????? ??????
            end_d = [w for w in tk_d if not w in s_w]  # ???????????? ?????? ???????????? ???????????? ??????
            X_data.append(' '.join(end_d))
        if self.tk is None:
            self.tk = Tokenizer(num_words=self.max_words)
            self.tk.fit_on_texts(X_data)
        X_data = self.tk.texts_to_sequences(X_data)
        drop_X = [index for index, sentence in enumerate(X_data) if len(sentence) < 1]
        X_data = pad_sequences(X_data, maxlen=self.pad_len)
        X_data = np.delete(X_data, drop_X, axis=0)
        X = X_data
        y = None
        if 'y' in df.columns:
            y = self.lbe.transform(df['y'])
            y = np.delete(y, drop_X, axis=0)
            y = to_categorical(y)
        return X, y

class EngClassifier(NLClassifier):
    def __engpreprocess(self, X_text, remove_stopwords=True):
        X_text = BeautifulSoup(X_text, 'lxml').get_text()  # html ?????? ??????
        X_text = re.sub("[^a-zA-Z]", " ", X_text)  # ????????? ????????? ????????? ???????????? ??????
        words = X_text.lower().split()  # ?????? ???????????? ????????? ?????? ???????????? ?????????.
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            # stops.add(????????? ?????????)
            words = [w for w in words if not w in stops]  # ???????????? ????????? ????????? ???????????? ??????
            clean_text = ' '.join(words)  # ???????????? ????????? ????????? ?????? ????????? ???????????? ??????
        else:
            clean_text = ' '.join(words)
        return clean_text
    def preprocessing(self, df):
        df = df.dropna(how='any')
        df['X'] = df['X'].apply(lambda x: self.__engpreprocess(X_text=x, remove_stopwords=True))
        df['X'] = df['X'].str.replace("[^a-zA-Z0-9 ]", "")
        df['X'] = df['X'].str.replace('^ +', "")
        df['X'].replace('', np.nan, inplace=True)
        df = df.dropna(how='any')
        X_data = df['X']
        if self.tk is None:
            self.tk = Tokenizer(num_words=self.max_words)
            self.tk.fit_on_texts(X_data)
        X_data = self.tk.texts_to_sequences(X_data)
        drop_X = [index for index, sentence in enumerate(X_data) if len(sentence) < 1]
        X_data = pad_sequences(X_data, maxlen=self.pad_len)
        X_data = np.delete(X_data, drop_X, axis=0)
        X = X_data
        y = None
        if 'y' in df.columns:
            y = self.lbe.transform(df['y'])
            y = np.delete(y, drop_X, axis=0)
            y = to_categorical(y)
        return X, y