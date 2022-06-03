import numpy as np
import re
import pickle
from konlpy.tag import Okt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class RestarantReviewClassifier:
    def __init__(self):
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.model = tf.keras.models.load_model('mango_review.h5')
        self.tagger = Okt()
    
    def predict(self, input_data):
        input_data = list(map(lambda x: re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', x), input_data))  # 한글 제외한 문자 삭제
        input_data = list(map(lambda x: re.sub('\s{2,}', ' ', x), input_data))

        X_data = []
        s_w = set(['은', '는', '이', '가', '를', '들', '에게', '의', '을', '도', '으로', '만', '라서', '하다',
                '아', '로', '저', '즉', '곧', '제', '좀', '참', '응', '그', '딱', '어', '네', '예', '게', '고',
                '하', '에', '한', '어요', '것', '았', '네요', '듯', '같', '나', '있', '었', '지', '하고', '먹다',
                '습니다', '기', '시', '과', '수', '먹', '와', '적', '보', '에서', '곳', '너무', '정말', '진짜',
                '있다', '다', '더', '인', '집', '면', '내', '라', '원', '요', '또', '하나', '전', '거', '엔',
                '이다', '되다', '까지', '인데', '정도', '나오다', '주문', '시키다'])
        for i in input_data:
            tk_d = self.tagger.morphs(i, stem=True)  # clean_X의 형태소 추출
            tk_d = [w for w in tk_d if w not in s_w]
            X_data.append(' '.join(tk_d))
        
        X = self.tokenizer.texts_to_sequences(X_data)
        X = pad_sequences(X, maxlen=90)
        X = np.array(X)

        return np.array(list(map(lambda x: np.round(x, 0), self.model.predict(X))))