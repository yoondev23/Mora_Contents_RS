
# 참고1(파이썬 라이브러리) : pip install pandas  # 데이터 조작 시 필요
# 참고2(파이썬 라이브러리) : pip install pyarrow  # Pandas 속도 효율
# 참고3(파이썬 라이브러리) : pip install scikit-learn  # 머신러닝 라이브러리

import pandas as pd

from sklearn.metrics.pairwise import linear_kernel  # 두 벡터 간의 유사도 계산
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF를 기반 Vectorizer 객체 생성

from methods.module_get_recommendations import get_recommend
from methods.module_stop_words import stop_words  # 불용어 제거


mora = pd.read_csv('data/mora_3000.csv')
mora['설명'] = mora['설명'].fillna('')  # na 삭제

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = tfidf_vectorizer.fit_transform(
    mora['대분류'] + ' ' + mora['난이도'] + ' ' + mora['부위'] + ' ' + mora['도구'] + ' ' + mora['목적'] + ' ' + mora['설명'])

# 단어의 중요성 반영한 단어 목록 반환
terms = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=terms)

# 코사인 유사도(문서 유사도 산출 기법 중 하나) 계산
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# 추천 받을 운동 동작 입력
exercise_movement = input("유사한 운동을 추천받을 동작을 입력하세요 ex)거꾸로 플랭크 : ")
recommendation_reason_list = get_recommend(mora, exercise_movement, cosine_similarities)
