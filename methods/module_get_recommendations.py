import json


# 운동 추천 함수
def get_recommend(data, exercise_movement, cosine_similarities):
    idx = data.index[data['운동 동작'] == exercise_movement].tolist()[0]  # 운동 동작 인덱스 찾기

    # 코사인 유사도 점수 정렬
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # 유사도 x[1] 기준
    sim_scores = sim_scores[1:11]  # 상위 10개 아이템 선택

    # 선택된 운동들의 인덱스 및 유사도 저장
    exercise_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # 결과 저장 리스트 초기화
    result_list = []

    for exercise_index, similarity in zip(exercise_indices, similarity_scores):
        result = [
            data['운동 동작'].iloc[exercise_index],
            round(similarity, 4),
        ]

        json_result = json.dumps(result, ensure_ascii=False)
        print(json_result)

        result_list.append(result)

    return result_list
