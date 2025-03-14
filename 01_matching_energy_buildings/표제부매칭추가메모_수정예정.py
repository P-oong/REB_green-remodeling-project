import openai
import json
import pandas as pd

openai.api_key = "YOUR_OPENAI_API_KEY"  # 안전하게 관리하세요

def gpt_match_building(e_row, registry_df):
    """
    e_row: 에너지소비량 보고 데이터의 한 행(pd.Series)
    registry_df: 해당 RECAP_PK에 속하는 건축물대장 후보들(DataFrame)
    """

    # 1) 에너지 보고 건물 정보 (BLD_NM, DONG_NM 추가)
    energy_info = (
        f"- 에너지 보고 건물명: {e_row.get('BLD_NM', 'N/A')}\n"
        f"- 에너지 보고 동 이름: {e_row.get('DONG_NM', 'N/A')}\n"
        f"- 용도: {e_row.get('MAIN_USE', 'N/A')}\n"
        f"- 면적 구분: {e_row.get('AREA_CATEGORY', 'N/A')}\n"
    )

    # 후보 목록 요약
    if registry_df.empty:
        candidates_text = "후보 없음\n"
    else:
        candidates_text = ""
        for i, r in registry_df.iterrows():
            # 후보 표제부의 건물명, 동 이름
            bld_name = r.get('BLD_NAME', 'N/A')
            dong_name = r.get('DONG_NAME', 'N/A')  # 실제 컬럼명에 맞게 수정
            main_use = r.get('MAIN_USE', 'N/A')
            total_area = r.get('TOTAL_AREA', 'N/A')
            candidates_text += (
                f"{i+1}. [MGMT_PK: {r['MGMT_PK']}]\n"
                f"   건물명: {bld_name}\n"
                f"   동 이름: {dong_name}\n"
                f"   용도: {main_use}\n"
                f"   연면적: {total_area}㎡\n\n"
            )

    # 2) LLM 프롬프트
    prompt = f"""
아래는 '에너지 사용량 보고'에 기록된 건물 정보입니다:
{energy_info}

그리고 아래는 같은 총괄표제부(RECAP_PK) 내에 속하는 '건축물대장 표제부' 후보 목록입니다:
{candidates_text}

이 정보를 바탕으로, 아래 네 가지 키를 JSON 형태로만 결정해 주세요.

1. best_match: 에너지 보고 건물과 가장 가깝다고 판단되는 후보의 MGMT_PK (후보가 없으면 null)
2. reason: 매칭 판단 이유를 간단히 설명
3. missing_registry: 만약 해당 RECAP_PK에 대해 후보 데이터가 전혀 없거나 매우 부족하면 1, 그렇지 않으면 0
4. ambiguous_match: 에너지 보고 건물명(BLD_NM)이나 동 이름(DONG_NM)에 '관리동' 등처럼 데이터 정보만으로는 매칭이 어려운 표현이 있으면 1, 아니면 0

응답은 반드시 JSON 형식으로만 제공해 주세요.
예시: {{"best_match": 101, "reason": "후보 중 건물명이 가장 유사함", "missing_registry": 0, "ambiguous_match": 1}}
"""

    # 3) GPT 호출
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 필요에 따라 gpt-4 등 다른 모델로 변경 가능
            messages=[
                {"role": "system", "content": "당신은 건물 매칭과 라벨링을 도와주는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=512
        )
        gpt_reply = response['choices'][0]['message']['content'].strip()

        # 4) JSON 파싱
        result = json.loads(gpt_reply)

        best_match = result.get("best_match", None)
        reason = result.get("reason", "")
        missing_registry = result.get("missing_registry", 0)
        ambiguous_match = result.get("ambiguous_match", 0)

        return best_match, reason, missing_registry, ambiguous_match

    except Exception as e:
        print(f"Error in GPT call: {e}")
        # 에러가 나면 기본값 반환
        return None, "", 0, 0

def perform_matching(energy_df, registry_df):
    """
    에너지 보고 데이터프레임(energy_df)의 각 행에 대해,
    같은 RECAP_PK를 가진 registry_df 후보 목록을 찾아
    LLM 매칭 및 라벨링 결과를 붙여주는 함수
    """
    results = []
    for idx, e_row in energy_df.iterrows():
        # RECAP_PK 기준 후보 추출
        recap = e_row['RECAP_PK']
        subset = registry_df[registry_df['RECAP_PK'] == recap]
        
        best_match, reason, missing_flag, ambiguous_flag = gpt_match_building(e_row, subset)
        results.append((best_match, reason, missing_flag, ambiguous_flag))

    # 결과 컬럼 4개를 추가
    energy_df[['BEST_MGMT_PK', 'MATCH_REASON', 'Missing_Registry', 'Ambiguous_Match']] = results
    return energy_df

# 예시 데이터
energy_data = pd.DataFrame({
    'SEQ_NO': [2676, 2677, 2678],
    'RECAP_PK': ['R1', 'R1', 'R1'],
    'BLD_NM': ['국립중앙과학관(과학기술관)', '국립중앙과학관(특별전시동)', '국립중앙과학관(관리동)'],
    'DONG_NM': ['1동', '2동', '관리동'],
    'MAIN_USE': ['문화및집회시설', '문화및집회시설', '문화및집회시설'],
    'AREA_CATEGORY': ['10,000㎡이상', '3,000㎡이상~5,000㎡미만', '3,000㎡이상~5,000㎡미만']
})

registry_data = pd.DataFrame({
    'MGMT_PK': [101, 102, 103, 104],
    'RECAP_PK': ['R1', 'R1', 'R1', 'R1'],
    'BLD_NAME': ['국립중앙과학관', '첨단연구성과종합전시관', '과학기술전시체험센터', '24.과학캠프관'],
    'DONG_NAME': ['1동', '2동', '1동', '3동'],
    'MAIN_USE': ['문화및집회시설', '문화및집회시설', '문화및집회시설', '교육연구시설'],
    'TOTAL_AREA': [30453.1, 6121.03, 6278.21, 4776.06]
})

# 매칭 실행
final_df = perform_matching(energy_data, registry_data)
print(final_df)
