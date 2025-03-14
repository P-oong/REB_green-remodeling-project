import openai
import json

def gpt_match_building(e_row, registry_df):
    """
    e_row: 에너지소비량 보고 데이터의 한 행(pd.Series)
    registry_df: 해당 RECAP_PK에 속하는 건축물대장 후보들(DataFrame)
    """

    # 1) GPT에게 건물 정보(에너지 보고 건물) + 후보 표제부 목록을 전달할 프롬프트 생성
    energy_info = (
        f"- 에너지 보고 건물명: {e_row['BLD_NM']}\n"
        f"- 용도: {e_row['MAIN_USE']}\n"
        f"- 면적 구분: {e_row['AREA_CATEGORY']}\n"
    )

    # 후보 목록 요약
    candidates_text = ""
    for i, r in registry_df.iterrows():
        candidates_text += (
            f"{i+1}. [MGMT_PK: {r['MGMT_PK']}]\n"
            f"   건물명: {r['BLD_NAME']}\n"
            f"   용도: {r['MAIN_USE']}\n"
            f"   연면적: {r['TOTAL_AREA']}㎡\n\n"
        )

    prompt = f"""
            아래는 '에너지 사용량 보고'에 기록된 건물 정보입니다:
            {energy_info}

            그리고 아래는 같은 총괄표제부(RECAP_PK) 안에 속하는 '건축물대장 표제부' 후보 목록입니다:
            {candidates_text}

            위 정보를 바탕으로, 에너지 보고 건물이 후보 중 어느 건물에 가장 가깝다고 판단되는지 골라주세요.
            그리고 그 판단 이유를 간단히 설명해주세요.

            응답은 JSON 형태로만 제공해 주세요. 
            키는 "best_match" (MGMT_PK 값), "reason" (판단 이유) 이렇게 두 개만 포함해주세요.
            """

    # 2) GPT 호출
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 필요에 따라 gpt-4 등
            messages=[
                {"role": "system", "content": "당신은 건물 매칭을 도와주는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=512
        )
        gpt_reply = response['choices'][0]['message']['content'].strip()

        # 3) JSON 파싱
        # GPT가 JSON 형식을 잘 지킨다는 전제. (실무에선 예외처리 권장)
        result = json.loads(gpt_reply)

        best_match = result.get("best_match", None)
        reason = result.get("reason", "")

        return best_match, reason

    except Exception as e:
        # 에러가 나면 None 반환
        print(f"Error in GPT call: {e}")
        return None, None
