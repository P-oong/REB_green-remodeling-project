import os
import json
import pandas as pd
import openai
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
openai.api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API 키
#langsmith_api_key = os.getenv("LANGSMITH_API_KEY")  # 필요시 사용

# ------------------------------------------------
# 1) 면적 텍스트 전처리 함수
# ------------------------------------------------
def parse_area_text(area_str):
    """
    EBD의 AREA 컬럼(예: "3,000㎡이상~5,000㎡미만", "5,000㎡이상~10,000㎡미만", "10,000㎡이상")을
    (lower_bound, upper_bound) 형태로 파싱.
    upper_bound == None 이면 상한이 없는 것(이상만).
    """
    if not isinstance(area_str, str):
        return (None, None)

    s = area_str.replace(",", "").replace("㎡", "").strip()
    # 예: "3000이상~5000미만", "5000이상~10000미만", "10000이상"
    lower, upper = None, None

    # A) "이상~" 있는 형태: "3000이상~5000미만"
    if "이상~" in s:
        parts = s.split("이상~")
        # parts[0] = "3000"
        # parts[1] = "5000미만" (or "5000")
        try:
            lower = float(parts[0])
        except:
            lower = None

        if "미만" in parts[1]:
            up_str = parts[1].replace("미만", "").strip()
            try:
                upper = float(up_str)
            except:
                upper = None
        else:
            upper = None

    # B) "이상"만 있는 경우: "10000이상"
    elif "이상" in s:
        val_str = s.replace("이상","").strip()
        try:
            lower = float(val_str)
        except:
            lower = None
        upper = None

    # C) (필요하다면 "미만"만 있는 케이스 추가)

    return (lower, upper)

# ------------------------------------------------
# 2) 1차 규칙 기반 매칭용 점수 계산
# ------------------------------------------------
def compute_rule_score_details(e_row, c_row):
    """
    e_row: EBD 한 행
    c_row: BD 후보 한 행
    각 항목별 점수를 개별 변수로 기록, 사유를 dict에 저장
      - usage_score
      - text_score
      - area_score
    최종 total_score 반환
    """
    details = {}
    reasons = {}

    # (1) 용도 점수: PUR_NM vs ETC_PURPS(또는 MAIN_USE)
    pur_nm = str(e_row.get('PUR_NM','')).lower()
    etc_purps = str(c_row.get('ETC_PURPS', c_row.get('MAIN_USE',''))).lower()
    if pur_nm and pur_nm in etc_purps:
        details["usage_score"] = 1
        reasons["usage"] = "PUR_NM in ETC_PURPS"
    else:
        details["usage_score"] = 0
        reasons["usage"] = "PUR_NM mismatch"

    # (2) 텍스트 결합 점수
    # EBD: OFFICE_NM + BLD_NM
    combined_ebd = (str(e_row.get('OFFICE_NM','')) + " " + str(e_row.get('BLD_NM',''))).strip().lower()
    # BD: BLD_NM + DONG_NM
    combined_bd = (str(c_row.get('BLD_NM','')) + " " + str(c_row.get('DONG_NM',''))).strip().lower()

    if combined_ebd and combined_bd and (combined_ebd in combined_bd):
        details["text_score"] = 1
        reasons["text"] = f"'{combined_ebd}' in '{combined_bd}'"
    else:
        details["text_score"] = 0
        reasons["text"] = "text mismatch"

    # (3) 면적 점수: EBD(문자열 범위) vs BD(TOTAREA)
    area_str = e_row.get('AREA','')
    totarea = float(c_row.get('TOTAREA',0))
    lower, upper = parse_area_text(area_str)
    if lower is None and upper is None:
        details["area_score"] = 0
        reasons["area"] = "Area range parse failed"
    else:
        ok_lower = (lower is None) or (totarea >= lower)
        ok_upper = (upper is None) or (totarea < upper)
        if ok_lower and ok_upper:
            details["area_score"] = 1
            reasons["area"] = f"{totarea} in range [{lower if lower else 0}, {upper if upper else '∞'})"
        else:
            details["area_score"] = 0
            reasons["area"] = f"{totarea} not in range"

    total_score = sum(details.values())
    rule_details = "; ".join([f"{k}:{details[k]}({reasons[k]})" for k in details])
    return total_score, rule_details

def rule_based_match_multi(e_row, candidate_df):
    """
    MULTI_YN='Y' 인 건물의 1차 규칙 기반 매칭
    - 각 후보별 score와 rule_details 계산
    - 최고 점수가 2점 이상이고 유일하면 매칭
    """
    if candidate_df.empty:
        return None, "No candidate"

    scores = []
    detail_list = []
    for _, c_row in candidate_df.iterrows():
        s, rule_str = compute_rule_score_details(e_row, c_row)
        scores.append(s)
        detail_list.append(rule_str)

    df_copy = candidate_df.copy()
    df_copy['total_score'] = scores
    df_copy['rule_details'] = detail_list

    max_score = df_copy['total_score'].max()
    best = df_copy[df_copy['total_score'] == max_score]

    if max_score >= 2 and len(best) == 1:
        br = best.iloc[0]
        return br['MGM_BLD_PK'], br['rule_details']
    else:
        return None, "1st rule-based match failed"

# ------------------------------------------------
# 3) 2차 작업: GPT API
# ------------------------------------------------
def build_user_prompt(e_row, candidate_df):
    """
    EBD 건물 + BD 후보 목록 → 최종 GPT user prompt
    """
    combined_ebd = (str(e_row.get('OFFICE_NM', '')) + " " + str(e_row.get('BLD_NM', ''))).strip()
    energy_info = f"""
        [Energy Report Building]
        - SEQ_NO: {e_row.get('SEQ_NO')}
        - RECAP_PK: {e_row.get('RECAP_PK')}
        - Combined Name: {combined_ebd}
        - Usage (PUR_NM): {e_row.get('PUR_NM', '')}
        - Floor Area Category (AREA): {e_row.get('AREA', '')}
        """

    cand_text = "[Candidate Building Registry]\n"
    for i, c_row in candidate_df.iterrows():
        cmb_bd = (str(c_row.get('BLD_NM', '')) + " " + str(c_row.get('DONG_NM', ''))).strip()
        cand_text += f"""
                    {i+1}. [MGM_BLD_PK: {c_row['MGM_BLD_PK']}]
                    Combined Name: {cmb_bd}
                    Usage: {c_row.get('ETC_PURPS', c_row.get('MAIN_USE', ''))}
                    TOTAREA: {c_row.get('TOTAREA', '')}
                    """

    final_prompt = f"""
                You are a real estate and building registry expert in South Korea.

                Determine which of the following candidate building registries best matches the given energy report building. 
                If there is no confident match, respond with "no_match".

                Format your response as JSON:
                {{
                "best_match": "MGM_BLD_PK or no_match",
                "reason": "Why matched or no match"
                }}

                Note: All candidate buildings share the same RECAP_PK as the energy report building ({e_row.get('RECAP_PK')}). Therefore, match only within this group.

                {energy_info}

                {cand_text}
                """
    return final_prompt

def gpt_based_match(e_row, candidate_df, openai_api_key):
    import openai
    openai.api_key = openai_api_key

    if candidate_df.empty:
        return None, "No candidate"

    user_content = build_user_prompt(e_row, candidate_df)

    system_prompt = """
            You are an expert in Korean real estate and building registry.
            You must match the 'energy report building' to the correct 'individual building registry (표제부)'.
            If not sure, respond with "no_match".
            """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # gpt-4o-mini 모델
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=600
        )
        content = response['choices'][0]['message']['content'].strip()
        result = json.loads(content)
        best = result.get("best_match","no_match")
        reason = result.get("reason","")
        if best == "no_match":
            return None, reason
        else:
            return best, reason
    except Exception as e:
        return None, f"GPT Error: {e}"

# ------------------------------------------------
# 4) 최종 매칭 함수
# ------------------------------------------------
def match_buildings(ebd_df, bd_df, openai_api_key):
    """
    ebd_df: EBD(에너지 보고) 데이터
    bd_df: BD(건축물대장) 데이터
    openai_api_key: GPT 호출용 API키

    반환: ebd_df에 [MATCHED_PK, MATCH_STAGE, RULE_DETAILS, GPT_REASON] 열 추가
    """
    # 결과 컬럼 init
    ebd_df['MATCHED_PK'] = None
    ebd_df['MATCH_STAGE'] = 0
    ebd_df['RULE_DETAILS'] = ""
    ebd_df['GPT_REASON'] = ""

    for idx, row in ebd_df.iterrows():
        recap = row['RECAP_PK']
        multi_yn = row['MULTI_YN']
        # BD 후보 필터링
        subset = bd_df[bd_df['RECAP_PK'] == recap]

        # (1) MULTI_YN='N'인 경우 후보가 1건이면 바로 매칭
        if multi_yn == 'N':
            if len(subset) == 1:
                ebd_df.at[idx,'MATCHED_PK'] = subset.iloc[0]['MGM_BLD_PK']
                ebd_df.at[idx,'MATCH_STAGE'] = 1
                continue

        # (2) MULTI_YN='Y' or candidate > 1 : 규칙 기반
        mgmt_pk_1st, detail_str = rule_based_match_multi(row, subset)
        if mgmt_pk_1st is not None:
            # 1차 매칭 성공
            ebd_df.at[idx,'MATCHED_PK'] = mgmt_pk_1st
            ebd_df.at[idx,'MATCH_STAGE'] = 1
            ebd_df.at[idx,'RULE_DETAILS'] = detail_str
        else:
            # 1차 매칭 실패 → 2차 GPT
            best_gpt, reason_gpt = gpt_based_match(row, subset, openai_api_key)
            if best_gpt is not None:
                ebd_df.at[idx,'MATCHED_PK'] = best_gpt
                ebd_df.at[idx,'MATCH_STAGE'] = 2
                ebd_df.at[idx,'GPT_REASON'] = reason_gpt
            else:
                ebd_df.at[idx,'MATCHED_PK'] = None
                ebd_df.at[idx,'MATCH_STAGE'] = 0
                ebd_df.at[idx,'GPT_REASON'] = reason_gpt

    return ebd_df

# ---------------------------
# 예시 사용
# ---------------------------
def main():
    # 1) EBD 데이터 (15개 샘플) 읽기
    ebd_df = pd.read_csv("./data/EBD_TABLE.csv")
    ebd_df = ebd_df.head(15)  
    # 예) 컬럼: SEQ_NO, RECAP_PK, MULTI_YN, OFFICE_NM, BLD_NM, DONG_NM, PUR_NM, AREA 등
    
    # 2) BD 데이터 (전체) 읽기
    bd_df = pd.read_csv("./data/BD_REGIST_no_remove3000.csv")
    # 예) 컬럼: MGM_BLD_PK, RECAP_PK, BLD_NM, DONG_NM, ETC_PURPS(또는 MAIN_USE), TOTAREA 등

    # 3) 매칭 수행 (1차 규칙 기반 → 2차 GPT)
    result_df = match_buildings(ebd_df, bd_df, openai_api_key)

    # 4) 결과 저장
    result_df.to_csv("./result/matching_result.csv", index=False)
    print("Done. See matching_result.csv")

if __name__ == "__main__":
    main()
