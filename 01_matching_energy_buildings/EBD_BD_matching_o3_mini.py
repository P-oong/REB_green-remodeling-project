import os
import json
import pandas as pd
from dotenv import load_dotenv
import openai
from pydantic import BaseModel

# 1) 환경 변수 로드 & OpenAI 키 체크
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
openai.api_key = openai_api_key

# ------------------------------------------------
# Pydantic 모델 정의: GPT 응답 스키마
# ------------------------------------------------
class GptMatchResponse(BaseModel):
    best_match: str
    reason: str

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
        val_str = s.replace("이상", "").strip()
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
    항목별 점수(usage_score, text_score, area_score) + 상세사유
    
    반환: (total_score, rule_details, usage_s, text_s, area_s)
    """

    # (1) 미리 키를 0으로 초기화
    details = {
        "usage_score": 0,
        "text_score": 0,
        "area_score": 0
    }
    # 사유도 마찬가지
    reasons = {
        "usage": "N/A",
        "text": "N/A",
        "area": "N/A"
    }

    # (2) 용도 점수
    pur_nm = str(e_row.get('PUR_NM','')).lower()
    etc_purps = str(c_row.get('ETC_PURPS','')).lower()  # MAIN_USE는 없는 상황
    if pur_nm and pur_nm in etc_purps:
        details["usage_score"] = 1
        reasons["usage"] = "PUR_NM in ETC_PURPS"
    else:
        details["usage_score"] = 0
        reasons["usage"] = "PUR_NM mismatch"

    # (3) 텍스트 결합 점수
    combined_ebd = (str(e_row.get('OFFICE_NM','')) + " " + str(e_row.get('BLD_NM',''))).strip().lower()
    combined_bd = (str(c_row.get('BLD_NM','')) + " " + str(c_row.get('DONG_NM',''))).strip().lower()

    if combined_ebd and combined_bd and (combined_ebd in combined_bd):
        details["text_score"] = 1
        reasons["text"] = f"'{combined_ebd}' in '{combined_bd}'"
    else:
        details["text_score"] = 0
        reasons["text"] = "text mismatch"

    # (4) 면적 점수
    area_str = e_row.get('AREA','')
    totarea = float(c_row.get('TOTAREA',0))
    lower, upper = parse_area_text(area_str)
    if (lower is None) and (upper is None):
        details["area_score"] = 0
        reasons["area"] = "Area range parse failed"
    else:
        ok_lower = (lower is None) or (totarea >= lower)
        ok_upper = (upper is None) or (totarea < upper)
        if ok_lower and ok_upper:
            details["area_score"] = 1
            reasons["area"] = f"{totarea} in [{lower if lower else 0}, {upper if upper else '∞'})"
        else:
            details["area_score"] = 0
            reasons["area"] = f"{totarea} not in range"

    # (5) 총점 및 상세 문자열
    total_score = sum(details.values())

    rule_details = "; ".join([
        f"{k}:{details[k]}({reasons[k.split('_')[0]]})"
        for k in ["usage_score", "text_score", "area_score"]
    ])

    return total_score, rule_details, details["usage_score"], details["text_score"], details["area_score"]

def rule_based_match_multi(e_row, candidate_df):
    """
    MULTI_YN='Y' 인 건물의 1차 규칙 기반 매칭
    - 최고 점수가 2점 이상이고 유일하면 매칭
    반환값: (mgmt_pk or None, rule_details, usage_s, text_s, area_s)
    """
    if candidate_df.empty:
        return None, "No candidate", 0, 0, 0

    scores = []
    detail_list = []
    usage_list = []
    text_list = []
    area_list = []

    for _, c_row in candidate_df.iterrows():
        total_s, rule_str, usage_s, text_s, area_s = compute_rule_score_details(e_row, c_row)
        scores.append(total_s)
        detail_list.append(rule_str)
        usage_list.append(usage_s)
        text_list.append(text_s)
        area_list.append(area_s)

    df_copy = candidate_df.copy()
    df_copy['total_score'] = scores
    df_copy['rule_details'] = detail_list
    df_copy['usage_score'] = usage_list
    df_copy['text_score'] = text_list
    df_copy['area_score'] = area_list

    max_score = df_copy['total_score'].max()
    best = df_copy[df_copy['total_score'] == max_score]

    if max_score >= 2 and len(best) == 1:
        br = best.iloc[0]
        return (
            br['MGM_BLD_PK'], 
            br['rule_details'],
            br['usage_score'],
            br['text_score'],
            br['area_score']
        )
    else:
        return None, "1st rule-based match failed", 0, 0, 0

# ------------------------------------------------
# 3) 2차 작업: GPT API (Structured Outputs 활용)
# ------------------------------------------------
def build_user_prompt(e_row, candidate_df):
    """
    간소화된 프롬프트: EBD 건물과 해당 RECAP_PK를 가진 BD 후보들을 간략히 정리
    """
    combined_ebd = f"{e_row.get('OFFICE_NM', '')} {e_row.get('BLD_NM', '')}".strip()
    energy_info = (
        f"Energy Report Building:\n"
        f"SEQ_NO: {e_row.get('SEQ_NO')}\n"
        f"RECAP_PK: {e_row.get('RECAP_PK')}\n"
        f"Name: {combined_ebd}\n"
        f"Usage: {e_row.get('PUR_NM', '')}\n"
        f"Area: {e_row.get('AREA', '')}\n"
    )
    cand_text = "Candidate Buildings:\n"
    for i, c_row in candidate_df.iterrows():
        cmb_bd = f"{c_row.get('BLD_NM', '')} {c_row.get('DONG_NM', '')}".strip()
        usage_bd = c_row.get('ETC_PURPS', '')
        cand_text += (
            f"{i+1}. [MGM_BLD_PK: {c_row['MGM_BLD_PK']}]\n"
            f"Name: {cmb_bd}\n"
            f"Usage: {usage_bd}\n"
            f"TOTAREA: {c_row.get('TOTAREA', '')}\n"
        )
    prompt = (
        "You are an expert in Korean real estate. "
        "Match the energy report building to the best candidate with the same RECAP_PK. "
        "If no confident match, respond with 'no_match'. "
        "Provide your explanation in Korean. "
        "Return your answer in JSON format: {\"best_match\": \"MGM_BLD_PK or no_match\", \"reason\": \"...\"}\n\n"
        + energy_info + "\n" + cand_text
    )
    return prompt

def gpt_based_match(e_row, candidate_df):
    """
    GPT 매칭 수행: 간소화된 프롬프트 사용, o3-mini에 맞춰 temperature 없이 호출
    """
    if candidate_df.empty:
        return None, "No candidate"

    user_content = build_user_prompt(e_row, candidate_df)

    system_prompt = (
        "You are an expert in Korean real estate. "
        "Match the energy report building to the correct candidate (same RECAP_PK) and explain in Korean."
    )

    try:
        completion = openai.beta.chat.completions.parse(
            model="o3-mini",  # o3-mini 모델 사용
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_completion_tokens=50,
            response_format=GptMatchResponse  # Pydantic 모델 사용
        )

        parsed = completion.choices[0].message.parsed
        print("✅ GPT Parsed Response:", parsed.model_dump_json(indent=2))
        
        if parsed.best_match == "no_match":
            return None, parsed.reason
        else:
            return parsed.best_match, parsed.reason

    except Exception as e:
        return None, f"GPT Error: {e}"

# ------------------------------------------------
# 4) 최종 매칭 함수 (점수 칼럼 저장 추가)
# ------------------------------------------------
def match_buildings(ebd_df, bd_df):
    """
    ebd_df: EBD(에너지 보고) 데이터
    bd_df: BD(건축물대장) 데이터 (MAIN_USE 없음, ETC_PURPS만 있음)

    반환: ebd_df에 [MATCHED_PK, MATCH_STAGE, RULE_DETAILS, GPT_REASON,
                    USAGE_SCORE, TEXT_SCORE, AREA_SCORE] 열 추가
    """
    # 결과 컬럼 초기화
    ebd_df['MATCHED_PK'] = None
    ebd_df['MATCH_STAGE'] = 0
    ebd_df['RULE_DETAILS'] = ""
    ebd_df['GPT_REASON'] = ""
    ebd_df['USAGE_SCORE'] = 0
    ebd_df['TEXT_SCORE'] = 0
    ebd_df['AREA_SCORE'] = 0

    for idx, row in ebd_df.iterrows():
        recap = row['RECAP_PK']
        multi_yn = row['MULTI_YN']

        # RECAP_PK 또는 MULTI_YN NA → 스킵
        if pd.isna(recap) or pd.isna(multi_yn):
            ebd_df.at[idx, 'GPT_REASON'] = "RECAP_PK or MULTI_YN is NA, skipped."
            continue

        # BD 후보
        subset = bd_df[bd_df['RECAP_PK'] == recap]

        # (1) MULTI_YN='N' & 후보가 1건이면 바로 매칭
        if multi_yn == 'N' and len(subset) == 1:
            ebd_df.at[idx, 'MATCHED_PK'] = subset.iloc[0]['MGM_BLD_PK']
            ebd_df.at[idx, 'MATCH_STAGE'] = 1
            continue

        # (2) 1차 규칙
        mgmt_pk_1st, detail_str, usage_s, text_s, area_s = rule_based_match_multi(row, subset)
        if mgmt_pk_1st is not None:
            ebd_df.at[idx, 'MATCHED_PK'] = mgmt_pk_1st
            ebd_df.at[idx, 'MATCH_STAGE'] = 2
            ebd_df.at[idx, 'RULE_DETAILS'] = detail_str
            ebd_df.at[idx, 'USAGE_SCORE'] = usage_s
            ebd_df.at[idx, 'TEXT_SCORE'] = text_s
            ebd_df.at[idx, 'AREA_SCORE'] = area_s
        else:
            # 1차 실패 → GPT
            best_gpt, reason_gpt = gpt_based_match(row, subset)
            if best_gpt is not None:
                ebd_df.at[idx, 'MATCHED_PK'] = best_gpt
                ebd_df.at[idx, 'MATCH_STAGE'] = 3
                ebd_df.at[idx, 'GPT_REASON'] = reason_gpt
                ebd_df.at[idx, 'USAGE_SCORE'] = 0
                ebd_df.at[idx, 'TEXT_SCORE'] = 0
                ebd_df.at[idx, 'AREA_SCORE'] = 0
            else:
                # 최종 실패
                ebd_df.at[idx, 'MATCHED_PK'] = None
                ebd_df.at[idx, 'MATCH_STAGE'] = 99
                ebd_df.at[idx, 'GPT_REASON'] = reason_gpt
                ebd_df.at[idx, 'USAGE_SCORE'] = 0
                ebd_df.at[idx, 'TEXT_SCORE'] = 0
                ebd_df.at[idx, 'AREA_SCORE'] = 0

    return ebd_df

def main():
    # EBD 15개 샘플
    ebd_df = pd.read_csv("./data/EBD_TABLE.csv", encoding='cp949').head(15)

    # BD 전체 (MAIN_USE 없음, ETC_PURPS만 있음)
    bd_df = pd.read_csv("./data/BD_REGIST_no_remove3000.csv", encoding='cp949')

    # 매칭
    result_df = match_buildings(ebd_df, bd_df)

    # 결과 저장
    result_df.to_csv("./result/matching_result_o3-mini.csv", index=False, quoting=1)
    print("Done. See matching_result_o3.csv")

if __name__ == "__main__":
    main()
