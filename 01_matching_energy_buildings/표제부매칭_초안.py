import os
import json
import pandas as pd
import openai
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
openai.api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API 키
# langsmith_api_key = os.getenv("LANGSMITH_API_KEY")  # 필요시 사용

# ------------------------------------------------------------------------
# 1) 텍스트 결합 함수
# ------------------------------------------------------------------------
def parse_area_text(area_str):
    """
    EBD의 AREA 컬럼(예: "3,000㎡이상~5,000㎡미만")을 파싱하여 (lower_bound, upper_bound)로 반환.
    upper_bound가 None이면 상한이 없다는 의미(이상만).
    """
    if not isinstance(area_str, str):
        return (None, None)

    # 쉼표, "㎡", 공백 등 제거
    s = area_str.replace(",", "").replace("㎡", "").strip()
    # 예) "3000이상~5000미만", "10000이상"

    lower, upper = None, None

    # 1) "이상~" 있는 경우
    if "이상~" in s:
        parts = s.split("이상~")
        # parts[0] = "3000"
        # parts[1] = "5000미만"
        try:
            lower = float(parts[0])
        except:
            lower = None

        # 두 번째 부분에 "미만" 있으면 upper 설정
        if "미만" in parts[1]:
            up_str = parts[1].replace("미만", "").strip()
            try:
                upper = float(up_str)
            except:
                upper = None
        else:
            # "미만"이 안 붙어있으면 upper = None (하한만 존재)
            upper = None

    # 2) "이상"만 있는 경우
    elif "이상" in s:
        val_str = s.replace("이상","").strip()
        try:
            lower = float(val_str)
        except:
            lower = None
        upper = None

    # 필요한 경우 "미만"만 있는 케이스도 추가 처리 가능

    return (lower, upper)


# ------------------------------------------------------------------------
# 2) 1차 매칭 - MULTI_YN='N' 인 건물 (단일 표제부)
# ------------------------------------------------------------------------
def compute_rule_score_details(e_row, c_row):
    """
    e_row: EBD 한 행
    c_row: BD 후보 한 행
    변수별 점수를 각각 기록하고, 그 이유를 dict에 저장
    """
    details = {}
    reasons = {}

    # 1) 용도 비교
    pur_nm = str(e_row.get('PUR_NM','')).lower()
    etc_purps = str(c_row.get('ETC_PURPS', c_row.get('MAIN_USE',''))).lower()
    if pur_nm and pur_nm in etc_purps:
        details["usage_score"] = 1
        reasons["usage"] = "PUR_NM in ETC_PURPS"
    else:
        details["usage_score"] = 0
        reasons["usage"] = "PUR_NM 불일치"

    # 2) 텍스트 결합 (OFFICE_NM+BLD_NM) vs (BLD_NM+DONG_NM)
    ebd_text = f"{e_row.get('OFFICE_NM','')} {e_row.get('BLD_NM','')}".strip().lower()
    bd_text = f"{c_row.get('BLD_NM','')} {c_row.get('DONG_NM','')}".strip().lower()
    if ebd_text and bd_text and (ebd_text in bd_text):
        details["text_score"] = 1
        reasons["text"] = f"EBD text in BD text ({ebd_text} in {bd_text})"
    else:
        details["text_score"] = 0
        reasons["text"] = "텍스트 불일치"

    # 3) 면적 범위
    area_str = e_row.get('AREA','')  # "3,000㎡이상~5,000㎡미만" 등
    totarea = float(c_row.get('TOTAREA',0))
    lower, upper = parse_area_text(area_str)
    if lower is None and upper is None:
        # 파싱 실패
        details["area_score"] = 0
        reasons["area"] = "면적 범위 파싱 실패"
    else:
        # 범위 확인
        ok_lower = (lower is None) or (totarea >= lower)
        ok_upper = (upper is None) or (totarea < upper)
        if ok_lower and ok_upper:
            details["area_score"] = 1
            reasons["area"] = f"{totarea} in [{lower if lower else 0}, {upper if upper else '∞'})"
        else:
            details["area_score"] = 0
            reasons["area"] = f"{totarea} not in range"

    total_score = sum(details.values())
    # rule_details 문자열로 정리
    rule_details = "; ".join([f"{k}:{details[k]}({reasons[k]})" for k in details])

    return total_score, rule_details


# ------------------------------------------------------------------------
# 3) 1차 매칭 - MULTI_YN='Y' 인 건물 (여러 표제부)
#   (규칙 기반 매칭: 텍스트 유사도, etc)
# ------------------------------------------------------------------------
def rule_based_match_multi(e_row, candidate_df):
    """
    MULTI_YN='Y'인 경우, 각 후보별 점수와 사유 계산.
    최고 점수 >= 2이고, 유일 후보면 매칭. 아니면 실패.
    """
    if candidate_df.empty:
        return None, "후보 없음"

    scores = []
    detail_list = []
    for idx, c_row in candidate_df.iterrows():
        s, d_str = compute_rule_score_details(e_row, c_row)
        scores.append(s)
        detail_list.append(d_str)

    df_copy = candidate_df.copy()
    df_copy['total_score'] = scores
    df_copy['rule_details'] = detail_list

    max_score = df_copy['total_score'].max()
    best = df_copy[df_copy['total_score'] == max_score]

    if max_score >= 2 and len(best) == 1:
        best_row = best.iloc[0]
        return best_row['MGM_BLD_PK'], best_row['rule_details']
    else:
        return None, "규칙 매칭 실패"


# ------------------------------------------------------------------------
# 4) 2차 작업 - GPT 매칭
# ------------------------------------------------------------------------
import openai

def gpt_based_match(e_row, candidate_df):
    if candidate_df.empty:
        return None, "후보 없음"

    text_candidates = ""
    for i, r in candidate_df.iterrows():
        text_candidates += (
            f"{i+1}. [MGM_BLD_PK: {r['MGM_BLD_PK']}]\n"
            f"   건물명: {r.get('BLD_NM','')}\n"
            f"   동이름: {r.get('DONG_NM','')}\n"
            f"   용도: {r.get('ETC_PURPS','')}\n"
            f"   연면적: {r.get('TOTAREA','')}㎡\n\n"
        )

    prompt = f"""
            에너지 보고 건물 정보:
            - SEQ_NO: {e_row.get('SEQ_NO')}
            - MULTI_YN: {e_row.get('MULTI_YN')}
            - PUR_NM: {e_row.get('PUR_NM')}
            - OFFICE_NM+BLD_NM: {e_row.get('OFFICE_NM','')} {e_row.get('BLD_NM','')}
            - AREA: {e_row.get('AREA')}
            - DONG_NM: {e_row.get('DONG_NM','')}

            건축물대장 후보 목록:
            {text_candidates}

            위 정보를 바탕으로 가장 적합한 후보의 MGM_BLD_PK를 결정해 주세요.
            없다면 "no_match".

            응답은 JSON 형식:
            {{"best_match":"MGM_BLD_PK or no_match","reason":"판단 근거"}}
            """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 건물 매칭 전문가입니다."},
                {"role": "user", "content": prompt}
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
        return None, f"GPT에러: {e}"


# ------------------------------------------------------------------------
# 5) 메인 매칭 함수
# ------------------------------------------------------------------------
def match_buildings(ebd_df, bd_df):
    """
    ebd_df, bd_df를 받아 2단계 매칭(1차 규칙, 2차 GPT)을 수행
    """
    # 우선 텍스트 결합
    ebd_df['combined_ebd_name'] = ebd_df.apply(create_ebd_combined_text, axis=1)
    bd_df['combined_bd_name'] = bd_df.apply(create_bd_combined_text, axis=1)

    # 결과 컬럼 초기화
    ebd_df['MATCHED_PK'] = None
    ebd_df['MATCH_STAGE'] = 0
    ebd_df['GPT_REASON'] = ""

    for idx, row in ebd_df.iterrows():
        recap = row['RECAP_PK']
        multi_yn = row['MULTI_YN']

        # 1) MULTI_YN='N'인 경우: RECAP_PK로 BD 필터링 → 정확히 1건이면 매칭
        if multi_yn == 'N':
            single_match = match_single_pk(row, bd_df)
            if single_match is not None:
                # 1차 매칭 성공
                ebd_df.at[idx, 'MATCHED_PK'] = single_match
                ebd_df.at[idx, 'MATCH_STAGE'] = 1
                continue  # 다음 행으로 넘어감

        # MULTI_YN='Y'이거나, single_match 실패 시
        # RECAP_PK에 해당하는 후보 목록
        subset = bd_df[bd_df['RECAP_PK'] == recap]
        
        # 2) 1차 작업(규칙 기반) 시도
        mgmt_pk_1st = rule_based_match_multi(row, subset) if multi_yn == 'Y' else None
        if mgmt_pk_1st is not None:
            ebd_df.at[idx, 'MATCHED_PK'] = mgmt_pk_1st
            ebd_df.at[idx, 'MATCH_STAGE'] = 1
        else:
            # 1차 실패 → 2차 GPT
            best_gpt, reason = gpt_based_match(row, subset)
            if best_gpt is not None:
                ebd_df.at[idx, 'MATCHED_PK'] = best_gpt
                ebd_df.at[idx, 'MATCH_STAGE'] = 2
                ebd_df.at[idx, 'GPT_REASON'] = reason
            else:
                # 최종 매칭 실패
                ebd_df.at[idx, 'MATCHED_PK'] = None
                ebd_df.at[idx, 'MATCH_STAGE'] = 0
                ebd_df.at[idx, 'GPT_REASON'] = reason

    return ebd_df

# ---------------------------
# 예시 사용
# ---------------------------
def main():
    # 1) EBD 데이터 (15개 샘플) 읽기
    ebd_df = pd.read_csv("EBD_15samples.csv")  
    # 예) 컬럼: SEQ_NO, RECAP_PK, MULTI_YN, OFFICE_NM, BLD_NM, DONG_NM, PUR_NM, AREA 등
    
    # 2) BD 데이터 (전체) 읽기
    bd_df = pd.read_csv("BD_REGIST_all.csv")
    # 예) 컬럼: MGM_BLD_PK, RECAP_PK, BLD_NM, DONG_NM, ETC_PURPS(또는 MAIN_USE), TOTAREA 등

    # 3) 매칭 수행 (1차 규칙 기반 → 2차 GPT)
    final_df = match_buildings(ebd_df, bd_df)

    # 4) 결과 확인 (csv로 저장하거나 print)
    final_df.to_csv("matching_result.csv", index=False)
    print("매칭 결과가 matching_result.csv에 저장되었습니다.")

if __name__ == "__main__":
    main()
