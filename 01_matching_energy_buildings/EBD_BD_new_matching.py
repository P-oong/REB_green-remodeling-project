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
    pur_nm = str(e_row.get('용도','')).lower()
    etc_purps = str(c_row.get('ETC_PURPS','')).lower()  # MAIN_USE는 없는 상황
    if pur_nm and pur_nm in etc_purps:
        details["usage_score"] = 1
        reasons["usage"] = "용도 in ETC_PURPS"
    else:
        details["usage_score"] = 0
        reasons["usage"] = "용도 mismatch"

    # (3) 텍스트 결합 점수
    combined_ebd = (str(e_row.get('기관명','')) + " " + str(e_row.get('건축물명',''))).strip().lower()
    combined_bd = (str(c_row.get('BLD_NM','')) + " " + str(c_row.get('DONG_NM',''))).strip().lower()

    if combined_ebd and combined_bd and (combined_ebd in combined_bd):
        details["text_score"] = 1
        reasons["text"] = f"'{combined_ebd}' in '{combined_bd}'"
    else:
        details["text_score"] = 0
        reasons["text"] = "text mismatch"

    # (4) 면적 점수 - 연면적 ±5% 범위 검사
    try:
        ebd_area = e_row.get('연면적', 0)  # 이미 float 형태로 제공됨
        bd_area = c_row.get('TOTAREA', 0)
        
        # 연면적의 ±5% 범위 계산
        lower_bound = ebd_area * 0.95  # 5% 작은 값
        upper_bound = ebd_area * 1.05  # 5% 큰 값
        
        if lower_bound <= bd_area <= upper_bound:
            details["area_score"] = 1
            reasons["area"] = f"{bd_area}는 {ebd_area}의 ±5% 범위({lower_bound:.2f}~{upper_bound:.2f}) 내에 있음"
        else:
            details["area_score"] = 0
            percent_diff = ((bd_area - ebd_area) / ebd_area) * 100
            reasons["area"] = f"{bd_area}는 {ebd_area}의 ±5% 범위 밖에 있음 (차이: {percent_diff:.2f}%)"
    except (ValueError, TypeError):
        details["area_score"] = 0
        reasons["area"] = "면적 데이터 처리 오류"

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
        # 딕셔너리 형태로 반환하여 일관성 유지
        return {
            "matched_pk": None,
            "rule_details": "No candidate",
            "usage_score": 0,
            "text_score": 0,
            "area_score": 0
        }

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
        # 딕셔너리 형태로 반환하여 일관성 유지
        return {
            "matched_pk": br['MGM_BLD_PK'],
            "rule_details": br['rule_details'],
            "usage_score": br['usage_score'],
            "text_score": br['text_score'],
            "area_score": br['area_score']
        }
    else:
        # 딕셔너리 형태로 반환하여 일관성 유지
        return {
            "matched_pk": None,
            "rule_details": "1st rule-based match failed",
            "usage_score": 0,
            "text_score": 0,
            "area_score": 0
        }

# ------------------------------------------------
# 3) 2차 작업: GPT API (Structured Outputs 활용)
# ------------------------------------------------
def build_user_prompt(e_row, candidate_df):
    """
    EBD 건물 + BD 후보 목록 → 최종 GPT user prompt
    """
    combined_ebd = (str(e_row.get('기관명', '')) + " " + str(e_row.get('건축물명', ''))).strip()
    energy_info = f"""
            [Energy Report Building]
            - SEQ_NO: {e_row.get('SEQ_NO')}
            - RECAP_PK: {e_row.get('RECAP_PK')}
            - Combined Name: {combined_ebd}
            - Usage (용도): {e_row.get('용도', '')}
            - Floor Area (연면적): {e_row.get('연면적', '')}
            """

    cand_text = "[Candidate Building Registry]\n"
    for i, c_row in candidate_df.iterrows():
        cmb_bd = (str(c_row.get('BLD_NM', '')) + " " + str(c_row.get('DONG_NM', ''))).strip()
        usage_bd = str(c_row.get('ETC_PURPS',''))  # MAIN_USE 제거
        cand_text += f"""
                    {i+1}. [MGM_BLD_PK: {c_row['MGM_BLD_PK']}]
                    Combined Name: {cmb_bd}
                    Usage: {usage_bd}
                    TOTAREA: {c_row.get('TOTAREA', '')}
                    """

    final_prompt = f"""
                You are a real estate and building registry expert in South Korea.

                Determine which of the following candidate building registries best matches the given energy report building. 
                If there is no confident match, respond with "no_match".

                You MUST select only from the candidate building registries that share the same RECAP_PK. Do not guess beyond this group.

                Format your response as JSON:
                {{
                "best_match": "MGM_BLD_PK or no_match",
                "reason": "Why matched or no match. 매칭 사유(reason)를 **한국어**로 작성해주세요."
                }}

                Note: All candidate buildings share the same RECAP_PK as the energy report building ({e_row.get('RECAP_PK')}). Therefore, match only within this group.

                {energy_info}

                {cand_text}
                """
    return final_prompt

def gpt_based_match(e_row, candidate_df):
    """
    최신 openai 패키지(v1.0.0 이상) 기준 GPT 매칭 수행
    Structured Outputs를 활용하여 응답을 파싱함.
    """
    if candidate_df.empty:
        return {
            "matched_pk": None,
            "reason": "No candidate"
        }

    user_content = build_user_prompt(e_row, candidate_df)

    system_prompt = """
        You are an expert in Korean real estate and building registry.
        Your task is to match the 'energy report building' to the correct 'individual building registry (표제부)'.
        You MUST select only from the candidate building registries that share the same RECAP_PK.
        Do not guess beyond this group. If unsure, respond with "no_match".
        
        Please provide your explanation (reason) in Korean.
    """

    try:
        # Structured Outputs를 사용하여 GPT 응답 파싱 (beta 엔드포인트 사용)
        completion = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",  # 모델 이름 재확인 필요 (Structured Outputs 지원 모델)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=600,
            response_format=GptMatchResponse  # Pydantic 모델을 스키마로 지정
        )

        parsed = completion.choices[0].message.parsed
        print("✅ GPT Parsed Response:", parsed.model_dump_json(indent=2))
        
        if parsed.best_match == "no_match":
            return {
                "matched_pk": None,
                "reason": parsed.reason
            }
        else:
            return {
                "matched_pk": parsed.best_match,
                "reason": parsed.reason
            }

    except Exception as e:
        return {
            "matched_pk": None, 
            "reason": f"GPT Error: {e}"
        }

# ------------------------------------------------
# 4) 최종 매칭 함수 (점수 칼럼 저장 추가)
# ------------------------------------------------
def match_buildings(ebd_df, bd_df):
    """
    중복 매칭을 방지하며 EBD와 BD를 매칭
    """
    # 결과 컬럼 초기화
    ebd_df['MATCHED_PK'] = None
    ebd_df['MATCH_STAGE'] = 0
    ebd_df['RULE_DETAILS'] = ""
    ebd_df['GPT_REASON'] = ""
    ebd_df['USAGE_SCORE'] = 0
    ebd_df['TEXT_SCORE'] = 0
    ebd_df['AREA_SCORE'] = 0
    
    # 이미 매칭된 BD PK를 추적할 세트
    matched_pks = set()
    
    # 각 RECAP_PK에 대한 EBD 개수 계산
    recap_ebd_counts = ebd_df['RECAP_PK'].value_counts().to_dict()
    
    # 1단계: EBD와 BD 모두 1건일 때만 매칭
    for idx, row in ebd_df.iterrows():
        recap = row['RECAP_PK']
        
        if pd.isna(recap):
            ebd_df.at[idx, 'RULE_DETAILS'] = "RECAP_PK is NA, skipped."
            continue
            
        # 해당 RECAP_PK의 EBD가 1건인지 확인
        ebd_count = recap_ebd_counts.get(recap, 0)
        
        # 해당 RECAP_PK의 BD 후보 찾기
        subset = bd_df[bd_df['RECAP_PK'] == recap]
        
        # EBD 1건, BD 1건일 때만 매칭
        if ebd_count == 1 and len(subset) == 1:
            mgm_bld_pk = subset.iloc[0]['MGM_BLD_PK']
            if mgm_bld_pk not in matched_pks:  # 아직 매칭되지 않은 경우만
                ebd_df.at[idx, 'MATCHED_PK'] = mgm_bld_pk
                ebd_df.at[idx, 'MATCH_STAGE'] = 1
                ebd_df.at[idx, 'RULE_DETAILS'] = "Both EBD and BD are single candidates"
                matched_pks.add(mgm_bld_pk)
    
    # 2단계: 매칭되지 않은 EBD에 대해 규칙 기반 점수 계산
    for idx, row in ebd_df.iterrows():
        if ebd_df.at[idx, 'MATCH_STAGE'] > 0:
            continue
            
        recap = row['RECAP_PK']
        if pd.isna(recap):
            continue
            
        subset = bd_df[bd_df['RECAP_PK'] == recap]
        # 이미 매칭된 BD 제외
        subset = subset[~subset['MGM_BLD_PK'].isin(matched_pks)]
        
        if not subset.empty:
            # 각 후보에 대한 점수 계산
            scores = []
            detail_list = []
            usage_list = []
            text_list = []
            area_list = []
            
            for _, c_row in subset.iterrows():
                total_s, rule_str, usage_s, text_s, area_s = compute_rule_score_details(row, c_row)
                scores.append(total_s)
                detail_list.append(rule_str)
                usage_list.append(usage_s)
                text_list.append(text_s)
                area_list.append(area_s)
            
            # 점수 정보를 DataFrame에 추가
            subset_copy = subset.copy()
            subset_copy['total_score'] = scores
            subset_copy['rule_details'] = detail_list
            subset_copy['usage_score'] = usage_list
            subset_copy['text_score'] = text_list
            subset_copy['area_score'] = area_list
            
            # 최대 점수와 최대 점수를 가진 후보들 찾기
            max_score = subset_copy['total_score'].max()
            best_candidates = subset_copy[subset_copy['total_score'] == max_score]
            
            # 최대 점수가 2점 이상이고, 최대 점수를 가진 후보가 유일할 때만 매칭
            if max_score >= 2 and len(best_candidates) == 1:
                best = best_candidates.iloc[0]
                bd_pk = best['MGM_BLD_PK']
                
                if bd_pk not in matched_pks:
                    ebd_df.at[idx, 'MATCHED_PK'] = bd_pk
                    ebd_df.at[idx, 'MATCH_STAGE'] = 2
                    ebd_df.at[idx, 'RULE_DETAILS'] = best['rule_details']
                    ebd_df.at[idx, 'USAGE_SCORE'] = best['usage_score']
                    ebd_df.at[idx, 'TEXT_SCORE'] = best['text_score']
                    ebd_df.at[idx, 'AREA_SCORE'] = best['area_score']
                    matched_pks.add(bd_pk)
    
    # 4단계: 남은 매칭되지 않은 EBD에 대해 GPT 기반 매칭
    for idx, row in ebd_df.iterrows():
        if ebd_df.at[idx, 'MATCH_STAGE'] > 0:  # 이미 매칭된 경우 스킵
            continue
            
        recap = row['RECAP_PK']
        if pd.isna(recap):
            continue
            
        subset = bd_df[bd_df['RECAP_PK'] == recap]
        # 이미 매칭된 BD 제외
        subset = subset[~subset['MGM_BLD_PK'].isin(matched_pks)]
        
        if not subset.empty:
            best_gpt, reason_gpt = gpt_based_match(row, subset)
            if best_gpt is not None and best_gpt not in matched_pks:
                ebd_df.at[idx, 'MATCHED_PK'] = best_gpt
                ebd_df.at[idx, 'MATCH_STAGE'] = 3
                ebd_df.at[idx, 'GPT_REASON'] = reason_gpt
                matched_pks.add(best_gpt)
            else:
                # GPT 매칭 실패
                ebd_df.at[idx, 'MATCH_STAGE'] = 99
                ebd_df.at[idx, 'GPT_REASON'] = reason_gpt if reason_gpt else "남은 BD 후보 없음"
        else:
            # 매칭할 BD 후보가 없음
            ebd_df.at[idx, 'MATCH_STAGE'] = 99
            ebd_df.at[idx, 'GPT_REASON'] = "남은 BD 후보 없음 (모두 매칭됨)"

    return ebd_df

def main():
    # 새 Excel 파일 로드
    ebd_df = pd.read_excel("./data/EBD_new_2.xlsx")
    bd_df = pd.read_excel("./data/BD_REGIST_no_remove3000.xlsx")
    
    # 매칭
    result_df = match_buildings(ebd_df, bd_df)
    
    # 결과 저장 (Excel 형식으로 변경)
    os.makedirs("./result", exist_ok=True)  # result 폴더가 없으면 생성
    result_df.to_excel("./result/matching_result_4o_mini.xlsx", index=False)
    print("Done. See matching_result_4o_mini.xlsx")

if __name__ == "__main__":
    main()
