import os
import pandas as pd
import numpy as np

def load_data():
    """
    기존 매칭 결과와 BD 데이터를 로드
    """
    # 기존 매칭 결과 로드
    rule_result_df = pd.read_excel("./result/rule_based_matching_result_ver2.xlsx")
    
    # BD 데이터 로드
    bd_df = pd.read_excel("./data/BD_data_all.xlsx")
    
    return rule_result_df, bd_df

def extract_unmatched_ebd(rule_result_df):
    """
    MATCH_STAGE가 '미매칭'인 EBD 데이터만 추출
    """
    # EBD 관련 컬럼만 추출 (영문 컬럼과 기본 정보 제외)
    unmatched_df = rule_result_df[rule_result_df['MATCH_STAGE'] == '미매칭'].copy()
    
    # 필요한 컬럼만 선택 (한글 변수명 + 기본 식별 정보)
    ebd_columns = ['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '기관명', '건축물명', '주소', '지상', '지하', 'MGM_BLD_PK', 'MATCH_STAGE', 'EBD_COUNT', 'BD_COUNT', 'EBD_OVER_BD']
    ebd_columns = [col for col in ebd_columns if col in unmatched_df.columns]
    unmatched_ebd = unmatched_df[ebd_columns]
    
    return unmatched_ebd

def calculate_area_score(ebd_area, bd_area):
    """
    연면적 점수 계산: ±5% 범위 내에 있으면 1점, 아니면 0점 
    """
    if pd.isna(ebd_area) or pd.isna(bd_area):
        return 0
    
    try:
        ebd_area = float(ebd_area)
        bd_area = float(bd_area)
        
        # 5% 범위 계산
        lower_bound = ebd_area * 0.95
        upper_bound = ebd_area * 1.05
        
        # 점수 할당
        if lower_bound <= bd_area <= upper_bound:
            return 1
        else:
            return 0
    except (ValueError, TypeError):
        return 0

def calculate_date_score(ebd_date, bd_date):
    """
    승인일자 점수 계산: 일치하면 1점, 아니면 0점
    """
    if pd.isna(ebd_date) or pd.isna(bd_date):
        return 0
    
    try:
        # 형식 통일 (문자열로 변환하여 비교)
        ebd_date_str = str(int(ebd_date)) if not pd.isna(ebd_date) else ""
        bd_date_str = str(int(bd_date)) if not pd.isna(bd_date) else ""
        
        # 점수 할당
        if ebd_date_str == bd_date_str:
            return 1
        else:
            return 0
    except (ValueError, TypeError):
        return 0

def calculate_text_score(ebd_row, bd_row):
    """
    텍스트 비교 점수 계산: 
    - EBD(기관명, 건축물명, 주소)와 BD(BLD_NM, DONG_NM)를 교차 비교
    - 각 유사성에 따라 점수 부여 (초안)
    """
    text_score = 0
    
    # 텍스트 변수 추출
    ebd_texts = {
        '기관명': str(ebd_row.get('기관명', '')).lower(),
        '건축물명': str(ebd_row.get('건축물명', '')).lower(),
        '주소': str(ebd_row.get('주소', '')).lower()
    }
    
    bd_texts = {
        'BLD_NM': str(bd_row.get('BLD_NM', '')).lower(),
        'DONG_NM': str(bd_row.get('DONG_NM', '')).lower()
    }
    
    # 1. 기관명이 BLD_NM에 포함되는지 확인
    if ebd_texts['기관명'] and ebd_texts['기관명'] in bd_texts['BLD_NM']:
        text_score += 1
    
    # 2. 건축물명이 BLD_NM 또는 DONG_NM에 포함되는지 확인
    if ebd_texts['건축물명'] and (ebd_texts['건축물명'] in bd_texts['BLD_NM'] or ebd_texts['건축물명'] in bd_texts['DONG_NM']):
        text_score += 1
    
    # 3. 결합 텍스트 비교
    ebd_combined = (ebd_texts['기관명'] + " " + ebd_texts['건축물명']).strip()
    bd_combined = (bd_texts['BLD_NM'] + " " + bd_texts['DONG_NM']).strip()
    
    if ebd_combined and ebd_combined in bd_combined:
        text_score += 1
    
    return text_score

def score_based_matching(unmatched_ebd, bd_df):
    """
    점수 기반 매칭 수행:
    1. 동일 RECAP의 EBD-BD 조합별 점수 계산
    2. 점수가 2점 이상이고 최고 점수가 유일한 경우 매칭
    """
    # 결과 저장을 위한 리스트
    match_results = []
    
    # 동일 RECAP별로 그룹화하여 처리
    recap_groups = unmatched_ebd.groupby('RECAP_PK')
    
    for recap, ebd_group in recap_groups:
        if pd.isna(recap):
            # RECAP이 NA인 경우 매칭 시도하지 않음
            for _, ebd_row in ebd_group.iterrows():
                result = ebd_row.to_dict()
                result['MATCH_STAGE'] = '미매칭(RECAP_NA)'
                match_results.append(result)
            continue
        
        # 동일 RECAP의 BD 후보들
        bd_candidates = bd_df[bd_df['RECAP_PK'] == recap].copy()
        
        if bd_candidates.empty:
            # BD 후보가 없는 경우 매칭 시도하지 않음
            for _, ebd_row in ebd_group.iterrows():
                result = ebd_row.to_dict()
                result['MATCH_STAGE'] = '미매칭(후보없음)'
                match_results.append(result)
            continue
        
        # EBD-BD 조합별 점수 계산 및 저장
        all_scores = []
        
        for _, ebd_row in ebd_group.iterrows():
            ebd_id = ebd_row['SEQ_NO']
            ebd_scores = []
            
            for _, bd_row in bd_candidates.iterrows():
                bd_id = bd_row['MGM_BLD_PK']
                
                # 점수 계산
                area_score = calculate_area_score(ebd_row.get('연면적'), bd_row.get('TOTAREA'))
                date_score = calculate_date_score(ebd_row.get('사용승인연도'), bd_row.get('USE_DATE'))
                text_score = calculate_text_score(ebd_row, bd_row)
                
                # 총점 계산
                total_score = area_score + date_score + text_score
                
                # 점수 정보 저장
                score_info = {
                    'ebd_id': ebd_id,
                    'bd_id': bd_id,
                    'area_score': area_score,
                    'date_score': date_score,
                    'text_score': text_score,
                    'total_score': total_score,
                    'ebd_row': ebd_row,
                    'bd_row': bd_row
                }
                
                ebd_scores.append(score_info)
                all_scores.append(score_info)
            
            # 해당 EBD에 대해 점수 기준으로 매칭 결정
            if ebd_scores:
                # 최고 점수 및 해당 후보들 찾기
                max_score = max(score['total_score'] for score in ebd_scores)
                best_candidates = [score for score in ebd_scores if score['total_score'] == max_score]
                
                # 매칭 여부 결정
                if max_score >= 2 and len(best_candidates) == 1:
                    # 점수가 2점 이상이고 최고 점수가 유일할 때 매칭
                    best = best_candidates[0]
                    
                    # 매칭 결과 생성
                    result = ebd_row.to_dict()
                    
                    # BD 정보 추가
                    for key, value in best['bd_row'].items():
                        if key not in result:  # 중복 컬럼 처리
                            result[key] = value
                    
                    # 점수 정보 추가
                    result['AREA_SCORE'] = best['area_score']
                    result['DATE_SCORE'] = best['date_score']
                    result['TEXT_SCORE'] = best['text_score']
                    result['TOTAL_SCORE'] = best['total_score']
                    result['MATCH_STAGE'] = '4차'
                    
                    match_results.append(result)
                else:
                    # 매칭되지 않은 경우
                    result = ebd_row.to_dict()
                    
                    if max_score < 2:
                        result['MATCH_STAGE'] = '미매칭(점수미달)'
                    else:
                        result['MATCH_STAGE'] = '미매칭(중복후보)'
                    
                    match_results.append(result)
        
    # 리스트를 데이터프레임으로 변환
    result_df = pd.DataFrame(match_results)
    
    return result_df

def main():
    # 데이터 로드
    rule_result_df, bd_df = load_data()
    
    # 원본 순서 보존을 위한 컬럼 추가
    rule_result_df['_원본순서'] = range(len(rule_result_df))
    
    # 미매칭 EBD 추출
    unmatched_ebd = extract_unmatched_ebd(rule_result_df)
    print(f"미매칭 EBD 건수: {len(unmatched_ebd)}")
    
    # 점수 기반 매칭 수행
    score_result_df = score_based_matching(unmatched_ebd, bd_df)
    
    # 통계 출력
    stage_counts = score_result_df['MATCH_STAGE'].value_counts()
    print("\n4차 점수 기반 매칭 결과:")
    for stage, count in stage_counts.items():
        print(f"- {stage}: {count}건")
    
    # 기존 매칭 결과에서 미매칭 레코드만 제외
    matched_from_rules = rule_result_df[rule_result_df['MATCH_STAGE'] != '미매칭'].copy()
    
    # 기존 매칭 결과와 새로운 점수 기반 매칭 결과 합치기
    final_result = pd.concat([matched_from_rules, score_result_df], ignore_index=False)
    
    # 원본 순서대로 정렬
    final_result = final_result.sort_values('_원본순서')
    
        # 통계 출력
    final_counts = final_result['MATCH_STAGE'].value_counts()
    print("\n4차 점수 기반 매칭 결과:")
    for stage, count in final_counts.items():
        print(f"- {stage}: {count}건")
    
    # 임시 순서 컬럼 제거
    if '_원본순서' in final_result.columns:
        final_result = final_result.drop('_원본순서', axis=1)
    
    # 결과 저장
    os.makedirs("./result", exist_ok=True)
    final_result.to_excel("./result/score_matching_result_ver1.xlsx", index=False)
    print("\n최종 결과가 './result/score_matching_result_ver1.xlsx'에 저장되었습니다.")
    
    # 점수 기반 매칭 결과만 따로 저장
    score_result_df.to_excel("./result/score_only_matching_result_ver1.xlsx", index=False)
    print("4차 점수 기반 매칭 결과가 './result/score_only_matching_result_ver1.xlsx'에 저장되었습니다.")
    
    return final_result

if __name__ == "__main__":
    main()
