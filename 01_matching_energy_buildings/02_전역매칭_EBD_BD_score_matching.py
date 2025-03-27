import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # 진행률 표시를 위해 추가
import time  # 실행 시간 측정을 위해 추가

def load_data():
    """
    기존 매칭 결과와 BD 데이터를 로드
    """
    # 기존 매칭 결과 로드
    rule_result_df = pd.read_excel("./result/rule_matching_result_ver3.xlsx")
    
    # BD 데이터 로드
    bd_df = pd.read_excel("./data/BD_data_all.xlsx")
    
    return rule_result_df, bd_df

def preprocess_data(df, columns, is_bd=False):
    """
    데이터 전처리: 숫자 컬럼 변환, 문자열 컬럼 소문자화
    """
    result = df.copy()
    
    # 숫자 컬럼 처리
    numeric_columns = ['연면적', 'TOTAREA', '사용승인연도', 'USE_DATE'] if is_bd else ['연면적', '사용승인연도']
    for col in numeric_columns:
        if col in result.columns:
            # NaN이 아닌 값만 변환 (에러 방지)
            mask = ~pd.isna(result[col])
            try:
                result.loc[mask, col] = pd.to_numeric(result.loc[mask, col], errors='coerce')
            except:
                pass
    
    # 문자열 컬럼 처리
    text_columns = ['BLD_NM', 'DONG_NM'] if is_bd else ['기관명', '건축물명', '주소']
    for col in text_columns:
        if col in result.columns:
            result[col] = result[col].astype(str).str.lower()
    
    return result[columns]

def calculate_area_score(ebd_area, bd_area): 
    """
    연면적 점수 계산: 다양한 범위에 따라 차등적 점수 부여
    - ±1% 범위: 1.0점 (매우 높은 정밀도)
    - ±5% 범위: 0.8점 (높은 일치도)
    - ±10% 범위: 0.5점 (중간 정도 일치)
    - ±20% 범위: 0.2점 (낮은 일치도)
    - 범위 외: 0점
    """
    if pd.isna(ebd_area) or pd.isna(bd_area):
        return 0.0
    
    try:
        ebd_area = float(ebd_area)
        bd_area = float(bd_area)
        
        # 면적 차이 비율 계산 (절대값)
        diff_ratio = abs(ebd_area - bd_area) / ebd_area
        
        # 범위에 따른 점수 부여
        if diff_ratio <= 0.01:  # ±1% 범위
            return 1.0
        elif diff_ratio <= 0.05:  # ±5% 범위
            return 0.8
        elif diff_ratio <= 0.10:  # ±10% 범위
            return 0.5
        elif diff_ratio <= 0.20:  # ±20% 범위
            return 0.2
        else:
            return 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0

def optimize_score_based_matching(unmatched_ebd, bd_df, already_matched_bd_pks):
    """
    최적화된 점수 기반 매칭 수행:
    1. 데이터 사전 처리로 반복 연산 최소화
    2. 필요한 컬럼만 사용하여 메모리 효율화
    3. 불필요한 조합은 사전에 필터링
    4. 이미 매칭된 BD는 다른 EBD의 후보에서 제외하여 중복 매칭 방지 + 전역 최적화도입
    5. SEQ_NO를 유니크 키로 활용하여 인덱스 문제 방지
    6. 이미 1~3차에서 매칭된 BD(MGM_BLD_PK)는 4차 매칭에서 제외
    7. RECAP_PK가 없는 레코드는 '미매칭(RECAP없음)'으로 처리
    """
    start_time = time.time()
    print("전역 최적화 점수 기반 매칭을 시작합니다...")
    print(f"이전 단계(1~3차)에서 매칭된 BD 개수: {len(already_matched_bd_pks)}개")
    
    # 결과 저장을 위한 리스트
    match_results = []
    
    # 데이터 사전 처리
    ebd_columns = ['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '기관명', '건축물명', '주소', 
                   'EBD_COUNT', 'BD_COUNT', 'EBD_OVER_BD']
    bd_columns = ['MGM_BLD_PK', 'RECAP_PK', 'TOTAREA', 'USE_DATE', 'BLD_NM', 'DONG_NM']
    
    # SEQ_NO를 인덱스로 사용할 수 있도록 원본 데이터 저장
    unmatched_ebd_by_seq = {row['SEQ_NO']: row.to_dict() for _, row in unmatched_ebd.iterrows()}
    
    # RECAP_PK가 없는 레코드는 '미매칭(RECAP없음)'으로 처리
    no_recap_mask = unmatched_ebd['RECAP_PK'].isna()
    if no_recap_mask.any():
        no_recap_count = no_recap_mask.sum()
        print(f"RECAP_PK가 없는 레코드 수: {no_recap_count}개 (이 레코드들은 '미매칭(RECAP없음)'으로 처리됩니다)")
        
        for _, row in unmatched_ebd[no_recap_mask].iterrows():
            seq_no = row['SEQ_NO']
            original_ebd_row = unmatched_ebd_by_seq[seq_no].copy()
            original_ebd_row['MATCH_STAGE'] = '미매칭(RECAP없음)'
            match_results.append(original_ebd_row)
    
    # RECAP_PK가 있는 레코드만 처리
    ebd_with_recap = unmatched_ebd[~no_recap_mask].copy()
    
    # 사용할 컬럼만 선택하고 전처리
    ebd_processed = preprocess_data(ebd_with_recap, ebd_columns)
    
    # 이미 매칭된 BD 필터링
    bd_df_filtered = bd_df[~bd_df['MGM_BLD_PK'].isin(already_matched_bd_pks)].copy()
    print(f"1~3차 매칭 제외 후 사용 가능한 BD 개수: {len(bd_df_filtered)}개")
    
    bd_processed = preprocess_data(bd_df_filtered, bd_columns, is_bd=True)
    
    # 전체 ebd 레코드 수 확인
    total_ebd_records = len(ebd_processed)
    print(f"처리할 RECAP이 있는 미매칭 EBD 레코드: {total_ebd_records}개")
    
    # RECAP별 처리
    valid_recaps = set(ebd_processed['RECAP_PK'].dropna().unique())
    
    # 결과 저장 통계
    processed_count = 0
    matched_count = 0
    
    # 면적 점수 분포 통계
    area_score_stats = {
        "1.0": 0,  # ±1% 범위
        "0.8": 0,  # ±5% 범위
        "0.5": 0,  # ±10% 범위
        "0.2": 0,  # ±20% 범위
        "0.0": 0   # 범위 외
    }
    
    # RECAP별로 처리 - 전역 최적화 적용
    for recap in tqdm(valid_recaps, desc="RECAP 처리"):
        # 해당 RECAP의 EBD와 BD
        ebd_recap = ebd_processed[ebd_processed['RECAP_PK'] == recap]
        bd_recap = bd_processed[bd_processed['RECAP_PK'] == recap]
        
        if ebd_recap.empty or bd_recap.empty:
            continue
        
        # 모든 가능한 EBD-BD 조합과 점수를 저장하는 리스트
        all_combinations = []
        
        # 각 EBD에 대한 동점 BD 후보 추적을 위한 딕셔너리
        ebd_score_counts = {}  # {seq_no: {score: count}}
        
        # 각 EBD에 대해 모든 BD와의 점수 계산
        for _, ebd_row in ebd_recap.iterrows():
            seq_no = ebd_row['SEQ_NO']
            original_ebd_row = unmatched_ebd_by_seq[seq_no]
            
            # 이 EBD의 점수 카운트 초기화
            if seq_no not in ebd_score_counts:
                ebd_score_counts[seq_no] = {}
            
            for bd_idx, bd_row in bd_recap.iterrows():
                # 1. 면적 점수
                area_score = calculate_area_score(ebd_row['연면적'], bd_row['TOTAREA'])
                
                # 면적 점수 통계 기록
                area_score_str = str(area_score)
                if area_score_str in area_score_stats:
                    area_score_stats[area_score_str] += 1
                
                # 2. 날짜 점수
                date_score = 0.0
                if not pd.isna(ebd_row['사용승인연도']) and not pd.isna(bd_row['USE_DATE']):
                    try:
                        ebd_date = str(int(ebd_row['사용승인연도']))
                        bd_date = str(int(bd_row['USE_DATE']))
                        if ebd_date == bd_date:
                            date_score = 1.0
                    except:
                        pass
                
                # 3. 텍스트 점수
                text_score = 0.0
                # 3.1 기관명이 BLD_NM에 포함
                if str(ebd_row['기관명']) != 'nan' and str(ebd_row['기관명']) in str(bd_row['BLD_NM']):
                    text_score += 1.0
                
                # 3.2 건축물명이 BLD_NM 또는 DONG_NM에 포함
                if str(ebd_row['건축물명']) != 'nan' and (
                    str(ebd_row['건축물명']) in str(bd_row['BLD_NM']) or 
                    str(ebd_row['건축물명']) in str(bd_row['DONG_NM'])
                ):
                    text_score += 1.0
                
                # 3.3 결합 텍스트 비교
                ebd_combined = (str(ebd_row['기관명']) + " " + str(ebd_row['건축물명'])).strip()
                bd_combined = (str(bd_row['BLD_NM']) + " " + str(bd_row['DONG_NM'])).strip()
                
                if ebd_combined != " " and ebd_combined in bd_combined:
                    text_score += 1.0
                
                # 총점 계산
                total_score = area_score + date_score + text_score
                
                # 점수 카운트 추적
                if total_score >= 1.8:
                    ebd_score_counts[seq_no][total_score] = ebd_score_counts[seq_no].get(total_score, 0) + 1
                
                # 매칭 최소 점수 이상인 경우만 저장
                if total_score >= 1.8:
                    all_combinations.append({
                        'seq_no': seq_no,
                        'bd_idx': bd_idx,
                        'area_score': area_score,
                        'date_score': date_score,
                        'text_score': text_score,
                        'total_score': total_score,
                        'original_ebd_row': original_ebd_row,
                        'original_bd_row': bd_df_filtered.loc[bd_idx].to_dict(),
                        'bd_pk': bd_row['MGM_BLD_PK'],
                        'has_tie': False  # 동점 여부 초기값
                    })
        
        # 각 조합에 동점 여부 표시
        for combo in all_combinations:
            seq_no = combo['seq_no']
            score = combo['total_score']
            if ebd_score_counts[seq_no][score] > 1:
                combo['has_tie'] = True
        
        # 점수 기준으로 내림차순 정렬
        all_combinations.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 이미 매칭된 EBD와 BD를 추적하기 위한 집합
        matched_seq_nos = set()
        matched_bd_indices = set()
        
        # 점수가 높은 순서대로 매칭
        for combo in all_combinations:
            seq_no = combo['seq_no']
            bd_pk = combo['bd_pk']
            
            # 이미 매칭된 EBD나 BD는 제외
            if seq_no in matched_seq_nos or bd_pk in matched_bd_indices:
                continue
            
            # 매칭 결과 생성
            result = combo['original_ebd_row'].copy()
            
            # BD 정보 추가
            for key, value in combo['original_bd_row'].items():
                if key not in result:
                    result[key] = value
            
            # 점수 정보 추가
            result['AREA_SCORE'] = combo['area_score']
            result['DATE_SCORE'] = combo['date_score']
            result['TEXT_SCORE'] = combo['text_score']
            result['TOTAL_SCORE'] = combo['total_score']
            
            # 동점 여부에 따라 매칭 스테이지 설정
            if combo['has_tie']:
                result['MATCH_STAGE'] = '4차(동점후보존재)'
            else:
                result['MATCH_STAGE'] = '4차'
            
            # 매칭된 항목 추적
            matched_seq_nos.add(seq_no)
            matched_bd_indices.add(bd_pk)
            
            match_results.append(result)
            matched_count += 1
        
        # 매칭되지 않은 EBD 처리
        for _, ebd_row in ebd_recap.iterrows():
            seq_no = ebd_row['SEQ_NO']
            if seq_no not in matched_seq_nos:
                original_ebd_row = unmatched_ebd_by_seq[seq_no].copy()
                
                # 해당 EBD의 최고 점수 찾기
                best_score = 0.0
                best_combo = None
                has_tie = False
                
                for combo in all_combinations:
                    if combo['seq_no'] == seq_no and combo['total_score'] > best_score:
                        best_score = combo['total_score']
                        best_combo = combo
                        has_tie = combo['has_tie']
                
                # 매칭되지 않은 이유에 따라 상태 설정
                if best_score >= 1.8:
                    if has_tie:
                        original_ebd_row['MATCH_STAGE'] = '미매칭(동점후보)' # 동점후보는 사람 검수 필수 (헝가리안 알고리즘 등 최적 매칭은 불가능으로 판단)
                    else:
                        original_ebd_row['MATCH_STAGE'] = '미매칭(더높은점수존재)'
                    # 점수 정보 추가 (best_combo가 None이 아님을 확인)
                    if best_combo:
                        original_ebd_row['AREA_SCORE'] = best_combo['area_score']
                        original_ebd_row['DATE_SCORE'] = best_combo['date_score']
                        original_ebd_row['TEXT_SCORE'] = best_combo['text_score']
                        original_ebd_row['TOTAL_SCORE'] = best_combo['total_score']
                elif best_score > 0:
                    original_ebd_row['MATCH_STAGE'] = '미매칭(점수미달)'
                    # 점수 정보 추가 (best_combo가 None이 아님을 확인)
                    if best_combo:
                        original_ebd_row['AREA_SCORE'] = best_combo['area_score']
                        original_ebd_row['DATE_SCORE'] = best_combo['date_score']
                        original_ebd_row['TEXT_SCORE'] = best_combo['text_score']
                        original_ebd_row['TOTAL_SCORE'] = best_combo['total_score']
                else:
                    original_ebd_row['MATCH_STAGE'] = '미매칭(후보없음)'
                
                match_results.append(original_ebd_row)
                processed_count += 1
    
    # 면적 점수 통계 출력
    print("\n면적 점수 분포:")
    for score, count in area_score_stats.items():
        print(f"- {score}점: {count}건")
    
    # 리스트를 데이터프레임으로 변환
    result_df = pd.DataFrame(match_results)
    
    # 미매칭 레코드의 점수 분포 출력
    if 'MATCH_STAGE' in result_df.columns and 'TOTAL_SCORE' in result_df.columns:
        unmatched_df = result_df[result_df['MATCH_STAGE'].str.startswith('미매칭', na=False)]
        if not unmatched_df.empty:
            # 점수 범위별 카운트
            score_bins = {
                "0~1점": 0,
                "1~1.7점": 0,
                "1.8~2.0점": 0,
                "2점 이상": 0,
                "점수 없음": 0
            }
            
            for _, row in unmatched_df.iterrows():
                if pd.isna(row.get('TOTAL_SCORE')):
                    score_bins["점수 없음"] += 1
                else:
                    score = float(row['TOTAL_SCORE'])
                    if score < 1.0:
                        score_bins["0~1점"] += 1
                    elif score < 1.8:
                        score_bins["1~1.7점"] += 1
                    elif score <= 2.0:
                        score_bins["1.8~2.0점"] += 1
                    else:
                        score_bins["2점 이상"] += 1
            
            print("\n미매칭 레코드의 총점 분포:")
            for range_label, count in score_bins.items():
                print(f"- {range_label}: {count}건")
            
            # 세부적인 미매칭 상태별 분포
            unmatch_stage_counts = unmatched_df['MATCH_STAGE'].value_counts()
            print("\n미매칭 유형별 분포:")
            for stage, count in unmatch_stage_counts.items():
                print(f"- {stage}: {count}건")
    
    # 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"매칭 완료: 총 {len(result_df)}개 중 {matched_count}개가 매칭됨 - 총 소요 시간: {elapsed_time:.2f}초")
    
    return result_df

def main():
    start_time = time.time()
    print("4차 점수 기반 매칭을 시작합니다...")
    
    # 데이터 로드
    rule_result_df, bd_df = load_data()
    print(f"데이터 로드 완료: EBD {len(rule_result_df)}개, BD {len(bd_df)}개")
    
    # 원본 순서 보존을 위한 컬럼 추가
    rule_result_df['_원본순서'] = range(len(rule_result_df))
    
    # 미매칭 EBD 레코드 직접 추출
    unmatched_ebd = rule_result_df[rule_result_df['MATCH_STAGE'] == '미매칭'].copy()
    print(f"미매칭 EBD 건수: {len(unmatched_ebd)}")
    
    # 1~3차 매칭 결과에서 이미 매칭된 BD의 MGM_BLD_PK 추출
    matched_from_rules = rule_result_df[rule_result_df['MATCH_STAGE'] != '미매칭'].copy()
    already_matched_bd_pks = set()
    
    if 'MGM_BLD_PK' in matched_from_rules.columns:
        already_matched_bd_pks = set(matched_from_rules['MGM_BLD_PK'].dropna().unique())
        print(f"1~3차에서 이미 매칭된 BD(MGM_BLD_PK) 개수: {len(already_matched_bd_pks)}개")
    
    # SEQ_NO가 누락된 레코드에 대한 처리
    if unmatched_ebd['SEQ_NO'].isnull().any():
        print("경고: SEQ_NO가 누락된 레코드가 있습니다. 임시 SEQ_NO를 할당합니다.")
        null_seq_mask = unmatched_ebd['SEQ_NO'].isnull()
        max_seq = unmatched_ebd['SEQ_NO'].max()
        unmatched_ebd.loc[null_seq_mask, 'SEQ_NO'] = range(max_seq + 1, max_seq + 1 + null_seq_mask.sum())
    
    # 최적화된 점수 기반 매칭 수행 - 이미 매칭된 BD는 제외
    score_result_df = optimize_score_based_matching(unmatched_ebd, bd_df, already_matched_bd_pks)
    
    # 통계 출력
    stage_counts = score_result_df['MATCH_STAGE'].value_counts()
    print("\n4차 점수 기반 매칭 결과:")
    for stage, count in stage_counts.items():
        print(f"- {stage}: {count}건")
    
    # 미매칭 레코드의 총점 분석
    if 'MATCH_STAGE' in score_result_df.columns and 'TOTAL_SCORE' in score_result_df.columns:
        unmatched_df = score_result_df[score_result_df['MATCH_STAGE'].str.startswith('미매칭', na=False)]
        if not unmatched_df.empty:
            # 미매칭 유형별 총점 분포
            print("\n미매칭 유형별 총점 분포:")
            for stage in unmatched_df['MATCH_STAGE'].unique():
                stage_df = unmatched_df[unmatched_df['MATCH_STAGE'] == stage]
                
                # '미매칭(RECAP없음)', '미매칭(후보없음)' 등의 경우 점수가 없을 수 있음
                if 'TOTAL_SCORE' in stage_df.columns:
                    stage_df = stage_df.dropna(subset=['TOTAL_SCORE'])
                    if not stage_df.empty:
                        mean_score = stage_df['TOTAL_SCORE'].mean()
                        max_score = stage_df['TOTAL_SCORE'].max()
                        print(f"- {stage}: 평균 {mean_score:.2f}점, 최대 {max_score:.2f}점, {len(stage_df)}건")
                    else:
                        print(f"- {stage}: 점수 정보 없음, {len(unmatched_df[unmatched_df['MATCH_STAGE'] == stage])}건")
                else:
                    print(f"- {stage}: 점수 정보 없음, {len(stage_df)}건")
    
    # 기존 매칭 결과와 새로운 점수 기반 매칭 결과 합치기
    final_result = pd.concat([matched_from_rules, score_result_df], ignore_index=False)
    
    # 원본 순서대로 정렬
    final_result = final_result.sort_values('_원본순서')
    
    # 통계 출력
    final_counts = final_result['MATCH_STAGE'].value_counts()
    print("\n최종 매칭 결과:")
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
    
    # 총 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    return final_result

if __name__ == "__main__":
    main()
