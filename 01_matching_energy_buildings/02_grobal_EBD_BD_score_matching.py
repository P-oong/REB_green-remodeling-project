import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # 진행률 표시를 위해 추가
import time  # 실행 시간 측정을 위해 추가
import re  # 정규 표현식 처리를 위해 추가

def preprocess_text(text):
    """
    텍스트 전처리: 소문자 변환, 특수문자를 공백으로 치환, 토큰화
    """
    if pd.isna(text) or text is None:
        return []
    
    # 문자열로 변환
    text = str(text).lower()
    
    # 특수문자를 공백으로 치환
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 공백으로 분할하여 토큰화
    tokens = text.split()
    
    # 빈 문자열이나 공백만 있는 토큰 제거
    tokens = [token.strip() for token in tokens if token.strip()]
    
    return tokens

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
    데이터 전처리: 숫자 컬럼 변환, 문자열 컬럼 토큰화
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
    
    # 문자열 컬럼 처리 및 토큰화
    token_columns = []
    
    if is_bd:
        # BD 데이터의 텍스트 컬럼
        text_columns = ['BLD_NM', 'DONG_NM']
        for col in text_columns:
            if col in result.columns:
                # 문자열 소문자화
                result[col] = result[col].astype(str).str.lower()
                # 토큰화된 컬럼 생성
                result[f'{col}_tokens'] = result[col].apply(preprocess_text)
                token_columns.append(f'{col}_tokens')
    else:
        # EBD 데이터의 텍스트 컬럼
        text_columns = ['기관명', '건축물명', '주소']
        for col in text_columns:
            if col in result.columns:
                # 문자열 소문자화
                result[col] = result[col].astype(str).str.lower()
                # 토큰화된 컬럼 생성
                result[f'{col}_tokens'] = result[col].apply(preprocess_text)
                token_columns.append(f'{col}_tokens')
        
        # 통합 토큰 생성 (중복 제거)
        result['ebd_unified_tokens'] = result.apply(
            lambda row: list(set(
                (row.get('기관명_tokens', []) or []) + 
                (row.get('건축물명_tokens', []) or []) + 
                (row.get('주소_tokens', []) or [])
            )),
            axis=1
        )
        token_columns.append('ebd_unified_tokens')
    
    # 원래 컬럼과 토큰 컬럼 합쳐서 반환
    return_columns = columns.copy()
    return_columns.extend(token_columns)
    
    return result[return_columns]

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
        
        # 원본 unmatched_ebd에서 토큰 컬럼 전처리
        ebd_no_recap = preprocess_data(unmatched_ebd[no_recap_mask], ebd_columns)
        
        for idx, row in unmatched_ebd[no_recap_mask].iterrows():
            seq_no = row['SEQ_NO']
            original_ebd_row = unmatched_ebd_by_seq[seq_no].copy()
            original_ebd_row['MATCH_STAGE'] = '미매칭(RECAP없음)'
            # 점수 정보 추가 (0점으로 설정)
            original_ebd_row['AREA_SCORE'] = 0.0
            original_ebd_row['DATE_SCORE'] = 0.0
            original_ebd_row['BLD_SCORE'] = 0.0
            original_ebd_row['DONG_SCORE'] = 0.0
            original_ebd_row['TOTAL_SCORE'] = 0.0
            
            # 토큰 컬럼 추가
            if seq_no in ebd_no_recap['SEQ_NO'].values:
                ebd_row = ebd_no_recap[ebd_no_recap['SEQ_NO'] == seq_no].iloc[0]
                if 'ebd_unified_tokens' in ebd_row:
                    original_ebd_row['ebd_unified_tokens'] = ebd_row['ebd_unified_tokens']
                if '기관명_tokens' in ebd_row:
                    original_ebd_row['기관명_tokens'] = ebd_row['기관명_tokens']
                if '건축물명_tokens' in ebd_row:
                    original_ebd_row['건축물명_tokens'] = ebd_row['건축물명_tokens']
                if '주소_tokens' in ebd_row:
                    original_ebd_row['주소_tokens'] = ebd_row['주소_tokens']
                
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
    
    # 텍스트 점수 분포 통계
    text_score_stats = {
        "BLD_NM (0.8)": 0,  # BLD_NM 토큰 일치
        "DONG_NM (1.0)": 0,  # DONG_NM 토큰 일치
        "모두 불일치 (0.0)": 0  # 둘 다 일치하지 않음
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
                
                # 3. 텍스트 점수 - 토큰 기반 새로운 로직
                bld_score = 0.0
                dong_score = 0.0
                
                # EBD 통합 토큰과 BD의 BLD_NM 토큰 비교
                ebd_tokens = ebd_row['ebd_unified_tokens']
                bld_nm_tokens = bd_row['BLD_NM_tokens']
                
                # 공통 토큰이 있는지 확인
                if any(token in bld_nm_tokens for token in ebd_tokens):
                    bld_score = 0.8
                    text_score_stats["BLD_NM (0.8)"] += 1
                
                # EBD 통합 토큰과 BD의 DONG_NM 토큰 비교
                dong_nm_tokens = bd_row['DONG_NM_tokens']
                
                # 공통 토큰이 있는지 확인
                if any(token in dong_nm_tokens for token in ebd_tokens):
                    dong_score = 1.0
                    text_score_stats["DONG_NM (1.0)"] += 1
                
                # 둘 다 일치하지 않는 경우 통계 기록
                if bld_score == 0.0 and dong_score == 0.0:
                    text_score_stats["모두 불일치 (0.0)"] += 1
                
                # 총점 계산 - area_score + date_score + bld_score + dong_score
                total_score = area_score + date_score + bld_score + dong_score
                
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
                        'bld_score': bld_score,
                        'dong_score': dong_score,
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
            # 필요한 BD 컬럼만 추가
            bd_essential_columns = ['MGM_BLD_PK', 'TOTAREA', 'BLD_NM', 'DONG_NM', 'USE_DATE']
            for key in bd_essential_columns:
                if key in combo['original_bd_row']:
                    result[key] = combo['original_bd_row'][key]
            
            # 토큰 컬럼 추가 (EBD와 BD 토큰 보존)
            # EBD 토큰 보존
            ebd_row = ebd_processed[ebd_processed['SEQ_NO'] == seq_no].iloc[0]
            if 'ebd_unified_tokens' in ebd_row:
                result['ebd_unified_tokens'] = ebd_row['ebd_unified_tokens']
            if '기관명_tokens' in ebd_row:
                result['기관명_tokens'] = ebd_row['기관명_tokens']
            if '건축물명_tokens' in ebd_row:
                result['건축물명_tokens'] = ebd_row['건축물명_tokens']
            if '주소_tokens' in ebd_row:
                result['주소_tokens'] = ebd_row['주소_tokens']
            
            # BD 토큰 보존
            bd_row = bd_processed.loc[bd_idx]
            if 'BLD_NM_tokens' in bd_row:
                result['BLD_NM_tokens'] = bd_row['BLD_NM_tokens']
            if 'DONG_NM_tokens' in bd_row:
                result['DONG_NM_tokens'] = bd_row['DONG_NM_tokens']
            
            # 점수 정보 추가
            result['AREA_SCORE'] = combo['area_score']
            result['DATE_SCORE'] = combo['date_score']
            result['BLD_SCORE'] = combo['bld_score']  # BLD_NM 점수
            result['DONG_SCORE'] = combo['dong_score']  # DONG_NM 점수
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
                
                # 해당 EBD의 최고 점수 찾기 (1.8 이상 & 1.8 미만 모두 포함)
                best_score = 0.0
                best_combo = None
                best_combo_under_threshold = None
                best_score_under_threshold = 0.0
                has_tie = False
                
                # 점수가 1.8 이상인 조합 중 최고 점수 찾기
                for combo in all_combinations:
                    if combo['seq_no'] == seq_no and combo['total_score'] > best_score:
                        best_score = combo['total_score']
                        best_combo = combo
                        has_tie = combo['has_tie']
                                
                # 전체 BD와의 점수 계산 결과 중 1.8 미만인 최고 점수 찾기
                for bd_idx, bd_row in bd_recap.iterrows():
                    # 이미 매칭된 BD는 제외
                    if bd_row['MGM_BLD_PK'] in matched_bd_indices:
                        continue
                                
                    # 점수 계산
                    area_score = calculate_area_score(ebd_row['연면적'], bd_row['TOTAREA'])
                    
                    # 날짜 점수
                    date_score = 0.0
                    if not pd.isna(ebd_row['사용승인연도']) and not pd.isna(bd_row['USE_DATE']):
                        try:
                            ebd_date = str(int(ebd_row['사용승인연도']))
                            bd_date = str(int(bd_row['USE_DATE']))
                            if ebd_date == bd_date:
                                date_score = 1.0
                        except:
                            pass
                    
                    # 텍스트 점수
                    bld_score = 0.0
                    dong_score = 0.0
                    
                    # EBD 통합 토큰과 BD의 BLD_NM 토큰 비교
                    ebd_tokens = ebd_row['ebd_unified_tokens']
                    bld_nm_tokens = bd_row['BLD_NM_tokens']
                    
                    # 공통 토큰이 있는지 확인
                    if any(token in bld_nm_tokens for token in ebd_tokens):
                        bld_score = 0.8
                    
                    # EBD 통합 토큰과 BD의 DONG_NM 토큰 비교
                    dong_nm_tokens = bd_row['DONG_NM_tokens']
                    
                    # 공통 토큰이 있는지 확인
                    if any(token in dong_nm_tokens for token in ebd_tokens):
                        dong_score = 1.0
                    
                    # 총점 계산
                    total_score = area_score + date_score + bld_score + dong_score
                    
                    # 1.8 미만인 경우, 최고 점수 기록
                    if total_score < 1.8 and total_score > best_score_under_threshold:
                        best_score_under_threshold = total_score
                        best_combo_under_threshold = {
                            'area_score': area_score,
                            'date_score': date_score,
                            'bld_score': bld_score,
                            'dong_score': dong_score,
                            'total_score': total_score
                        }
                
                # 매칭되지 않은 이유에 따라 상태 설정
                if best_score >= 1.8:
                    # 1.8점 이상 조합이 있는 경우
                    if has_tie:
                        original_ebd_row['MATCH_STAGE'] = '미매칭(동점후보)'
                    else:
                        original_ebd_row['MATCH_STAGE'] = '미매칭(더높은점수존재)'
                    
                    # 점수 정보 추가 (best_combo가 None이 아님을 확인)
                    if best_combo:
                        original_ebd_row['AREA_SCORE'] = best_combo['area_score']
                        original_ebd_row['DATE_SCORE'] = best_combo['date_score']
                        original_ebd_row['BLD_SCORE'] = best_combo['bld_score']
                        original_ebd_row['DONG_SCORE'] = best_combo['dong_score']
                        original_ebd_row['TOTAL_SCORE'] = best_combo['total_score']
                elif best_score_under_threshold >= 0:
                    # 1.8점 미만이지만 점수가 있는 경우
                    original_ebd_row['MATCH_STAGE'] = '미매칭(점수미달)'
                    
                    # 최고 점수 기록
                    if best_combo_under_threshold is not None:
                        original_ebd_row['AREA_SCORE'] = best_combo_under_threshold['area_score']
                        original_ebd_row['DATE_SCORE'] = best_combo_under_threshold['date_score']
                        original_ebd_row['BLD_SCORE'] = best_combo_under_threshold['bld_score']
                        original_ebd_row['DONG_SCORE'] = best_combo_under_threshold['dong_score']
                        original_ebd_row['TOTAL_SCORE'] = best_combo_under_threshold['total_score']
                    else:
                        # best_combo_under_threshold가 None인 경우 모든 점수를 0으로 설정
                        original_ebd_row['AREA_SCORE'] = 0.0
                        original_ebd_row['DATE_SCORE'] = 0.0
                        original_ebd_row['BLD_SCORE'] = 0.0
                        original_ebd_row['DONG_SCORE'] = 0.0
                        original_ebd_row['TOTAL_SCORE'] = 0.0
                else:
                    # 후보가 없거나 모든 점수가 0점인 경우
                    # BD가 있지만 이미 매칭된 경우와 진짜 후보가 없는 경우 구분
                    has_bd_but_matched = bd_recap.shape[0] > 0 and all(bd_pk in matched_bd_indices for bd_pk in bd_recap['MGM_BLD_PK'])
                    
                    if has_bd_but_matched:
                        original_ebd_row['MATCH_STAGE'] = '미매칭(이미매칭)'
                    else:
                        original_ebd_row['MATCH_STAGE'] = '미매칭(후보없음)'
                    
                    # 점수 정보 추가 (0점으로 설정)
                    original_ebd_row['AREA_SCORE'] = 0.0
                    original_ebd_row['DATE_SCORE'] = 0.0
                    original_ebd_row['BLD_SCORE'] = 0.0
                    original_ebd_row['DONG_SCORE'] = 0.0
                    original_ebd_row['TOTAL_SCORE'] = 0.0
                
                # 토큰 컬럼 추가 (EBD 토큰 보존)
                if seq_no in ebd_processed['SEQ_NO'].values:
                    ebd_row = ebd_processed[ebd_processed['SEQ_NO'] == seq_no].iloc[0]
                    if 'ebd_unified_tokens' in ebd_row:
                        original_ebd_row['ebd_unified_tokens'] = ebd_row['ebd_unified_tokens']
                    if '기관명_tokens' in ebd_row:
                        original_ebd_row['기관명_tokens'] = ebd_row['기관명_tokens']
                    if '건축물명_tokens' in ebd_row:
                        original_ebd_row['건축물명_tokens'] = ebd_row['건축물명_tokens']
                    if '주소_tokens' in ebd_row:
                        original_ebd_row['주소_tokens'] = ebd_row['주소_tokens']
                
                match_results.append(original_ebd_row)
                processed_count += 1
    
    # 면적 점수 통계 출력
    print("\n면적 점수 분포:")
    for score, count in area_score_stats.items():
        print(f"- {score}점: {count}건")
    
    # 텍스트 점수 통계 출력
    print("\n텍스트 점수 분포:")
    for score, count in text_score_stats.items():
        print(f"- {score}: {count}건")
    
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
    
    # 1~3차 매칭 레코드에 BD 토큰 추가 처리 - 사용자 요청으로 제거
    # 이미 매칭된 레코드는 토큰 정보 불필요
    print("이미 매칭된 1~3차 레코드는 토큰 정보를 추가하지 않습니다.")
    
    # 기존 매칭 결과와 새로운 점수 기반 매칭 결과 합치기
    final_result = pd.concat([matched_from_rules, score_result_df], ignore_index=False)
    
    # 원본 순서대로 정렬
    final_result = final_result.sort_values('_원본순서')
    
    # 통계 출력
    final_counts = final_result['MATCH_STAGE'].value_counts()
    print("\n최종 매칭 결과:")
    for stage, count in final_counts.items():
        print(f"- {stage}: {count}건")
    
    # 결과 저장
    os.makedirs("./result", exist_ok=True)
    
    # 원하는 컬럼 순서 정의
    desired_columns = ['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '기관명', '건축물명', '주소', '지상', '지하',
        'TOTAREA', 'BLD_NM', 'DONG_NM', 'USE_DATE', 'MGM_BLD_PK', 'MATCH_STAGE',
        'EBD_COUNT', 'BD_COUNT', 'EBD_OVER_BD']
    
    # 점수 컬럼
    score_columns = ['AREA_SCORE', 'DATE_SCORE', 'BLD_SCORE', 'DONG_SCORE', 'TOTAL_SCORE']
    
    # 토큰 컬럼 (중요: 4차 매칭 결과에만 있음)
    token_columns = ['ebd_unified_tokens', '기관명_tokens', '건축물명_tokens', '주소_tokens', 'BLD_NM_tokens', 'DONG_NM_tokens']
    
    # 최종 컬럼 순서 구성
    final_columns = []
    
    # 기본 컬럼 추가
    for col in desired_columns:
        if col in final_result.columns:
            final_columns.append(col)
    
    # 점수 컬럼 추가
    for col in score_columns:
        if col in final_result.columns:
            final_columns.append(col)
    
    # 토큰 컬럼 추가 (4차 매칭 결과에만 있음)
    for col in token_columns:
        if col in final_result.columns and col in score_result_df.columns:
            final_columns.append(col)
    
    # 혹시 누락된 컬럼이 있으면 마지막에 추가
    for col in final_result.columns:
        if col not in final_columns and col != '_원본순서':
            final_columns.append(col)
    
    # 임시 순서 컬럼 제거
    if '_원본순서' in final_result.columns:
        final_result = final_result.drop('_원본순서', axis=1)
    
    # 컬럼 존재 여부 출력 (디버깅용)
    print("\n최종 결과에 포함된 토큰 컬럼:")
    for col in token_columns:
        token_count = 0
        if col in final_result.columns:
            # 4차 매칭된 결과에서만 토큰 카운트
            token_count = len(score_result_df[score_result_df[col].notna()]) if col in score_result_df.columns else 0
            print(f"- {col}: 존재함 ({token_count}건 - 4차 매칭 결과만)")
        else:
            print(f"- {col}: 존재하지 않음")
    
    # 컬럼 순서 재정렬 (존재하는 컬럼만 선택)
    existing_columns = [col for col in final_columns if col in final_result.columns]
    final_result = final_result[existing_columns]
    
    # 안전한 파일 저장 (권한 오류 방지)
    try:
        final_result.to_excel("./result/score_matching_result_ver5.xlsx", index=False)
        print("\n최종 결과가 './result/score_matching_result_ver5.xlsx'에 저장되었습니다.")
    except PermissionError:
        # 파일이 열려있는 경우 다른 이름으로 저장 시도
        try:
            final_result.to_excel("./result/score_matching_result_ver5_new.xlsx", index=False)
            print("\n파일 권한 문제로 './result/score_matching_result_ver5_new.xlsx'에 저장되었습니다.")
        except Exception as e:
            print(f"\n파일 저장 중 오류 발생: {e}")
            
        # 마지막 시도: CSV 형식으로 저장
        try:
            final_result.to_csv("./result/score_matching_result_ver5_emergency.csv", index=False)
            print("\n엑셀 저장 실패로 CSV 형식으로 './result/score_matching_result_ver5_emergency.csv'에 저장되었습니다.")
        except Exception as e:
            print(f"\n모든 저장 시도 실패: {e}")
    
    # 총 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    return final_result

if __name__ == "__main__":
    main()
