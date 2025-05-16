import os
import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm

def tokenize_text(text):
    """
    텍스트를 전처리하고 토큰화하여 집합(set)으로 반환
    """
    if pd.isna(text):
        return set()
    
    # 특수문자를 공백으로 변환 후 토큰화
    clean_tokens = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', str(text)).strip().split()
    return set(clean_tokens)

def tokenize_ebd_text(building_name, address):
    """
    건축물명과 주소를 전처리하여 통합 토큰 집합 생성
    - 주소의 앞 3개 토큰(지역명 등)은 제거
    """
    name_tokens = tokenize_text(building_name)
    
    # 주소 토큰화 및 앞 3개 토큰 제거
    addr_tokens = list(tokenize_text(address))
    addr_tokens = set(addr_tokens[3:]) if len(addr_tokens) > 3 else set()
    
    # 통합 토큰 집합 반환
    return name_tokens.union(addr_tokens)

def load_data():
    """
    EBD와 BD 데이터 로드
    """
    print("데이터 로드 중...")
    
    # EBD 데이터 로드
    ebd_df = pd.read_excel("./data/EBD_new_3.xlsx")
    print(f"EBD 데이터 로드: {len(ebd_df)}건")
    
    # BD 데이터 로드
    bd_df = pd.read_excel("./data/BD_data_all.xlsx")
    print(f"BD 데이터 로드: {len(bd_df)}건")
    
    return ebd_df, bd_df

def preprocess_data(ebd_df, bd_df):
    """
    EBD와 BD 데이터 전처리
    """
    print("데이터 전처리 중...")
    
    # 데이터 복사
    ebd_processed = ebd_df.copy()
    bd_processed = bd_df.copy()
    
    # === EBD 데이터 전처리 ===
    # 연면적 숫자 변환
    if '연면적' in ebd_processed.columns:
        ebd_processed['연면적_원본'] = ebd_processed['연면적']
        ebd_processed['연면적'] = pd.to_numeric(ebd_processed['연면적'], errors='coerce')
    
    # 사용승인연도 변환
    if '사용승인연도' in ebd_processed.columns:
        ebd_processed['사용승인연도_원본'] = ebd_processed['사용승인연도']
        
        # 타입 확인
        if pd.api.types.is_datetime64_dtype(ebd_processed['사용승인연도']):
            ebd_processed['사용승인연도_year'] = ebd_processed['사용승인연도'].dt.year
        else:
            try:
                # 날짜 형식으로 변환 시도
                ebd_processed['사용승인연도'] = pd.to_datetime(ebd_processed['사용승인연도'], errors='coerce')
                ebd_processed['사용승인연도_year'] = ebd_processed['사용승인연도'].dt.year
            except:
                # 숫자만 있는 경우 (연도만 있는 경우) 숫자로 변환
                ebd_processed['사용승인연도'] = pd.to_numeric(ebd_processed['사용승인연도'], errors='coerce')
                ebd_processed['사용승인연도_year'] = ebd_processed['사용승인연도']
    
    # EBD 텍스트 토큰화
    ebd_processed['ebd_tokens'] = ebd_processed.apply(
        lambda row: tokenize_ebd_text(row.get('건축물명', ''), row.get('주소', '')),
        axis=1
    )
    
    # === BD 데이터 전처리 ===
    # 연면적 숫자 변환
    if 'TOTAREA' in bd_processed.columns:
        bd_processed['TOTAREA_원본'] = bd_processed['TOTAREA']
        bd_processed['TOTAREA'] = pd.to_numeric(bd_processed['TOTAREA'], errors='coerce')
    
    # 사용승인일자 변환
    if 'USE_DATE' in bd_processed.columns:
        bd_processed['USE_DATE_원본'] = bd_processed['USE_DATE']
        
        # 타입 확인
        if pd.api.types.is_datetime64_dtype(bd_processed['USE_DATE']):
            bd_processed['USE_DATE_year'] = bd_processed['USE_DATE'].dt.year
        else:
            try:
                # 날짜 형식으로 변환 시도
                bd_processed['USE_DATE'] = pd.to_datetime(bd_processed['USE_DATE'], errors='coerce')
                bd_processed['USE_DATE_year'] = bd_processed['USE_DATE'].dt.year
            except:
                # 숫자만 있는 경우 (연도만 있는 경우) 숫자로 변환
                bd_processed['USE_DATE'] = pd.to_numeric(bd_processed['USE_DATE'], errors='coerce')
                bd_processed['USE_DATE_year'] = bd_processed['USE_DATE']
    
    # BD 텍스트 토큰화 (건축물명, 동명칭)
    bd_processed['bld_tokens'] = bd_processed['BLD_NM'].apply(tokenize_text)
    bd_processed['dong_tokens'] = bd_processed['DONG_NM'].apply(tokenize_text)
    
    # 샘플 텍스트 토큰 출력
    print("\nEBD 토큰화 샘플:")
    sample_ebd = ebd_processed[['SEQ_NO', '건축물명', '주소', 'ebd_tokens']].head(2)
    for _, row in sample_ebd.iterrows():
        print(f"SEQ_NO: {row['SEQ_NO']}")
        print(f"건축물명: {row['건축물명']}")
        print(f"주소: {row['주소']}")
        print(f"통합 토큰: {row['ebd_tokens']}")
        print()
    
    print("\nBD 토큰화 샘플:")
    sample_bd = bd_processed[['MGM_BLD_PK', 'BLD_NM', 'DONG_NM', 'bld_tokens', 'dong_tokens']].head(2)
    for _, row in sample_bd.iterrows():
        print(f"MGM_BLD_PK: {row['MGM_BLD_PK']}")
        print(f"BLD_NM: {row['BLD_NM']} -> 토큰: {row['bld_tokens']}")
        print(f"DONG_NM: {row['DONG_NM']} -> 토큰: {row['dong_tokens']}")
        print()
    
    return ebd_processed, bd_processed

def safe_equals(a, b):
    """
    NaN 값을 안전하게 비교하는 함수
    """
    if pd.isna(a) and pd.isna(b):
        return True  # 둘 다 NaN이면 같다고 처리
    elif pd.isna(a) or pd.isna(b):
        return False  # 하나만 NaN이면 다르다고 처리
    
    # datetime 타입인 경우 날짜 비교
    if isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp):
        return a == b  # 전체 날짜(연월일) 비교
    
    # 숫자면 근사값 비교 (부동소수점 오차 고려)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 0.01  # 적절한 오차 범위 설정
    
    return a == b  # 그 외에는 정확히 일치해야 함

def is_within_percentage(a, b, percentage):
    """
    두 숫자가 지정된 백분율 이내인지 확인
    """
    if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
        return False
    
    # 더 작은 값을 기준으로 허용 오차 계산
    min_val = min(a, b)
    max_val = max(a, b)
    
    # 백분율 차이 계산
    diff_percentage = ((max_val - min_val) / min_val) * 100
    
    return diff_percentage <= percentage

def has_token_match(ebd_tokens, bd_row):
    """
    EBD 토큰과 BD 토큰 간 일치 여부 확인
    """
    # DONG_NM이 있으면 먼저 확인
    if not pd.isna(bd_row['DONG_NM']) and bd_row['DONG_NM'] != 'nan':
        if ebd_tokens.intersection(bd_row['dong_tokens']):
            return True
    
    # DONG_NM이 없거나 일치하는 토큰이 없으면 BLD_NM 확인
    if not pd.isna(bd_row['BLD_NM']) and bd_row['BLD_NM'] != 'nan':
        if ebd_tokens.intersection(bd_row['bld_tokens']):
            return True
    
    return False

def run_step_matching(ebd_row, bd_candidates, step, matched_bd_pks):
    """
    각 매칭 단계에 따라 매칭 수행
    
    각 단계는 다음과 같은 조건을 검사:
    1차: 연면적 일치 + 사용승인연도 일치
    2차: 연면적만 일치
    3차: 연면적 ±1% 이내 + 사용승인연도 일치
    4차: 연면적 ±1% 이내 + 텍스트 토큰 매칭
    5차: 연면적 ±1% 이내
    6차: 연면적 ±5% 이내 + 사용승인연도 일치
    7차: 연면적 ±5% 이내 + 텍스트 토큰 매칭
    8차: 텍스트 토큰 매칭만
    9차: 연면적 ±5% 이내
    10차: RECAP 단위로 EBD와 BD가 각각 1건씩인 경우
    """
    # 이미 매칭된 BD 제외
    bd_candidates = bd_candidates[~bd_candidates['MGM_BLD_PK'].isin(matched_bd_pks)].copy()
    
    if bd_candidates.empty:
        return None, '후보없음'
    
    # 각 단계별 조건에 맞는 후보 필터링
    candidates = []
    
    if step == 1:  # 1차: 연면적 일치 + 사용승인연도 일치
        for _, bd_row in bd_candidates.iterrows():
            area_match = safe_equals(ebd_row['연면적'], bd_row['TOTAREA'])
            year_match = safe_equals(ebd_row['사용승인연도_year'], bd_row['USE_DATE_year'])
            
            if area_match and year_match:
                candidates.append(bd_row)
                
    elif step == 2:  # 2차: 연면적만 일치
        for _, bd_row in bd_candidates.iterrows():
            if safe_equals(ebd_row['연면적'], bd_row['TOTAREA']):
                candidates.append(bd_row)
                
    elif step == 3:  # 3차: 연면적 ±1% 이내 + 사용승인연도 일치
        for _, bd_row in bd_candidates.iterrows():
            area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 1)
            year_match = safe_equals(ebd_row['사용승인연도_year'], bd_row['USE_DATE_year'])
            
            if area_match and year_match:
                candidates.append(bd_row)
                
    elif step == 4:  # 4차: 연면적 ±1% 이내 + 텍스트 토큰 매칭
        for _, bd_row in bd_candidates.iterrows():
            area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 1)
            token_match = has_token_match(ebd_row['ebd_tokens'], bd_row)
            
            if area_match and token_match:
                candidates.append(bd_row)
                
    elif step == 5:  # 5차: 연면적 ±1% 이내
        for _, bd_row in bd_candidates.iterrows():
            if is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 1):
                candidates.append(bd_row)
                
    elif step == 6:  # 6차: 연면적 ±5% 이내 + 사용승인연도 일치
        for _, bd_row in bd_candidates.iterrows():
            area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 5)
            year_match = safe_equals(ebd_row['사용승인연도_year'], bd_row['USE_DATE_year'])
            
            if area_match and year_match:
                candidates.append(bd_row)
                
    elif step == 7:  # 7차: 연면적 ±5% 이내 + 텍스트 토큰 매칭
        for _, bd_row in bd_candidates.iterrows():
            area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 5)
            token_match = has_token_match(ebd_row['ebd_tokens'], bd_row)
            
            if area_match and token_match:
                candidates.append(bd_row)
                
    elif step == 8:  # 8차: 텍스트 토큰 매칭만
        for _, bd_row in bd_candidates.iterrows():
            if has_token_match(ebd_row['ebd_tokens'], bd_row):
                candidates.append(bd_row)
                
    elif step == 9:  # 9차: 연면적 ±5% 이내
        for _, bd_row in bd_candidates.iterrows():
            if is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 5):
                candidates.append(bd_row)
    
    # 10차는 이후 로직에서 별도 처리 (RECAP 단위로 EBD와 BD가 각각 1건씩인 경우)
    
    # 매칭 후보가 단 1개인 경우에만 매칭 성립
    if len(candidates) == 1:
        return candidates[0], f'{step}차'
    
    return None, '미매칭'

def match_ebd_bd(ebd_df, bd_df):
    """
    10단계 매칭 로직을 적용하여 EBD와 BD 매칭
    """
    print("EBD-BD 매칭 시작...")
    start_time = time.time()
    
    # 매칭 결과를 저장할 데이터프레임 준비
    results = ebd_df.copy()
    results['MATCH_STAGE'] = '미매칭'
    results['MGM_BLD_PK'] = None
    results['TOTAREA'] = None
    results['BLD_NM'] = None
    results['DONG_NM'] = None
    results['USE_DATE'] = None
    
    # RECAP_PK별 EBD와 BD 개수 계산
    ebd_counts = ebd_df.groupby('RECAP_PK').size().to_dict()
    bd_counts = bd_df.groupby('RECAP_PK').size().to_dict()
    
    # 이미 매칭된 BD 추적
    matched_bd_pks = set()
    
    # 각 EBD에 대해 매칭 시도
    for i, ebd_row in tqdm(ebd_df.iterrows(), total=len(ebd_df), desc="EBD 매칭"):
        # RECAP_PK가 없으면 '미매칭(RECAP없음)'으로 처리
        if pd.isna(ebd_row['RECAP_PK']):
            results.loc[i, 'MATCH_STAGE'] = '미매칭(RECAP없음)'
            continue
        
        recap = ebd_row['RECAP_PK']
        bd_candidates = bd_df[bd_df['RECAP_PK'] == recap].copy()
        
        # 해당 RECAP의 BD 후보가 없으면 '미매칭(후보없음)'으로 처리
        if bd_candidates.empty:
            results.loc[i, 'MATCH_STAGE'] = '미매칭(후보없음)'
            continue
        
        # 1~9차 매칭 시도
        matched = False
        for step in range(1, 10):
            bd_match, match_stage = run_step_matching(ebd_row, bd_candidates, step, matched_bd_pks)
            
            if bd_match is not None:
                # 매칭 정보 저장
                results.loc[i, 'MGM_BLD_PK'] = bd_match['MGM_BLD_PK']
                results.loc[i, 'TOTAREA'] = bd_match['TOTAREA']
                results.loc[i, 'BLD_NM'] = bd_match['BLD_NM']
                results.loc[i, 'DONG_NM'] = bd_match['DONG_NM']
                results.loc[i, 'USE_DATE'] = bd_match['USE_DATE']
                results.loc[i, 'MATCH_STAGE'] = match_stage
                
                # 토큰 정보도 저장
                results.loc[i, 'bld_tokens'] = bd_match['bld_tokens']
                results.loc[i, 'dong_tokens'] = bd_match['dong_tokens']
                
                # 매칭된 BD 추적
                matched_bd_pks.add(bd_match['MGM_BLD_PK'])
                matched = True
                break
        
        # 1~9차에서 매칭되지 않은 경우, 10차 매칭 시도
        if not matched:
            # 해당 RECAP의 EBD와 BD가 각각 1건씩만 존재하는 경우
            if ebd_counts.get(recap, 0) == 1 and bd_counts.get(recap, 0) == 1:
                # 유일한 BD 후보 선택 (이미 매칭되지 않은 경우에만)
                available_bd = bd_candidates[~bd_candidates['MGM_BLD_PK'].isin(matched_bd_pks)]
                
                if len(available_bd) == 1:
                    bd_match = available_bd.iloc[0]
                    
                    # 매칭 정보 저장
                    results.loc[i, 'MGM_BLD_PK'] = bd_match['MGM_BLD_PK']
                    results.loc[i, 'TOTAREA'] = bd_match['TOTAREA']
                    results.loc[i, 'BLD_NM'] = bd_match['BLD_NM']
                    results.loc[i, 'DONG_NM'] = bd_match['DONG_NM']
                    results.loc[i, 'USE_DATE'] = bd_match['USE_DATE']
                    results.loc[i, 'MATCH_STAGE'] = '10차'
                    
                    # 토큰 정보도 저장
                    results.loc[i, 'bld_tokens'] = bd_match['bld_tokens']
                    results.loc[i, 'dong_tokens'] = bd_match['dong_tokens']
                    
                    # 매칭된 BD 추적
                    matched_bd_pks.add(bd_match['MGM_BLD_PK'])
    
    # 매칭 통계
    elapsed_time = time.time() - start_time
    print(f"\n매칭 완료: 소요 시간 {elapsed_time:.2f}초")
    
    stage_counts = results['MATCH_STAGE'].value_counts()
    print("\n매칭 단계별 통계:")
    
    total_count = len(results)
    matched_count = 0
    
    for stage, count in stage_counts.items():
        percentage = (count / total_count) * 100
        print(f"- {stage}: {count}건 ({percentage:.2f}%)")
        
        if not stage.startswith('미매칭'):
            matched_count += count
    
    # 전체 매칭율
    match_percentage = (matched_count / total_count) * 100
    print(f"\n총 EBD: {total_count}건")
    print(f"매칭 성공: {matched_count}건 ({match_percentage:.2f}%)")
    print(f"미매칭: {total_count - matched_count}건 ({100 - match_percentage:.2f}%)")
    
    # RECAP별 EBD, BD 개수 정보 추가
    for recap in results['RECAP_PK'].dropna().unique():
        results.loc[results['RECAP_PK'] == recap, 'EBD_COUNT'] = ebd_counts.get(recap, 0)
        results.loc[results['RECAP_PK'] == recap, 'BD_COUNT'] = bd_counts.get(recap, 0)
    
    return results

def main():
    start_time = time.time()
    print("10단계 EBD-BD 매칭 프로세스 시작...")
    
    # 데이터 로드
    ebd_df, bd_df = load_data()
    
    # 데이터 전처리
    ebd_processed, bd_processed = preprocess_data(ebd_df, bd_df)
    
    # EBD-BD 매칭 수행
    matching_results = match_ebd_bd(ebd_processed, bd_processed)
    
    # 결과 저장
    os.makedirs("./result", exist_ok=True)
    
    # 원하는 컬럼 순서 정의
    desired_columns = [
        'SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '건축물명', '주소', '지상', '지하',
        'TOTAREA', 'BLD_NM', 'DONG_NM', 'USE_DATE', 'MGM_BLD_PK', 'MATCH_STAGE',
        'EBD_COUNT', 'BD_COUNT'
    ]
    
    # 토큰 컬럼
    token_columns = ['ebd_tokens', 'bld_tokens', 'dong_tokens']
    
    # 최종 컬럼 순서 구성
    final_columns = []
    
    # 기본 컬럼 추가
    for col in desired_columns:
        if col in matching_results.columns:
            final_columns.append(col)
    
    # 토큰 컬럼 추가
    for col in token_columns:
        if col in matching_results.columns:
            final_columns.append(col)
    
    # 혹시 누락된 컬럼이 있으면 마지막에 추가
    for col in matching_results.columns:
        if col not in final_columns:
            final_columns.append(col)
    
    # 컬럼 순서 재정렬 (존재하는 컬럼만 선택)
    existing_columns = [col for col in final_columns if col in matching_results.columns]
    final_results = matching_results[existing_columns]
    
    # 토큰 컬럼을 문자열로 변환 (셋을 문자열로 변환하여 가독성 향상)
    if 'ebd_tokens' in final_results.columns:
        final_results['ebd_tokens_str'] = final_results['ebd_tokens'].apply(lambda x: ', '.join(sorted(x)) if isinstance(x, set) else str(x))
    
    if 'bld_tokens' in final_results.columns:
        final_results['bld_tokens_str'] = final_results['bld_tokens'].apply(lambda x: ', '.join(sorted(x)) if isinstance(x, set) else str(x))
    
    if 'dong_tokens' in final_results.columns:
        final_results['dong_tokens_str'] = final_results['dong_tokens'].apply(lambda x: ', '.join(sorted(x)) if isinstance(x, set) else str(x))
    
    # 안전한 파일 저장
    try:
        final_results.to_excel("./result/multi_stage_matching_result.xlsx", index=False)
        print("\n최종 결과가 './result/multi_stage_matching_result.xlsx'에 저장되었습니다.")
    except PermissionError:
        # 파일이 열려있는 경우 다른 이름으로 저장 시도
        try:
            final_results.to_excel("./result/multi_stage_matching_result_new.xlsx", index=False)
            print("\n파일 권한 문제로 './result/multi_stage_matching_result_new.xlsx'에 저장되었습니다.")
        except Exception as e:
            print(f"\n파일 저장 중 오류 발생: {e}")
            # CSV 형식으로 저장 시도
            final_results.to_csv("./result/multi_stage_matching_result_emergency.csv", index=False)
            print("\nCSV 형식으로 './result/multi_stage_matching_result_emergency.csv'에 저장되었습니다.")
    
    # 총 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    return final_results

if __name__ == "__main__":
    main() 