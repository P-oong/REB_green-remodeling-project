import os
import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm
from collections import defaultdict

def tokenize_text(text):
    """
    텍스트를 전처리하고 토큰화하여 집합(set)으로 반환
    """
    if pd.isna(text) or text is None:
        return set()
    
    # 문자열로 변환 및 소문자화
    text = str(text).lower()
    
    # 특수문자를 공백으로 치환 (02와 동일한 정규식 사용)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 공백으로 분할하여 토큰화
    tokens = text.split()
    
    # 빈 문자열이나 공백만 있는 토큰 제거
    tokens = [token.strip() for token in tokens if token.strip()]
    
    # 리스트를 집합으로 변환
    return set(tokens)

def tokenize_ebd_text(building_name, address):
    """
    건축물명과 주소를 전처리하여 통합 토큰 집합 생성
    - 주소의 앞 3개 토큰(지역명 등)은 제거
    """
    # 건축물명 토큰화
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
    
    # BD 텍스트 토큰화 - 02_text_based와 유사한 방식 사용
    print("BD 토큰화 진행 중...")
    
    # DONG_NM 토큰화 - NaN이 아닌 경우에만 처리
    bd_processed['dong_tokens'] = bd_processed['DONG_NM'].apply(
        lambda x: set() if pd.isna(x) or x == 'nan' else tokenize_text(x)
    )
    
    # BLD_NM 토큰화 - NaN이 아닌 경우에만 처리
    bd_processed['bld_tokens'] = bd_processed['BLD_NM'].apply(
        lambda x: set() if pd.isna(x) or x == 'nan' else tokenize_text(x)
    )
    
    # 샘플 텍스트 토큰 출력 (디버깅용, 실제 매칭에는 영향 없음)
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

def safe_equals(a, b, strict=False):
    """
    NaN 값을 안전하게 비교하는 함수
    strict=True이면 숫자에 대해 완전히 정확한 비교 수행(1차 매칭용)
    """
    if pd.isna(a) and pd.isna(b):
        return True  # 둘 다 NaN이면 같다고 처리
    elif pd.isna(a) or pd.isna(b):
        return False  # 하나만 NaN이면 다르다고 처리
    
    # datetime 타입인 경우 날짜 비교
    if isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp):
        return a == b  # 전체 날짜(연월일) 비교
    
    # 숫자 비교
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if strict:
            # 1차 매칭: 완전히 정확한 일치 필요
            return a == b
        else:
            # 2차 이후 매칭: 근사값 비교 (부동소수점 오차 고려)
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
    # DONG_NM 토큰이 있는 경우 확인
    if 'dong_tokens' in bd_row and bd_row['dong_tokens']:  # 빈 집합이 아닌 경우
        # 토큰 교집합 확인
        if ebd_tokens.intersection(bd_row['dong_tokens']):
            return True
    
    # BLD_NM 토큰이 있는 경우 확인
    if 'bld_tokens' in bd_row and bd_row['bld_tokens']:  # 빈 집합이 아닌 경우
        # 토큰 교집합 확인
        if ebd_tokens.intersection(bd_row['bld_tokens']):
            return True
    
    return False

def check_match_condition(ebd_row, bd_row, step):
    """
    각 단계별 매칭 조건 확인
    4, 7, 8차는 토큰 교집합 개수(int) 반환, 나머지는 True/False 반환
    """
    if step == 1:  # 1차: 연면적 일치 + 사용승인연도 일치 (정확히 일치)
        area_match = safe_equals(ebd_row['연면적'], bd_row['TOTAREA'], strict=True)
        year_match = safe_equals(ebd_row['사용승인연도_year'], bd_row['USE_DATE_year'], strict=True)
        return area_match and year_match
    elif step == 2:  # 2차: 연면적만 일치
        return safe_equals(ebd_row['연면적'], bd_row['TOTAREA'])
    elif step == 3:  # 3차: 연면적 ±1% 이내 + 사용승인연도 일치
        area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 1)
        year_match = safe_equals(ebd_row['사용승인연도_year'], bd_row['USE_DATE_year'])
        return area_match and year_match
    elif step == 4:  # 4차: 연면적 ±1% 이내 + 텍스트 토큰 매칭
        area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 1)
        if not area_match:
            return 0
        dong_score = len(ebd_row['ebd_tokens'].intersection(bd_row['dong_tokens']))
        bld_score = len(ebd_row['ebd_tokens'].intersection(bd_row['bld_tokens']))
        return max(dong_score, bld_score)
    elif step == 5:  # 5차: 연면적 ±1% 이내
        return is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 1)
    elif step == 6:  # 6차: 연면적 ±5% 이내 + 사용승인연도 일치
        area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 5)
        year_match = safe_equals(ebd_row['사용승인연도_year'], bd_row['USE_DATE_year'])
        return area_match and year_match
    elif step == 7:  # 7차: 연면적 ±5% 이내 + 텍스트 토큰 매칭
        area_match = is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 5)
        if not area_match:
            return 0
        dong_score = len(ebd_row['ebd_tokens'].intersection(bd_row['dong_tokens']))
        bld_score = len(ebd_row['ebd_tokens'].intersection(bd_row['bld_tokens']))
        return max(dong_score, bld_score)
    elif step == 8:  # 8차: 텍스트 토큰 매칭만
        dong_score = len(ebd_row['ebd_tokens'].intersection(bd_row['dong_tokens']))
        bld_score = len(ebd_row['ebd_tokens'].intersection(bd_row['bld_tokens']))
        return max(dong_score, bld_score)
    elif step == 9:  # 9차: 연면적 ±5% 이내
        return is_within_percentage(ebd_row['연면적'], bd_row['TOTAREA'], 5)
    elif step == 10:  # 10차: RECAP 단위로 EBD와 BD가 각각 1건씩인 경우
        return True
    return False

def find_unique_matches(possible_matches):
    """
    유일한 매칭만 찾아서 반환
    - EBD가 하나의 BD에만 매칭되고
    - BD가 하나의 EBD에만 매칭되는 경우만 선택
    """
    ebd_to_bds = defaultdict(list)  # 각 EBD가 매칭 가능한 BD 목록
    bd_to_ebds = defaultdict(list)  # 각 BD가 매칭 가능한 EBD 목록
    
    # 매칭 가능성 수집
    for ebd_idx, bd_pk in possible_matches:
        ebd_to_bds[ebd_idx].append(bd_pk)
        bd_to_ebds[bd_pk].append(ebd_idx)
    
    # 유일한 매칭만 선택
    confirmed = []
    for ebd_idx, bd_list in ebd_to_bds.items():
        if len(bd_list) == 1:  # EBD가 하나의 BD에만 매칭 가능
            bd_pk = bd_list[0]
            if len(bd_to_ebds[bd_pk]) == 1:  # 그 BD도 하나의 EBD에만 매칭 가능
                confirmed.append((ebd_idx, bd_pk))
    
    return confirmed

def token_intersection_count(ebd_tokens, bd_tokens):
    if not ebd_tokens or not bd_tokens:
        return 0
    return len(ebd_tokens.intersection(bd_tokens))

def match_ebd_bd_batch(ebd_df, bd_df):
    print("단계별 일괄 EBD-BD 매칭 시작...")
    start_time = time.time()
    results = ebd_df.copy()
    results['MATCH_STAGE'] = '미매칭'
    results['MGM_BLD_PK'] = None
    results['TOTAREA'] = None
    results['BLD_NM'] = None
    results['DONG_NM'] = None
    results['USE_DATE'] = None
    ebd_counts = ebd_df.groupby('RECAP_PK').size().to_dict()
    bd_counts = bd_df.groupby('RECAP_PK').size().to_dict()
    unmatched_ebd_indices = set(ebd_df.index)
    unmatched_bd_pks = set(bd_df['MGM_BLD_PK'])
    for step in tqdm(range(1, 10), desc="단계별 매칭"):
        print(f"\n{step}차 매칭 시작...")
        possible_matches = []
        to_remove_ebd_indices = []
        for i in tqdm(list(unmatched_ebd_indices), desc=f"{step}차 EBD 평가", leave=False):
            ebd_row = ebd_df.loc[i]
            if pd.isna(ebd_row['RECAP_PK']):
                results.loc[i, 'MATCH_STAGE'] = '미매칭(RECAP없음)'
                to_remove_ebd_indices.append(i)
                continue
            bd_candidates = bd_df[(bd_df['RECAP_PK'] == ebd_row['RECAP_PK']) & (bd_df['MGM_BLD_PK'].isin(unmatched_bd_pks))]
            if bd_candidates.empty:
                if results.loc[i, 'MATCH_STAGE'] == '미매칭':
                    results.loc[i, 'MATCH_STAGE'] = '미매칭(후보없음)'
                continue
            # 4,7,8차는 토큰 교집합 개수 최대값 매칭
            if step in [4, 7, 8]:
                candidate_scores = []
                for _, bd_row in bd_candidates.iterrows():
                    score = check_match_condition(ebd_row, bd_row, step)
                    if score > 0:
                        candidate_scores.append((bd_row['MGM_BLD_PK'], score))
                if not candidate_scores:
                    continue
                max_score = max([s for _, s in candidate_scores])
                if max_score == 0:
                    continue
                max_score_candidates = [pk for pk, s in candidate_scores if s == max_score]
                if len(max_score_candidates) == 1:
                    possible_matches.append((i, max_score_candidates[0]))
                # 동점 2개 이상이면 매칭하지 않음(다음 단계로)
            else:
                for _, bd_row in bd_candidates.iterrows():
                    if check_match_condition(ebd_row, bd_row, step):
                        possible_matches.append((i, bd_row['MGM_BLD_PK']))
        for idx in to_remove_ebd_indices:
            unmatched_ebd_indices.remove(idx)
        confirmed_matches = find_unique_matches(possible_matches)
        print(f"{step}차 가능한 매칭: {len(possible_matches)}건, 확정된 매칭: {len(confirmed_matches)}건")
        for ebd_idx, bd_pk in confirmed_matches:
            bd_row = bd_df[bd_df['MGM_BLD_PK'] == bd_pk].iloc[0]
            results.loc[ebd_idx, 'MGM_BLD_PK'] = bd_pk
            results.loc[ebd_idx, 'TOTAREA'] = bd_row['TOTAREA']
            results.loc[ebd_idx, 'BLD_NM'] = bd_row['BLD_NM']
            results.loc[ebd_idx, 'DONG_NM'] = bd_row['DONG_NM']
            results.loc[ebd_idx, 'USE_DATE'] = bd_row['USE_DATE']
            results.loc[ebd_idx, 'MATCH_STAGE'] = f'{step}차'
        to_remove_ebd_indices = [ebd_idx for ebd_idx, _ in confirmed_matches]
        to_remove_bd_pks = [bd_pk for _, bd_pk in confirmed_matches]
        for idx in to_remove_ebd_indices:
            if idx in unmatched_ebd_indices:
                unmatched_ebd_indices.remove(idx)
        for pk in to_remove_bd_pks:
            if pk in unmatched_bd_pks:
                unmatched_bd_pks.remove(pk)
    # 10차 매칭: RECAP 단위로 EBD와 BD가 각각 1건씩인 경우
    print("\n10차 매칭 시작 (RECAP 단위 1:1 매칭)...")
    # 미매칭 EBD를 RECAP_PK로 그룹화
    unmatched_ebd_by_recap = {}
    for i in unmatched_ebd_indices:
        ebd_row = ebd_df.loc[i]
        recap_pk = ebd_row['RECAP_PK']
        if pd.isna(recap_pk):
            continue
        
        if recap_pk not in unmatched_ebd_by_recap:
            unmatched_ebd_by_recap[recap_pk] = []
        unmatched_ebd_by_recap[recap_pk].append(i)
    
    # 미매칭 BD를 RECAP_PK로 그룹화
    unmatched_bd_by_recap = {}
    for pk in unmatched_bd_pks:
        bd_row = bd_df[bd_df['MGM_BLD_PK'] == pk].iloc[0]
        recap_pk = bd_row['RECAP_PK']
        if pd.isna(recap_pk):
            continue
        
        if recap_pk not in unmatched_bd_by_recap:
            unmatched_bd_by_recap[recap_pk] = []
        unmatched_bd_by_recap[recap_pk].append(pk)
    
    # 10차 매칭 실행 (RECAP 당 EBD와 BD가 각각 1개씩인 경우)
    tenth_stage_count = 0
    to_remove_ebd_indices = []
    to_remove_bd_pks = []
    
    for recap_pk in unmatched_ebd_by_recap.keys():
        # 해당 RECAP에 EBD와 BD가 각각 1개씩인 경우만 처리
        if recap_pk in unmatched_bd_by_recap and len(unmatched_ebd_by_recap[recap_pk]) == 1 and len(unmatched_bd_by_recap[recap_pk]) == 1:
            ebd_idx = unmatched_ebd_by_recap[recap_pk][0]
            bd_pk = unmatched_bd_by_recap[recap_pk][0]
            
            bd_row = bd_df[bd_df['MGM_BLD_PK'] == bd_pk].iloc[0]
            
            # 매칭 정보 저장
            results.loc[ebd_idx, 'MGM_BLD_PK'] = bd_pk
            results.loc[ebd_idx, 'TOTAREA'] = bd_row['TOTAREA']
            results.loc[ebd_idx, 'BLD_NM'] = bd_row['BLD_NM']
            results.loc[ebd_idx, 'DONG_NM'] = bd_row['DONG_NM']
            results.loc[ebd_idx, 'USE_DATE'] = bd_row['USE_DATE']
            results.loc[ebd_idx, 'MATCH_STAGE'] = '10차'
            
            # 제거할 항목 리스트에 추가
            to_remove_ebd_indices.append(ebd_idx)
            to_remove_bd_pks.append(bd_pk)
            tenth_stage_count += 1
    
    # 매칭 완료된 항목 한꺼번에 제외
    for idx in to_remove_ebd_indices:
        if idx in unmatched_ebd_indices:
            unmatched_ebd_indices.remove(idx)
            
    for pk in to_remove_bd_pks:
        if pk in unmatched_bd_pks:
            unmatched_bd_pks.remove(pk)
    
    print(f"10차 매칭 완료: {tenth_stage_count}건")
    
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

def test_tokenization():
    """
    토큰화 및 토큰 매칭 테스트를 수행하는 함수
    """
    print("\n===== 토큰화 테스트 =====")
    test_words = [
        "과학기술전시체험센터",
        "과학 기술 전시 체험 센터",
        "과학기술 전시체험센터",
        "서울 과학기술전시체험센터"
    ]
    
    for word in test_words:
        tokens = tokenize_text(word)
        print(f"원본: '{word}' -> 토큰: {tokens}")
    
    print("\n===== 토큰 매칭 테스트 =====")
    # 가상의 EBD 토큰
    ebd_tokens = tokenize_text("과학기술전시체험센터")
    print(f"EBD 토큰: {ebd_tokens}")
    
    # 가상의 BD 행 생성
    bd_row1 = {
        'DONG_NM': "과학기술전시체험센터",
        'BLD_NM': "교육센터",
        'dong_tokens': tokenize_text("과학기술전시체험센터"),
        'bld_tokens': tokenize_text("교육센터")
    }
    
    bd_row2 = {
        'DONG_NM': "과학 기술 전시 체험 센터",
        'BLD_NM': "교육센터",
        'dong_tokens': tokenize_text("과학 기술 전시 체험 센터"),
        'bld_tokens': tokenize_text("교육센터")
    }
    
    bd_row3 = {
        'DONG_NM': "지원센터",
        'BLD_NM': "과학기술 전시체험센터",
        'dong_tokens': tokenize_text("지원센터"),
        'bld_tokens': tokenize_text("과학기술 전시체험센터")
    }
    
    # 토큰 매칭 테스트
    print(f"BD 행1 (DONG_NM='과학기술전시체험센터') 매칭 결과: {has_token_match(ebd_tokens, bd_row1)}")
    print(f"BD 행2 (DONG_NM='과학 기술 전시 체험 센터') 매칭 결과: {has_token_match(ebd_tokens, bd_row2)}")
    print(f"BD 행3 (BLD_NM='과학기술 전시체험센터') 매칭 결과: {has_token_match(ebd_tokens, bd_row3)}")
    

def main():
    # 토큰화 테스트 먼저 실행
    test_tokenization()
    
    start_time = time.time()
    print("\n단계별 일괄 EBD-BD 매칭 프로세스 시작...")
    
    # 데이터 로드
    ebd_df, bd_df = load_data()
    
    # 데이터 전처리
    ebd_processed, bd_processed = preprocess_data(ebd_df, bd_df)
    
    # EBD-BD 매칭 수행 (단계별 일괄 매칭 알고리즘)
    matching_results = match_ebd_bd_batch(ebd_processed, bd_processed)
    
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
    
    # 안전한 파일 저장
    try:
        final_results.to_excel("./result/batch_stage_matching_result_ver5.xlsx", index=False)
        print("\n최종 결과가 './result/batch_stage_matching_result_ver5.xlsx'에 저장되었습니다.")
    except PermissionError:
        # 파일이 열려있는 경우 다른 이름으로 저장 시도
        try:
            final_results.to_excel("./result/batch_stage_matching_result_ver5.xlsx", index=False)
            print("\n파일 권한 문제로 './result/batch_stage_matching_result_ver5.xlsx'에 저장되었습니다.")
        except Exception as e:
            print(f"\n파일 저장 중 오류 발생: {e}")
            # CSV 형식으로 저장 시도
            final_results.to_csv("./result/batch_stage_matching_result_ver5.csv", index=False)
            print("\nCSV 형식으로 './result/batch_stage_matching_result_ver5.csv'에 저장되었습니다.")
    
    # 총 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    return final_results

if __name__ == "__main__":
    main() 