import os
import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm

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
    rule_result_df = pd.read_excel("./result/rule_matching_result_ver5.xlsx")
    print(f"1~3차 매칭 결과 데이터 로드: {len(rule_result_df)}건")
    
    # BD 데이터 로드
    bd_df = pd.read_excel("./data/BD_data_all.xlsx")
    print(f"BD 데이터 로드: {len(bd_df)}건")
    
    return rule_result_df, bd_df

def preprocess_data(ebd_df, bd_df):
    """
    EBD와 BD 데이터의 텍스트 컬럼을 전처리하고 토큰화
    """
    print("데이터 전처리 및 토큰화 중...")
    
    # EBD 데이터 복사
    ebd_processed = ebd_df.copy()
    
    # EBD 텍스트 컬럼 처리 (기관명, 건축물명, 주소)
    for col in ['기관명', '건축물명', '주소']:
        if col in ebd_processed.columns:
            # 문자열 소문자화
            ebd_processed[col] = ebd_processed[col].astype(str).str.lower()
            # 토큰화된 컬럼 생성
            ebd_processed[f'{col}_tokens'] = ebd_processed[col].apply(preprocess_text)
    
    # EBD 통합 토큰 생성 (중복 제거)
    ebd_processed['ebd_unified_tokens'] = ebd_processed.apply(
        lambda row: list(set(
            (row.get('기관명_tokens', []) or []) + 
            (row.get('건축물명_tokens', []) or []) + 
            (row.get('주소_tokens', []) or [])
        )),
        axis=1
    )
    
    # BD 데이터 복사
    bd_processed = bd_df.copy()
    
    # BD 텍스트 컬럼 처리 (BLD_NM, DONG_NM)
    for col in ['BLD_NM', 'DONG_NM']:
        if col in bd_processed.columns:
            # 문자열 소문자화
            bd_processed[col] = bd_processed[col].astype(str).str.lower()
            # 토큰화된 컬럼 생성
            bd_processed[f'{col}_tokens'] = bd_processed[col].apply(preprocess_text)
    
    # 토큰화 결과 샘플 출력
    print("\nEBD 토큰화 결과 샘플:")
    if 'ebd_unified_tokens' in ebd_processed.columns:
        sample_ebd = ebd_processed[['SEQ_NO', '기관명', '건축물명', '주소', 'ebd_unified_tokens']].head(2)
        for _, row in sample_ebd.iterrows():
            print(f"SEQ_NO: {row['SEQ_NO']}")
            print(f"기관명: {row['기관명']} -> 토큰: {row.get('기관명_tokens', [])}")
            print(f"건축물명: {row['건축물명']} -> 토큰: {row.get('건축물명_tokens', [])}")
            print(f"주소: {row['주소']} -> 토큰: {row.get('주소_tokens', [])}")
            print(f"통합 토큰: {row['ebd_unified_tokens']}")
            print()
    
    print("\nBD 토큰화 결과 샘플:")
    if 'BLD_NM_tokens' in bd_processed.columns and 'DONG_NM_tokens' in bd_processed.columns:
        sample_bd = bd_processed[['MGM_BLD_PK', 'BLD_NM', 'DONG_NM', 'BLD_NM_tokens', 'DONG_NM_tokens']].head(2)
        for _, row in sample_bd.iterrows():
            print(f"MGM_BLD_PK: {row['MGM_BLD_PK']}")
            print(f"BLD_NM: {row['BLD_NM']} -> 토큰: {row['BLD_NM_tokens']}")
            print(f"DONG_NM: {row['DONG_NM']} -> 토큰: {row['DONG_NM_tokens']}")
            print()
    
    return ebd_processed, bd_processed

def count_common_tokens(ebd_tokens, bd_tokens):
    """
    두 토큰 리스트 간의 공통 토큰 개수를 반환
    """
    if not ebd_tokens or not bd_tokens:
        return 0
    
    # 공통 토큰 개수 세기
    common_count = sum(1 for token in ebd_tokens if token in bd_tokens)
    return common_count

def text_based_matching(unmatched_ebd, bd_df, already_matched_bd_pks):
    """
    향상된 텍스트 기반 4차 매칭 수행:
    1. 미매칭 EBD 레코드를 동일 RECAP_PK를 가진 BD 후보와 매칭
    2. DONG_NM 토큰 일치 개수를 우선 확인, 일치 토큰 개수가 가장 많고 유일한 경우만 매칭
    3. DONG_NM에서 최고 일치 개수가 동일한 BD가 여러 개이거나 일치하는 토큰이 없는 경우 BLD_NM 토큰 확인
    """
    start_time = time.time()
    print("향상된 텍스트 기반 4차 매칭을 시작합니다...")
    print(f"이전 단계(1~3차)에서 매칭된 BD 개수: {len(already_matched_bd_pks)}개")
    
    # 이미 매칭된 BD 필터링
    bd_df_filtered = bd_df[~bd_df['MGM_BLD_PK'].isin(already_matched_bd_pks)].copy()
    print(f"1~3차 매칭 제외 후 사용 가능한 BD 개수: {len(bd_df_filtered)}개")
    
    # 결과 저장을 위한 리스트
    match_results = []
    
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
    
    # 전체 ebd 레코드 수 확인
    total_ebd_records = len(ebd_with_recap)
    print(f"처리할 RECAP이 있는 미매칭 EBD 레코드: {total_ebd_records}개")
    
    # RECAP별 처리
    valid_recaps = set(ebd_with_recap['RECAP_PK'].dropna().unique())
    
    # 결과 저장 통계
    dong_match_count = 0
    bld_match_count = 0
    unmatched_count = 0
    
    # 이미 매칭된 BD를 추적하기 위한 집합
    matched_bd_pks = set(already_matched_bd_pks)  # 1~3차에서 이미 매칭된 BD
    
    # RECAP별로 처리
    for recap in tqdm(valid_recaps, desc="RECAP 처리"):
        # 해당 RECAP의 EBD와 BD
        ebd_recap = ebd_with_recap[ebd_with_recap['RECAP_PK'] == recap]
        bd_recap = bd_df_filtered[bd_df_filtered['RECAP_PK'] == recap]
        
        if ebd_recap.empty or bd_recap.empty:
            # BD 후보가 없는 경우, 모든 EBD를 미매칭으로 처리
            for _, ebd_row in ebd_recap.iterrows():
                seq_no = ebd_row['SEQ_NO']
                original_ebd_row = unmatched_ebd_by_seq[seq_no].copy()
                original_ebd_row['MATCH_STAGE'] = '미매칭(후보없음)'
                match_results.append(original_ebd_row)
                unmatched_count += 1
            continue
        
        # 각 EBD에 대해 처리
        for _, ebd_row in ebd_recap.iterrows():
            seq_no = ebd_row['SEQ_NO']
            original_ebd_row = unmatched_ebd_by_seq[seq_no].copy()
            ebd_tokens = ebd_row['ebd_unified_tokens']
            
            # DONG_NM 토큰 일치 개수 확인
            dong_matches = []
            max_dong_match = 0
            
            for _, bd_row in bd_recap.iterrows():
                # 이미 매칭된 BD는 제외
                if bd_row['MGM_BLD_PK'] in matched_bd_pks:
                    continue
                
                # DONG_NM이 NaN인지 확인
                if pd.isna(bd_row['DONG_NM']) or bd_row['DONG_NM'] == 'nan':
                    continue
                
                # DONG_NM 토큰과 EBD 통합 토큰의 일치 개수 계산
                match_count = count_common_tokens(ebd_tokens, bd_row['DONG_NM_tokens'])
                
                if match_count > 0:
                    # 더 많은 토큰이 일치하는 경우, 기존 결과 초기화
                    if match_count > max_dong_match:
                        max_dong_match = match_count
                        dong_matches = [bd_row]
                    # 동일한 수의 토큰이 일치하는 경우, 결과에 추가
                    elif match_count == max_dong_match:
                        dong_matches.append(bd_row)
            
            # DONG_NM 일치 후보가 유일한 경우 매칭
            if len(dong_matches) == 1 and max_dong_match > 0:
                # 매칭된 BD 정보 추가
                bd_match = dong_matches[0]
                for key in ['MGM_BLD_PK', 'TOTAREA', 'BLD_NM', 'DONG_NM', 'USE_DATE']:
                    if key in bd_match:
                        original_ebd_row[key] = bd_match[key]
                
                # 일치 토큰 개수 정보 추가
                original_ebd_row['MATCH_TOKEN_COUNT'] = max_dong_match
                original_ebd_row['MATCH_STAGE'] = '4차(DONG일치)'
                match_results.append(original_ebd_row)
                
                # 매칭된 BD 추적
                matched_bd_pks.add(bd_match['MGM_BLD_PK'])
                dong_match_count += 1
                continue
            
            # DONG_NM 일치 후보가 없거나 여러 개인 경우, BLD_NM 토큰으로 시도
            bld_matches = []
            max_bld_match = 0
            
            for _, bd_row in bd_recap.iterrows():
                # 이미 매칭된 BD는 제외
                if bd_row['MGM_BLD_PK'] in matched_bd_pks:
                    continue
                
                # BLD_NM이 NaN인지 확인
                if pd.isna(bd_row['BLD_NM']) or bd_row['BLD_NM'] == 'nan':
                    continue
                
                # BLD_NM 토큰과 EBD 통합 토큰의 일치 개수 계산
                match_count = count_common_tokens(ebd_tokens, bd_row['BLD_NM_tokens'])
                
                if match_count > 0:
                    # 더 많은 토큰이 일치하는 경우, 기존 결과 초기화
                    if match_count > max_bld_match:
                        max_bld_match = match_count
                        bld_matches = [bd_row]
                    # 동일한 수의 토큰이 일치하는 경우, 결과에 추가
                    elif match_count == max_bld_match:
                        bld_matches.append(bd_row)
            
            # BLD_NM 일치 후보가 유일한 경우 매칭
            if len(bld_matches) == 1 and max_bld_match > 0:
                # 매칭된 BD 정보 추가
                bd_match = bld_matches[0]
                for key in ['MGM_BLD_PK', 'TOTAREA', 'BLD_NM', 'DONG_NM', 'USE_DATE']:
                    if key in bd_match:
                        original_ebd_row[key] = bd_match[key]
                
                # 일치 토큰 개수 정보 추가
                original_ebd_row['MATCH_TOKEN_COUNT'] = max_bld_match
                original_ebd_row['MATCH_STAGE'] = '4차(BLD일치)'
                match_results.append(original_ebd_row)
                
                # 매칭된 BD 추적
                matched_bd_pks.add(bd_match['MGM_BLD_PK'])
                bld_match_count += 1
                continue
            
            # 매칭되지 않은 경우
            original_ebd_row['MATCH_STAGE'] = '미매칭'
            
            # 최고 일치 토큰 개수 기록 (미매칭의 경우에도)
            max_match_count = max(max_dong_match, max_bld_match)
            if max_match_count > 0:
                original_ebd_row['MATCH_TOKEN_COUNT'] = max_match_count
            
            match_results.append(original_ebd_row)
            unmatched_count += 1
    
    # 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"4차 매칭 완료: 총 {total_ebd_records}개 중")
    print(f"- DONG 일치 매칭: {dong_match_count}개")
    print(f"- BLD 일치 매칭: {bld_match_count}개")
    print(f"- 미매칭: {unmatched_count}개")
    print(f"총 소요 시간: {elapsed_time:.2f}초")
    
    # 리스트를 데이터프레임으로 변환
    result_df = pd.DataFrame(match_results)
    
    return result_df

def main():
    start_time = time.time()
    print("텍스트 기반 4차 매칭을 시작합니다...")
    
    # 데이터 로드
    rule_result_df, bd_df = load_data()
    
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
    
    # 데이터 전처리 및 토큰화
    unmatched_ebd_processed, bd_df_processed = preprocess_data(unmatched_ebd, bd_df)
    
    # 텍스트 기반 매칭 수행
    text_match_result_df = text_based_matching(unmatched_ebd_processed, bd_df_processed, already_matched_bd_pks)
    
    # 기존 매칭 결과와 새로운 텍스트 기반 매칭 결과 합치기
    final_result = pd.concat([matched_from_rules, text_match_result_df], ignore_index=True)
    
    # 원본 순서대로 정렬
    if '_원본순서' in final_result.columns:
        final_result = final_result.sort_values('_원본순서')
        final_result = final_result.drop('_원본순서', axis=1)
    
    # 통계 출력
    final_counts = final_result['MATCH_STAGE'].value_counts()
    print("\n최종 매칭 결과:")
    for stage, count in final_counts.items():
        print(f"- {stage}: {count}건")
    
    # 결과 저장
    os.makedirs("./result", exist_ok=True)
    
    # 원하는 컬럼 순서 정의
    desired_columns = ['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '기관명', '건축물명', '주소', '지상', '지하',
        'TOTAREA', 'BLD_NM', 'DONG_NM', 'USE_DATE', 'MGM_BLD_PK', 'MATCH_STAGE', 'MATCH_TOKEN_COUNT',
        'EBD_COUNT', 'BD_COUNT', 'EBD_OVER_BD']
    
    # 토큰 컬럼
    token_columns = ['ebd_unified_tokens', '기관명_tokens', '건축물명_tokens', '주소_tokens', 'BLD_NM_tokens', 'DONG_NM_tokens']
    
    # 최종 컬럼 순서 구성
    final_columns = []
    
    # 기본 컬럼 추가
    for col in desired_columns:
        if col in final_result.columns:
            final_columns.append(col)
    
    # 토큰 컬럼 추가
    for col in token_columns:
        if col in final_result.columns:
            final_columns.append(col)
    
    # 혹시 누락된 컬럼이 있으면 마지막에 추가
    for col in final_result.columns:
        if col not in final_columns:
            final_columns.append(col)
    
    # 컬럼 순서 재정렬 (존재하는 컬럼만 선택)
    existing_columns = [col for col in final_columns if col in final_result.columns]
    final_result = final_result[existing_columns]
    
    # 안전한 파일 저장 (권한 오류 방지)
    try:
        final_result.to_excel("./result/text_matching_result_ver1.xlsx", index=False)
        print("\n최종 결과가 './result/text_matching_result_ver1.xlsx'에 저장되었습니다.")
    except PermissionError:
        # 파일이 열려있는 경우 다른 이름으로 저장 시도
        try:
            final_result.to_excel("./result/text_matching_result_ver1_new.xlsx", index=False)
            print("\n파일 권한 문제로 './result/text_matching_result_ver1_new.xlsx'에 저장되었습니다.")
        except Exception as e:
            print(f"\n파일 저장 중 오류 발생: {e}")
            
        # 마지막 시도: CSV 형식으로 저장
        try:
            final_result.to_csv("./result/text_matching_result_ver1_emergency.csv", index=False)
            print("\n엑셀 저장 실패로 CSV 형식으로 './result/text_matching_result_ver1_emergency.csv'에 저장되었습니다.")
        except Exception as e:
            print(f"\n모든 저장 시도 실패: {e}")
    
    # 총 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    return final_result

if __name__ == "__main__":
    main() 