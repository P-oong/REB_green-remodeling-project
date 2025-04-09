import os
import pandas as pd
import numpy as np

def parse_area_values(df, area_column):
    """
    연면적 컬럼의 값을 숫자로 변환
    """
    df = df.copy()
    if area_column in df.columns:
        # 변환 전 값 저장 (디버깅용)
        df[f'{area_column}_원본'] = df[area_column]
        df[area_column] = pd.to_numeric(df[area_column], errors='coerce')
    return df

def standardize_date(df, date_column):
    """
    날짜 데이터 표준화 - 전체 날짜 정보 보존
    """
    df = df.copy()
    if date_column in df.columns:
        # 변환 전 값 저장 (디버깅용)
        df[f'{date_column}_원본'] = df[date_column]
        
        # 원본 타입 확인
        print(f"{date_column} 컬럼의 원본 타입: {df[date_column].dtype}")
        
        # 이미 datetime 타입이면 그대로 사용
        if pd.api.types.is_datetime64_dtype(df[date_column]):
            print(f"{date_column}은 이미 datetime 타입입니다.")
            # 연도만 저장하는 컬럼 추가 (이전 버전과의 호환성 유지)
            df[f'{date_column}_year'] = df[date_column].dt.year
        else:
            try:
                # 날짜 형식으로 변환 시도 (전체 날짜 정보 보존)
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                # 연도만 저장하는 컬럼 추가 (이전 버전과의 호환성 유지)
                df[f'{date_column}_year'] = df[date_column].dt.year
                print(f"{date_column}이 datetime 타입으로 변환되었습니다.")
            except:
                # 숫자만 있는 경우 (연도만 있는 경우) 숫자로 변환
                df[date_column] = pd.to_numeric(df[date_column], errors='coerce')
                df[f'{date_column}_year'] = df[date_column]  # 숫자형 날짜는 그대로 연도로 간주
                print(f"{date_column}이 숫자 타입으로 변환되었습니다.")
        
        # 변환 후 샘플 데이터 출력
        print(f"{date_column} 변환 후 샘플: {df[date_column].head(3)}")
    return df

def safe_equals(a, b):
    """
    NaN 값을 안전하게 비교하는 함수 - 날짜 비교 로직 개선
    """
    if pd.isna(a) and pd.isna(b):
        return True  # 둘 다 NaN이면 같다고 처리
    elif pd.isna(a) or pd.isna(b):
        return False  # 하나만 NaN이면 다르다고 처리
    else:
        # datetime 타입인 경우 날짜 비교
        if isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp):
            return a == b  # 전체 날짜(연월일) 비교
        # 숫자면 근사값 비교 (부동소수점 오차 고려)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) < 0.01  # 적절한 오차 범위 설정
        else:
            return a == b  # 그 외에는 정확히 일치해야 함

def match_candidate(group, bd_counts, ebd_counts):
    """
    group: 동일한 EBD 레코드(SEQ_NO)에 대해, 같은 RECAP 내 BD 후보들의 병합된 데이터
    순서대로 조건을 적용하여 매칭 후보를 선택:
      1. 1차: 연면적과 승인일자가 모두 일치하는 후보가 있으면 선택
      2. 2차: 1차 후보가 없으면, 연면적만 일치하는 후보가 있으면 선택
      3. 3차: 위 후보가 없고, 동일 RECAP의 EBD와 BD가 각각 1건씩인 경우에만 선택
      4. 그 외에는 미매칭 처리 (이때, MGM_BLD_PK를 None으로 설정)
    """
    group = group.copy()
    
    # 디버깅 정보 출력
    seq_no = group['SEQ_NO'].iloc[0]
    recap = group['RECAP_PK'].iloc[0]
    
    # 데이터 타입 및 값 로그
    ebd_area = group['연면적'].iloc[0]
    bd_area = group['TOTAREA'].iloc[0]
    ebd_date = group['사용승인연도'].iloc[0]
    bd_date = group['USE_DATE'].iloc[0]
    
    print(f"\n디버깅 - RECAP_PK: {recap}, SEQ_NO: {seq_no}")
    print(f"연면적(EBD): {ebd_area} (타입: {type(ebd_area)}), TOTAREA(BD): {bd_area} (타입: {type(bd_area)})")
    print(f"사용승인연도(EBD): {ebd_date} (타입: {type(ebd_date)}), USE_DATE(BD): {bd_date} (타입: {type(bd_date)})")
    
    # 1차 매칭: 연면적과 승인일자 모두 일치 (안전한 비교 사용)
    cond1 = []
    for i in range(len(group)):
        area_match = safe_equals(group['연면적'].iloc[i], group['TOTAREA'].iloc[i])
        date_match = safe_equals(group['사용승인연도'].iloc[i], group['USE_DATE'].iloc[i])
        cond1.append(area_match and date_match)
    
    cond1 = pd.Series(cond1, index=group.index)
    print(f"1차 매칭 조건(연면적+승인일자 일치) 만족 여부: {cond1.any()}")
    
    if cond1.any():
        candidate = group[cond1].iloc[0]
        candidate['MATCH_STAGE'] = '1차'
        return candidate
    
    # 2차 매칭: 연면적만 일치 (안전한 비교 사용)
    cond2 = []
    for i in range(len(group)):
        area_match = safe_equals(group['연면적'].iloc[i], group['TOTAREA'].iloc[i])
        cond2.append(area_match)
    
    cond2 = pd.Series(cond2, index=group.index)
    print(f"2차 매칭 조건(연면적만 일치) 만족 여부: {cond2.any()}")
    
    if cond2.any():
        candidate = group[cond2].iloc[0]
        candidate['MATCH_STAGE'] = '2차'
        return candidate
    
    # 3차 매칭: 위 두 조건에 해당하는 후보가 없고, 원본 EBD와 BD에서 해당 RECAP의 각 데이터가 단 1건씩인 경우
    if bd_counts.get(recap, 0) == 1 and ebd_counts.get(recap, 0) == 1:
        candidate = group.iloc[0]
        candidate['MATCH_STAGE'] = '3차'
        print(f"3차 매칭 선택: RECAP_PK {recap}의 BD와 EBD가 각각 1건씩 존재")
        return candidate
    
    # 어느 조건에도 해당하지 않으면 미매칭 처리:
    candidate = group.iloc[0].copy()
    candidate['MATCH_STAGE'] = '미매칭'
    print(f"미매칭: 모든 조건 불만족. BD 건수: {bd_counts.get(recap, 0)}, EBD 건수: {ebd_counts.get(recap, 0)}")
    
    # 미매칭인 경우 BD 정보는 붙지 않도록 함
    candidate['MGM_BLD_PK'] = None
    candidate['TOTAREA'] = None
    candidate['USE_DATE'] = None
    candidate['DONG_NM'] = None
    candidate['BLD_NM'] = None
    return candidate

def rule_based_matching(ebd_df, bd_df):
    """
    규칙 기반으로 EBD와 BD 데이터 매칭
    
    인자:
        ebd_df: EBD 데이터프레임 (컬럼: 'RECAP_PK', '연면적', '사용승인연도', ...)
        bd_df: BD 데이터프레임 (컬럼: 'RECAP_PK', 'TOTAREA', 'USE_DATE', 'MGM_BLD_PK', ...)
        
    반환:
        matched_df: 매칭 결과 데이터프레임
    """
    # 데이터 전처리: 연면적 숫자 변환
    ebd_df = parse_area_values(ebd_df, '연면적')
    bd_df = parse_area_values(bd_df, 'TOTAREA')
    
    # 날짜 데이터 표준화
    ebd_df = standardize_date(ebd_df, '사용승인연도')
    bd_df = standardize_date(bd_df, 'USE_DATE')
    
    # 변환 후 데이터 샘플 출력 (디버깅용)
    print("\n=== EBD 데이터 샘플 (변환 후) ===")
    print(ebd_df[['연면적', '연면적_원본', '사용승인연도', '사용승인연도_원본']].head())
    
    print("\n=== BD 데이터 샘플 (변환 후) ===")
    print(bd_df[['TOTAREA', 'TOTAREA_원본', 'USE_DATE', 'USE_DATE_원본']].head())
    
    # 중복 데이터 체크 및 로깅
    ebd_dupes = ebd_df[ebd_df.duplicated(subset=['RECAP_PK'], keep=False)]
    if not ebd_dupes.empty:
        print(f"\nEBD 데이터에 중복된 RECAP_PK가 있습니다. 중복 건수: {len(ebd_dupes)}")
        print(ebd_dupes[['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도']].head())
    
    bd_dupes = bd_df[bd_df.duplicated(subset=['RECAP_PK'], keep=False)]
    if not bd_dupes.empty:
        print(f"\nBD 데이터에 중복된 RECAP_PK가 있습니다. 중복 건수: {len(bd_dupes)}")
        print(bd_dupes[['MGM_BLD_PK', 'RECAP_PK', 'TOTAREA', 'USE_DATE']].head())
    
    # 원본 BD 데이터에서 RECAP별 BD 건수 구하기
    bd_counts = bd_df.groupby('RECAP_PK').size().to_dict()
    
    # 원본 EBD 데이터에서 RECAP별 EBD 건수 구하기
    ebd_counts = ebd_df.groupby('RECAP_PK').size().to_dict()
    
    # 각 RECAP별 EBD > BD 여부를 미리 계산
    ebd_over_bd = {}
    for recap in set(ebd_counts.keys()) | set(bd_counts.keys()):
        ebd_count = ebd_counts.get(recap, 0)
        bd_count = bd_counts.get(recap, 0)
        ebd_over_bd[recap] = 'yes' if ebd_count > bd_count else 'no'

    # RECAP_PK를 기준으로 병합 (many-to-many join)
    merged = pd.merge(ebd_df, bd_df, on='RECAP_PK', how='left', suffixes=('_ebd', '_bd'))
    
    # EBD와 BD 건수 비교 정보 추가
    for recap in merged['RECAP_PK'].unique():
        merged.loc[merged['RECAP_PK'] == recap, 'EBD_COUNT'] = ebd_counts.get(recap, 0)
        merged.loc[merged['RECAP_PK'] == recap, 'BD_COUNT'] = bd_counts.get(recap, 0)
        merged.loc[merged['RECAP_PK'] == recap, 'EBD_OVER_BD'] = ebd_over_bd.get(recap, 'no')
    
    # 각 EBD 레코드별 매칭 후보 선택
    print("\n=== 매칭 진행 ===")
    matched_candidates = merged.groupby('SEQ_NO').apply(
        lambda group: match_candidate(group, bd_counts, ebd_counts)
    ).reset_index(drop=True)
    
    # 결과 반환
    return matched_candidates

def get_matching_stats(matched_df):
    """
    매칭 결과에 대한 통계 출력
    """
    stage_counts = matched_df['MATCH_STAGE'].value_counts()
    total_count = len(matched_df)
    
    print("각 매칭 단계별 건수:")
    for stage, count in stage_counts.items():
        print(f"- {stage}: {count}건 ({count/total_count*100:.1f}%)")
    
    print(f"\n총 EBD 건수: {total_count}건")
    print(f"매칭된 건수: {total_count - stage_counts.get('미매칭', 0)}건")
    print(f"매칭율: {(total_count - stage_counts.get('미매칭', 0))/total_count*100:.1f}%")
    
    # EBD > BD 케이스에 대한 통계 추가
    ebd_over_bd_count = matched_df[matched_df['EBD_OVER_BD'] == 'yes'].shape[0]
    print(f"\nEBD가 BD보다 많은 RECAP 케이스: {ebd_over_bd_count}건 ({ebd_over_bd_count/total_count*100:.1f}%)")
    
    return stage_counts

def main():
    # 데이터 로드
    ebd_df = pd.read_excel("./data/EBD_new_3.xlsx")
    bd_df = pd.read_excel("./data/BD_data_all.xlsx")
    
    # 매칭 실행
    matched_results = rule_based_matching(ebd_df, bd_df)
    
    # 통계 출력
    stats = get_matching_stats(matched_results)
    
    # 필요한 컬럼만 선택
    essential_columns = ['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도','기관명', '건축물명', '주소','지상','지하',
                       'TOTAREA','BLD_NM','DONG_NM','USE_DATE', 'MGM_BLD_PK', 'MATCH_STAGE', 
                       'EBD_COUNT', 'BD_COUNT', 'EBD_OVER_BD']
    
    # 모든 컬럼이 존재하는지 확인
    existing_columns = [col for col in essential_columns if col in matched_results.columns]
    summary_results = matched_results[existing_columns] 
    
    # 결과 저장 (필요한 컬럼만 저장)
    os.makedirs("./result", exist_ok=True) 
    summary_results.to_excel("./result/rule_matching_result_ver5.xlsx", index=False)
    print("\n결과가 './result/rule_matching_result_ver5.xlsx'에 저장되었습니다.")
    
    # 모든 컬럼이 포함된 원본 결과도 필요한 경우 별도 저장
    # matched_results.to_excel("./result/rule_based_matching_result_full_ver1.xlsx", index=False)
    # print("모든 컬럼이 포함된 전체 결과는 './result/rule_based_matching_result_full_ver1.xlsx'에 저장되었습니다.")
    
    return summary_results

if __name__ == "__main__":
    main()
