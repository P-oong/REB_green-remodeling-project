import os
import pandas as pd

def parse_area_values(df, area_column):
    """
    연면적 컬럼의 값을 숫자로 변환
    """
    df = df.copy()
    if area_column in df.columns:
        df[area_column] = pd.to_numeric(df[area_column], errors='coerce')
    return df

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
    
    # 1차 매칭: 연면적과 승인일자 모두 일치
    cond1 = (group['연면적'] == group['TOTAREA']) & (group['사용승인연도'] == group['USE_DATE'])
    if cond1.any():
        candidate = group[cond1].iloc[0]
        candidate['MATCH_STAGE'] = '1차'
        return candidate
    
    # 2차 매칭: 연면적만 일치 (승인일자는 불일치)
    cond2 = (group['연면적'] == group['TOTAREA'])
    if cond2.any():
        candidate = group[cond2].iloc[0]
        candidate['MATCH_STAGE'] = '2차'
        return candidate
    
    # 3차 매칭: 위 두 조건에 해당하는 후보가 없고, 원본 EBD와 BD에서 해당 RECAP의 각 데이터가 단 1건씩인 경우
    recap = group['RECAP_PK'].iloc[0]
    if bd_counts.get(recap, 0) == 1 and ebd_counts.get(recap, 0) == 1:
        candidate = group.iloc[0]
        candidate['MATCH_STAGE'] = '3차'
        return candidate
    
    # 어느 조건에도 해당하지 않으면 미매칭 처리:
    candidate = group.iloc[0].copy()
    candidate['MATCH_STAGE'] = '미매칭'
    candidate['MGM_BLD_PK'] = None  # 미매칭인 경우 BD 정보는 붙지 않도록 함
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
    summary_results.to_excel("./result/rule_matching_result_ver3.xlsx", index=False)
    print("\n결과가 './result/rule_matching_result_ver3.xlsx'에 저장되었습니다.")
    
    # 모든 컬럼이 포함된 원본 결과도 필요한 경우 별도 저장
    # matched_results.to_excel("./result/rule_based_matching_result_full_ver1.xlsx", index=False)
    # print("모든 컬럼이 포함된 전체 결과는 './result/rule_based_matching_result_full_ver1.xlsx'에 저장되었습니다.")
    
    return summary_results

if __name__ == "__main__":
    main()
