import pandas as pd
import numpy as np

# 파일 경로 설정
quarterly_sum_path = '../result/energy_quarterly_summary.xlsx'
raw_data_path = '../result/raw23_energy_data.xlsx'

# 데이터 로드
quarterly_df = pd.read_excel(quarterly_sum_path)
raw_df = pd.read_excel(raw_data_path)

# 비교할 컬럼 쌍 정의
column_pairs = {
    '전기 (KWH)_연간합계': '단위면적당 1차 E사용량_전기 (KWH)',
    '가스 (KWH)_연간합계': '단위면적당 1차 E사용량_가스 (KWH)',
    '지역냉난방 (KWH)_연간합계': '단위면적당 1차 E사용량_지역냉난방 (KWH)',
    '유류 (KWH)_연간합계': '단위면적당 1차 E사용량_유류 (KWH)',
    '기타 (KWH)_연간합계': '단위면적당 1차 E사용량_기타 (KWH)'
}

# 결과를 저장할 데이터프레임 생성
comparison_results = []

# 각 에너지 종류별로 비교
for quarterly_col, raw_col in column_pairs.items():
    # 두 데이터프레임 병합
    merged_df = pd.merge(
        quarterly_df[['건축물ID', quarterly_col]], 
        raw_df[['건축물ID', raw_col]], 
        on='건축물ID', 
        how='outer',
        suffixes=('_quarterly', '_raw')
    )
    
    # 차이 계산 (소수점 둘째자리까지 반올림)
    merged_df['차이'] = (merged_df[quarterly_col] - merged_df[raw_col]).round(2)
    
    # 차이가 있는 데이터만 필터링 (절대값 0.01 이상인 경우)
    diff_records = merged_df[abs(merged_df['차이']) >= 0.01].copy()
    
    if len(diff_records) > 0:
        # 에너지 종류 컬럼 추가
        diff_records['에너지종류'] = quarterly_col.split('_')[0]
        
        # quarterly 값이 없는 경우
        diff_records['비고'] = '정상'
        diff_records.loc[diff_records[quarterly_col].isna(), '비고'] = 'quarterly 데이터 누락'
        diff_records.loc[diff_records[raw_col].isna(), '비고'] = 'raw 데이터 누락'
        
        # 결과 저장
        comparison_results.append(diff_records[['건축물ID', '에너지종류', quarterly_col, raw_col, '차이', '비고']])

# 모든 결과 합치기
if comparison_results:
    final_results = pd.concat(comparison_results, ignore_index=True)
    
    # 결과 출력
    print("\n=== 데이터 불일치 요약 ===")
    print(f"총 불일치 건수: {len(final_results)}")
    print("\n에너지 종류별 불일치 건수:")
    print(final_results['에너지종류'].value_counts())
    print("\n비고별 건수:")
    print(final_results['비고'].value_counts())
    
    print("\n처음 10개 불일치 데이터:")
    print(final_results.head(10))
    
    # 결과 저장
    final_results.to_excel('../result/energy_data_discrepancy.xlsx', index=False)
    print("\n상세 결과가 'energy_data_discrepancy.xlsx' 파일로 저장되었습니다.")
else:
    print("\n모든 데이터가 일치합니다!")

# 추가적인 통계 정보
print("\n=== 데이터셋 기본 정보 ===")
print(f"quarterly_sum 데이터 건수: {len(quarterly_df)}")
print(f"raw23_energy_data 건수: {len(raw_df)}")

# 건축물ID 비교
quarterly_ids = set(quarterly_df['건축물ID'])
raw_ids = set(raw_df['건축물ID'])

print(f"\n건축물ID 비교:")
print(f"quarterly_sum에만 있는 건축물ID 수: {len(quarterly_ids - raw_ids)}")
print(f"raw23_energy_data에만 있는 건축물ID 수: {len(raw_ids - quarterly_ids)}")
print(f"공통 건축물ID 수: {len(quarterly_ids & raw_ids)}") 