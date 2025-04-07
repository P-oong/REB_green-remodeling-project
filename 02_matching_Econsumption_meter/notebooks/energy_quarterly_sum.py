import pandas as pd
import numpy as np

# 파일 경로 설정
file_path = '../data/2023년_에너지소비량_보고 건축물_20250331.xlsx'

# 시트 이름 리스트
sheet_names = ['1-1_2023년_1분기', '1-2_2023년_2분기', '1-3_2023년_3분기', '1-4_2023년_4분기']

# 에너지 변수 컬럼명
energy_cols = ['전기 (KWH)', '가스 (KWH)', '지역냉난방 (KWH)', '유류 (KWH)', '기타 (KWH)']

# 각 분기 데이터 로드
quarterly_data = {}
for sheet in sheet_names:
    quarterly_data[sheet] = pd.read_excel(file_path, sheet_name=sheet, header=2)

# 결과를 저장할 빈 데이터프레임 생성
result_df = pd.DataFrame()
result_df['건축물ID'] = quarterly_data[sheet_names[0]]['건축물ID']

# 각 에너지 변수별로 1~4분기 합계 및 개별 분기 데이터 추가
for energy in energy_cols:
    # 1~4분기 합계 계산
    total = sum(quarterly_data[sheet][energy] for sheet in sheet_names)
    result_df[f'{energy}_연간합계'] = total
    
    # 각 분기별 데이터 추가
    for i, sheet in enumerate(sheet_names, 1):
        result_df[f'{energy}_{i}분기'] = quarterly_data[sheet][energy]

# 데이터 확인
print("데이터 크기:", result_df.shape)
print("\n처음 5개 행:")
print(result_df.head())

# 결과 저장
result_df.to_excel('../result/energy_quarterly_summary.xlsx', index=False)
print("\n결과가 'energy_quarterly_summary.xlsx' 파일로 저장되었습니다.")