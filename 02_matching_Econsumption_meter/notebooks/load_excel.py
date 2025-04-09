import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_excel_file(file_path, sheet_name=0, header=2):
    """
    엑셀 파일을 로드하는 함수
    
    Args:
        file_path (str): 엑셀 파일 경로
        sheet_name (str or int): 시트 이름 또는 인덱스 (기본값: 0, 첫 번째 시트)
        header (int): 헤더 행 번호 (기본값: 2, 3번째 행이 헤더)
    
    Returns:
        pandas.DataFrame: 로드된 데이터프레임
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        print(f"'{file_path}' 파일이 성공적으로 로드되었습니다.")
        print(f"데이터 크기: {df.shape}")
        return df
    except Exception as e:
        print(f"파일 로딩 오류: {str(e)}")
        return None

def display_data_info(df):
    """
    데이터프레임 기본 정보 출력
    
    Args:
        df (pandas.DataFrame): 데이터프레임
    """
    if df is None:
        print("데이터프레임이 존재하지 않습니다.")
        return
    
    print("\n=== 데이터프레임 기본 정보 ===")
    print("\n컬럼명:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\n데이터 타입:")
    print(df.dtypes)
    
    print("\n처음 5개 행:")
    print(df.head())
    
    print("\nNaN 값이 있는 열:")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])

def load_multiple_sheets(file_path, sheet_names=None, header=2):
    """
    엑셀 파일의 여러 시트를 로드하는 함수
    
    Args:
        file_path (str): 엑셀 파일 경로
        sheet_names (list): 로드할 시트 이름 목록 (기본값: None, 모든 시트 로드)
        header (int): 헤더 행 번호 (기본값: 2, 3번째 행이 헤더)
    
    Returns:
        dict: {시트 이름: 데이터프레임} 형태의 딕셔너리
    """
    try:
        if sheet_names is None:
            # 모든 시트 이름 가져오기
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
        
        # 각 시트를 로드하여 딕셔너리에 저장
        sheet_data = {}
        for sheet in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet, header=header)
            sheet_data[sheet] = df
            print(f"시트 '{sheet}' 로드 완료: {df.shape}행 x {df.shape[1]}열")
        
        return sheet_data
    except Exception as e:
        print(f"파일 로딩 오류: {str(e)}")
        return None

def combine_quarterly_data(file_path, sheet_names, energy_cols, header=2):
    """
    분기별 시트의 에너지 데이터를 합치는 함수
    
    Args:
        file_path (str): 엑셀 파일 경로
        sheet_names (list): 분기별 시트 이름 목록
        energy_cols (list): 합산할 에너지 변수 컬럼명 목록
        header (int): 헤더 행 번호 (기본값: 2, 3번째 행이 헤더)
    
    Returns:
        pandas.DataFrame: 건축물ID별 에너지 변수의 분기별 및 합산 데이터
    """
    # 각 시트 데이터 로드
    quarterly_data = {}
    for sheet in sheet_names:
        quarterly_data[sheet] = pd.read_excel(file_path, sheet_name=sheet, header=header)
    
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
    
    return result_df

if __name__ == "__main__":
    # 파일 경로 설정
    file_path = '../data/2023년_에너지소비량_보고 건축물_20250331.xlsx'
    
    # 기본 로드 예제
    print("\n=== 기본 로드 예제 ===")
    df = load_excel_file(file_path, header=2)
    display_data_info(df)
    
    # 분기별 시트 로드 예제
    print("\n=== 분기별 시트 로드 예제 ===")
    sheet_names = ['1-1_2023년_1분기', '1-2_2023년_2분기', '1-3_2023년_3분기', '1-4_2023년_4분기']
    quarterly_sheets = load_multiple_sheets(file_path, sheet_names, header=2)
    
    # 에너지 변수별 분기별 데이터 합산 예제
    print("\n=== 에너지 변수별 분기별 데이터 합산 예제 ===")
    energy_cols = ['전기 (KWH)', '가스 (KWH)', '지역냉난방 (KWH)', '유류 (KWH)', '기타 (KWH)']
    
    # 분기별 데이터 합산
    if quarterly_sheets:
        energy_summary = combine_quarterly_data(file_path, sheet_names, energy_cols, header=2)
        print(f"에너지 요약 데이터 크기: {energy_summary.shape}")
        print(energy_summary.head()) 