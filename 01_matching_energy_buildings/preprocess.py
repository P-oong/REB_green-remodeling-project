# preprocess.py
# 10차 매칭(1~9차 후 RECAP_PK별로 EBD·BD가 각각 1건만 남은 경우)은 batch_matcher.py에서 처리됨. 이 모듈은 전처리만 담당.

import pandas as pd
import re
from datetime import datetime

def tokenize_text(text):
    if pd.isna(text):
        return set()
    # 소문자화
    text = str(text).lower()
    # 괄호를 공백으로 치환
    text = re.sub(r'[\(\)\[\]\{\}]', ' ', text)
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 연속된 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text)
    # 띄어쓰기 기준으로 분리
    tokens = set(text.split())
    # 빈 문자열 제거
    tokens = {token for token in tokens if token}
    return tokens

def tokenize_ebd_text(building_name, address):
    name_tokens = tokenize_text(building_name)
    addr_tokens = list(tokenize_text(address))
    addr_tokens = set(addr_tokens[3:]) if len(addr_tokens) > 3 else set()
    return name_tokens.union(addr_tokens)

def preprocess_area(area):
    if pd.isna(area):
        return None
    try:
        # 문자열인 경우 숫자만 추출
        if isinstance(area, str):
            area = re.sub(r'[^\d.]', '', area)
        # 숫자로 변환
        area = pd.to_numeric(area, errors='coerce')
        return float(area) if not pd.isna(area) else None
    except:
        return None

def preprocess_year(year):
    if pd.isna(year):
        return None
    try:
        # 날짜 형식으로 변환 시도
        if isinstance(year, str):
            # YYYYMM 형식인 경우 YYYY만 추출
            if len(year) == 6 and year.isdigit():
                year = year[:4]
            # 날짜 형식으로 변환 시도
            try:
                year = pd.to_datetime(year, errors='coerce')
                return year.year if not pd.isna(year) else None
            except:
                # 숫자만 있는 경우 (연도만 있는 경우) 숫자로 변환
                year = pd.to_numeric(year, errors='coerce')
                return int(year) if not pd.isna(year) else None
        # 이미 datetime인 경우
        elif pd.api.types.is_datetime64_dtype(year):
            return year.year
        # 이미 숫자인 경우
        else:
            year = pd.to_numeric(year, errors='coerce')
            return int(year) if not pd.isna(year) else None
    except:
        return None

def preprocess_data(ebd_df, bd_df):
    # EBD 토큰 생성
    ebd_df = ebd_df.copy()
    ebd_df['ebd_tokens'] = ebd_df['건축물명'].apply(tokenize_text)
    
    # EBD 연면적 전처리
    if '연면적' in ebd_df.columns:
        ebd_df['연면적_원본'] = ebd_df['연면적']
        ebd_df['연면적'] = pd.to_numeric(ebd_df['연면적'], errors='coerce')
    
    # EBD 사용승인연도 전처리
    if '사용승인연도' in ebd_df.columns:
        ebd_df['사용승인연도_원본'] = ebd_df['사용승인연도']
        try:
            # 날짜 형식으로 변환 시도
            ebd_df['사용승인연도'] = pd.to_datetime(ebd_df['사용승인연도'], errors='coerce')
        except:
            # YYYYMM 형식인 경우 YYYY-MM-01로 변환
            ebd_df['사용승인연도'] = pd.to_datetime(ebd_df['사용승인연도'].astype(str).str[:6] + '01', format='%Y%m%d', errors='coerce')
    
    # BD 토큰 생성
    bd_df = bd_df.copy()
    bd_df['dong_tokens'] = bd_df['DONG_NM'].apply(tokenize_text)
    bd_df['bld_tokens'] = bd_df['BLD_NM'].apply(tokenize_text)
    
    # BD 연면적 전처리
    if 'TOTAREA' in bd_df.columns:
        bd_df['TOTAREA_원본'] = bd_df['TOTAREA']
        bd_df['TOTAREA'] = pd.to_numeric(bd_df['TOTAREA'], errors='coerce')
    
    # BD 사용승인연도 전처리
    if 'USE_DATE' in bd_df.columns:
        bd_df['USE_DATE_원본'] = bd_df['USE_DATE']
        try:
            # 날짜 형식으로 변환 시도
            bd_df['USE_DATE'] = pd.to_datetime(bd_df['USE_DATE'], errors='coerce')
        except:
            # YYYYMM 형식인 경우 YYYY-MM-01로 변환
            bd_df['USE_DATE'] = pd.to_datetime(bd_df['USE_DATE'].astype(str).str[:6] + '01', format='%Y%m%d', errors='coerce')
    
    return ebd_df, bd_df 