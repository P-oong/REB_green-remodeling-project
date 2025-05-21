#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
건축HUB API 매칭 메인 스크립트
그린리모델링 대상 건축물 데이터와 건축HUB API 데이터를 수집하고 매칭합니다.
"""

import os
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from datetime import datetime
import sys
import time

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 자체 모듈 임포트
from src.api.api_client import BuildingLedgerClient
from src.matching.matcher import BuildingMatcher

# 로깅 설정
def setup_logger():
    """로깅 설정"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_target_buildings(file_path):
    """
    그린리모델링 대상 건축물 데이터 로드
    
    Args:
        file_path (str): 데이터 파일 경로
        
    Returns:
        pandas.DataFrame: 로드된 데이터프레임
    """
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        return pd.read_csv(file_path, encoding='utf-8')
    elif ext.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"지원되지 않는 파일 형식입니다: {ext}")

def prepare_api_request_list(gr_df):
    """
    API 요청을 위한 건물 목록 준비
    
    Args:
        gr_df (pandas.DataFrame): 그린리모델링 대상 건축물 데이터프레임
        
    Returns:
        list: API 요청 파라미터가 담긴 딕셔너리 목록
    """
    # 필요한 컬럼 존재 여부 확인
    required_cols = ['sigungu_code', 'bdong_code', 'bun']
    missing_cols = [col for col in required_cols if col not in gr_df.columns]
    
    if missing_cols:
        # 열 이름이 다를 경우 매핑
        col_mapping = {
            'sigungu_cd': 'sigungu_code',
            'bjdong_cd': 'bdong_code',
            'plat_gb_cd': 'plat_gb_cd',
            'plat_no': 'bun',
            'ji_no': 'ji'
        }
        
        # 데이터프레임 컬럼 이름 변환
        for old_col, new_col in col_mapping.items():
            if old_col in gr_df.columns and new_col not in gr_df.columns:
                gr_df[new_col] = gr_df[old_col]
    
    # 필요한 컬럼 다시 확인
    missing_cols = [col for col in required_cols if col not in gr_df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing_cols)}")
    
    # API 요청 목록 생성
    request_list = []
    
    for _, row in gr_df.iterrows():
        request_item = {
            'sigungu_code': str(row['sigungu_code']),
            'bdong_code': str(row['bdong_code']),
            'bun': str(row['bun'])
        }
        
        # 지번이 있는 경우 추가
        if 'ji' in gr_df.columns and not pd.isna(row['ji']):
            request_item['ji'] = str(row['ji'])
        
        request_list.append(request_item)
    
    return request_list

def run_matching_process(target_file, output_dir, api_key=None, ledger_types=None, delay=1.0):
    """
    전체 매칭 프로세스 실행
    
    Args:
        target_file (str): 그린리모델링 대상 건축물 데이터 파일 경로
        output_dir (str): 결과 저장 디렉토리
        api_key (str, optional): API 키 (None인 경우 .env 파일에서 로드)
        ledger_types (list, optional): 처리할 건축물대장 유형 목록
        delay (float, optional): API 요청 간 지연 시간(초)
    """
    # 로거 설정
    logger = setup_logger()
    logger.info("매칭 프로세스 시작")
    
    # 프로젝트 루트 디렉토리의 .env 파일 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
    module_dir = os.path.dirname(current_dir)  # 04_건축HUB_API_matching 디렉토리
    project_root = os.path.dirname(module_dir)  # 프로젝트 루트 디렉토리
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 그린리모델링 대상 건축물 데이터 로드
        logger.info(f"대상 건축물 데이터 로드 중: {target_file}")
        gr_buildings = load_target_buildings(target_file)
        logger.info(f"로드된 건축물 데이터: {len(gr_buildings)}개")
        
        # API 요청 목록 준비
        request_list = prepare_api_request_list(gr_buildings)
        logger.info(f"API 요청 목록 생성 완료: {len(request_list)}개")
        
        # API 클라이언트 초기화
        api_client = BuildingLedgerClient(api_key)
        
        # 처리할 건축물대장 유형 결정
        if not ledger_types:
            ledger_types = ["표제부", "총괄표제부", "기본개요"]
        
        # 데이터 수집
        collected_data = {}
        
        for ledger_type in ledger_types:
            logger.info(f"{ledger_type} 데이터 수집 시작")
            start_time = time.time()
            
            collected_df = api_client.batch_get_building_data(
                ledger_type=ledger_type,
                target_buildings=request_list,
                delay=delay
            )
            
            # 수집 결과 처리
            collected_data[ledger_type] = collected_df
            
            # 중간 결과 저장
            interim_output_file = os.path.join(output_dir, f"collected_{ledger_type}.csv")
            collected_df.to_csv(interim_output_file, index=False, encoding='utf-8-sig')
            
            elapsed_time = time.time() - start_time
            logger.info(f"{ledger_type} 데이터 수집 완료: {len(collected_df)}개 항목, 소요 시간: {elapsed_time:.2f}초")
        
        # 매칭 프로세스 시작
        logger.info("건축물 데이터 매칭 시작")
        matcher = BuildingMatcher()
        
        # 각 유형별로 매칭 결과 처리
        for ledger_type, hub_data in collected_data.items():
            if hub_data.empty:
                logger.warning(f"{ledger_type} 데이터가 없습니다. 매칭을 건너뜁니다.")
                continue
            
            logger.info(f"{ledger_type} 데이터 매칭 중...")
            
            # 주소 컬럼 확인 및 생성
            # 예시: 시군구명 + 법정동명 + 번 + 지 조합
            if 'address' not in hub_data.columns:
                address_components = []
                
                if '시군구명' in hub_data.columns:
                    address_components.append(hub_data['시군구명'])
                elif 'sigungu_nm' in hub_data.columns:
                    address_components.append(hub_data['sigungu_nm'])
                
                if '법정동명' in hub_data.columns:
                    address_components.append(hub_data['법정동명'])
                elif 'bjdong_nm' in hub_data.columns:
                    address_components.append(hub_data['bjdong_nm'])
                
                if '번' in hub_data.columns:
                    address_components.append(hub_data['번'].astype(str))
                elif 'bun' in hub_data.columns:
                    address_components.append(hub_data['bun'].astype(str))
                
                if '지' in hub_data.columns:
                    address_components.append(hub_data['지'].astype(str))
                elif 'ji' in hub_data.columns:
                    address_components.append(hub_data['ji'].astype(str))
                
                if address_components:
                    hub_data['address'] = pd.concat(address_components, axis=1).apply(
                        lambda x: ' '.join(x.dropna().astype(str)), axis=1
                    )
            
            # 매칭 수행
            exact_matches = matcher.match_by_exact_address(gr_buildings, hub_data)
            fuzzy_matches = matcher.match_by_fuzzy_address(gr_buildings, hub_data, threshold=80)
            
            # 좌표 데이터가 있는 경우에만 좌표 매칭 수행
            if all(col in gr_buildings.columns for col in ['latitude', 'longitude']) and \
               all(col in hub_data.columns for col in ['latitude', 'longitude']):
                coord_matches = matcher.match_by_coordinates(gr_buildings, hub_data, distance_threshold=100)
            else:
                # 좌표 정보가 없는 경우 빈 데이터프레임으로 설정
                coord_matches = gr_buildings.copy()
                coord_matches['is_coord_matched'] = False
            
            # 매칭 결과 통합
            combined_results = matcher.combine_matching_results(
                exact_matches, fuzzy_matches, coord_matches
            )
            
            # 매칭 결과 저장
            output_file = os.path.join(output_dir, f"matched_{ledger_type}.csv")
            combined_results.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"{ledger_type} 매칭 결과 저장 완료: {output_file}")
            logger.info(f"매칭된 건물 수: {combined_results['is_matched'].sum()} / {len(combined_results)}")
            
            # 매칭 방법별 통계
            if 'matching_method' in combined_results.columns:
                method_counts = combined_results['matching_method'].value_counts()
                for method, count in method_counts.items():
                    logger.info(f"  - {method} 매칭: {count}개")
        
        # 모든 매칭 데이터 통합 (선택 사항)
        logger.info("매칭 프로세스 완료")
        
    except Exception as e:
        logger.error(f"매칭 프로세스 중 오류 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="건축HUB API 매칭 프로그램")
    
    parser.add_argument(
        "target_file",
        help="그린리모델링 대상 건축물 데이터 파일 경로 (.csv 또는 .xlsx)"
    )
    
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
        help="결과 저장 디렉토리 (기본값: ./results)"
    )
    
    parser.add_argument(
        "--api_key",
        help="API 키 (기본값: 프로젝트 루트 디렉토리의 .env 파일의 BUILDING_HUB_API_KEY 환경변수)"
    )
    
    parser.add_argument(
        "--ledger_types",
        nargs="+",
        choices=["기본개요", "총괄표제부", "표제부", "층별개요", "부속지번", "전유공용면적", "오수정화시설", "주택가격", "전유부", "지역지구구역", "소유자"],
        default=["표제부", "총괄표제부", "기본개요"],
        help="처리할 건축물대장 유형 (기본값: 표제부, 총괄표제부, 기본개요)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="API 요청 간 지연 시간(초) (기본값: 1.0)"
    )
    
    args = parser.parse_args()
    
    run_matching_process(
        target_file=args.target_file,
        output_dir=args.output_dir,
        api_key=args.api_key,
        ledger_types=args.ledger_types,
        delay=args.delay
    ) 