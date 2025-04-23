#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
국토교통부 건축HUB 건축물대장정보 서비스 API 클라이언트
"""

import os
import pandas as pd
from PublicDataReader import BuildingLedger
from dotenv import load_dotenv
import time
from tqdm import tqdm

class BuildingLedgerClient:
    """
    건축물대장정보 API 클라이언트 클래스
    PublicDataReader 패키지를 활용하여 건축물대장 정보를 조회합니다.
    """
    
    def __init__(self, service_key=None):
        """
        건축물대장정보 API 클라이언트 초기화
        
        Args:
            service_key (str, optional): 공공데이터포털에서 발급받은 서비스키. 
                                        None인 경우 환경변수(BUILDING_HUB_API_KEY)에서 로드
        """
        # 프로젝트 루트 디렉토리의 .env 파일 경로 찾기
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
        src_dir = os.path.dirname(current_dir)  # src 디렉토리
        module_dir = os.path.dirname(src_dir)  # 04_건축HUB_API_matching 디렉토리
        project_root = os.path.dirname(module_dir)  # 프로젝트 루트 디렉토리
        
        # 프로젝트 루트 디렉토리의 .env 파일 로드
        dotenv_path = os.path.join(project_root, '.env')
        load_dotenv(dotenv_path)
        
        # 서비스키 설정
        self.service_key = service_key or os.getenv('BUILDING_HUB_API_KEY')
        if not self.service_key:
            raise ValueError("API 키가 제공되지 않았습니다. 프로젝트 루트 디렉토리의 .env 파일에 BUILDING_HUB_API_KEY를 설정하거나 서비스키를 직접 제공하세요.")
        
        # API 클라이언트 초기화
        self.api = BuildingLedger(self.service_key)
        
        # 유효한 건축물대장 유형 목록
        self.valid_ledger_types = [
            "기본개요", "총괄표제부", "표제부", "층별개요", 
            "부속지번", "전유공용면적", "오수정화시설", 
            "주택가격", "전유부", "지역지구구역", "소유자"
        ]
    
    def get_building_data(self, ledger_type, sigungu_code, bdong_code, bun, ji=""):
        """
        건축물대장 정보 조회
        
        Args:
            ledger_type (str): 건축물대장 유형
            sigungu_code (str): 시군구코드
            bdong_code (str): 법정동코드
            bun (str): 번
            ji (str, optional): 지
            
        Returns:
            pandas.DataFrame: 조회 결과 데이터프레임
        """
        if ledger_type not in self.valid_ledger_types:
            raise ValueError(f"유효하지 않은 건축물대장 유형입니다. 유효한 유형: {', '.join(self.valid_ledger_types)}")
        
        try:
            df = self.api.get_data(
                ledger_type=ledger_type,
                sigungu_code=sigungu_code,
                bdong_code=bdong_code,
                bun=bun,
                ji=ji
            )
            return df
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def batch_get_building_data(self, ledger_type, target_buildings, delay=1):
        """
        여러 건축물에 대한 건축물대장 정보를 일괄 조회
        
        Args:
            ledger_type (str): 건축물대장 유형
            target_buildings (list): 조회할 건축물 목록 
                                    [{'sigungu_code': '...', 'bdong_code': '...', 'bun': '...', 'ji': '...'}]
            delay (float, optional): API 호출 간 지연 시간(초)
            
        Returns:
            pandas.DataFrame: 조회 결과를 통합한 데이터프레임
        """
        results = []
        
        for building in tqdm(target_buildings, desc=f"{ledger_type} 데이터 수집 중"):
            try:
                df = self.get_building_data(
                    ledger_type=ledger_type,
                    sigungu_code=building['sigungu_code'],
                    bdong_code=building['bdong_code'],
                    bun=building['bun'],
                    ji=building.get('ji', '')
                )
                
                if not df.empty:
                    # 조회 키를 데이터프레임에 추가
                    for key, value in building.items():
                        df[f'request_{key}'] = value
                    
                    results.append(df)
                
                # API 호출 간 지연
                time.sleep(delay)
                
            except Exception as e:
                print(f"건물 정보 조회 중 오류 발생: {building}, 오류: {e}")
        
        # 결과 통합
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_all_ledger_types(self, target_buildings, delay=1):
        """
        모든 건축물대장 유형에 대한 정보를 일괄 조회
        
        Args:
            target_buildings (list): 조회할 건축물 목록
            delay (float, optional): API 호출 간 지연 시간(초)
            
        Returns:
            dict: 건축물대장 유형별 조회 결과 데이터프레임 사전
        """
        results = {}
        
        for ledger_type in self.valid_ledger_types:
            print(f"\n{ledger_type} 데이터 수집 시작...")
            df = self.batch_get_building_data(ledger_type, target_buildings, delay)
            results[ledger_type] = df
            print(f"{ledger_type} 데이터 수집 완료: {len(df)}개 항목")
        
        return results


# 사용 예시
if __name__ == "__main__":
    # 클라이언트 인스턴스 생성
    client = BuildingLedgerClient()
    
    # 단일 건축물 정보 조회 예시
    sigungu_code = "41135"  # 성남시 분당구
    bdong_code = "11000"    # 백현동
    bun = "542"
    ji = ""
    
    df = client.get_building_data(
        ledger_type="표제부",
        sigungu_code=sigungu_code,
        bdong_code=bdong_code,
        bun=bun,
        ji=ji
    )
    
    print(df.head()) 