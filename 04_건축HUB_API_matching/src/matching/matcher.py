#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
그린리모델링 대상 건축물과 건축HUB API 데이터 매칭 유틸리티
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import re


class BuildingMatcher:
    """
    건축물 정보 매칭 클래스
    그린리모델링 대상 건축물 데이터와 건축HUB API에서 가져온 데이터를 매칭합니다.
    """
    
    def __init__(self):
        """
        건축물 매칭 클래스 초기화
        """
        pass
    
    def clean_address(self, address):
        """
        주소 정보 정제
        
        Args:
            address (str): 정제할 주소 문자열
            
        Returns:
            str: 정제된 주소 문자열
        """
        if not isinstance(address, str):
            return ""
        
        # 주소 정제 로직
        address = re.sub(r'\s+', ' ', address)  # 여러 공백을 하나로 변경
        address = address.strip()  # 양쪽 공백 제거
        address = re.sub(r'[^\w\s\(\)-]', '', address)  # 특수문자 제거 (단, 괄호, 하이픈 제외)
        
        return address
    
    def match_by_exact_address(self, gr_buildings, hub_buildings):
        """
        정확한 주소 기반 매칭
        
        Args:
            gr_buildings (pd.DataFrame): 그린리모델링 대상 건축물 데이터프레임
            hub_buildings (pd.DataFrame): 건축HUB API에서 가져온 건축물 데이터프레임
            
        Returns:
            pd.DataFrame: 매칭 결과 데이터프레임
        """
        # 주소 정제
        gr_buildings['cleaned_address'] = gr_buildings['address'].apply(self.clean_address)
        hub_buildings['cleaned_address'] = hub_buildings['address'].apply(self.clean_address)
        
        # 정확한 주소 매칭
        merged_df = pd.merge(
            gr_buildings, 
            hub_buildings,
            on='cleaned_address',
            how='left',
            suffixes=('_gr', '_hub')
        )
        
        # 매칭 여부 플래그 추가
        merged_df['is_exact_matched'] = ~merged_df['cleaned_address_hub'].isna()
        
        return merged_df
    
    def match_by_fuzzy_address(self, gr_buildings, hub_buildings, threshold=80):
        """
        퍼지 매칭을 통한 주소 매칭 (유사도 기반)
        
        Args:
            gr_buildings (pd.DataFrame): 그린리모델링 대상 건축물 데이터프레임
            hub_buildings (pd.DataFrame): 건축HUB API에서 가져온 건축물 데이터프레임
            threshold (int, optional): 유사도 임계값 (0-100)
            
        Returns:
            pd.DataFrame: 매칭 결과 데이터프레임
        """
        # 주소 정제
        gr_buildings['cleaned_address'] = gr_buildings['address'].apply(self.clean_address)
        hub_buildings['cleaned_address'] = hub_buildings['address'].apply(self.clean_address)
        
        # 주소 사전 생성 (HUB 데이터)
        hub_addresses = hub_buildings['cleaned_address'].dropna().unique()
        hub_address_dict = {addr: idx for idx, addr in enumerate(hub_addresses)}
        
        # 매칭 결과 저장용 리스트
        matches = []
        
        # 각 그린리모델링 건축물에 대해 퍼지 매칭 수행
        for _, row in gr_buildings.iterrows():
            gr_addr = row['cleaned_address']
            if not gr_addr:
                continue
                
            # 퍼지 매칭 수행
            match_result = process.extractOne(
                gr_addr, 
                hub_addresses, 
                scorer=fuzz.token_sort_ratio
            )
            
            if match_result and match_result[1] >= threshold:
                matched_addr, similarity, _ = match_result
                matched_idx = hub_address_dict[matched_addr]
                
                # 매칭된 HUB 데이터
                hub_data = hub_buildings[hub_buildings['cleaned_address'] == matched_addr]
                
                # 각 매칭 결과 저장
                for _, hub_row in hub_data.iterrows():
                    match = {
                        'gr_id': row['id'],
                        'hub_id': hub_row['id'],
                        'gr_address': row['address'],
                        'hub_address': hub_row['address'],
                        'similarity': similarity,
                        'is_fuzzy_matched': True
                    }
                    matches.append(match)
        
        # 결과 데이터프레임 생성
        if matches:
            result_df = pd.DataFrame(matches)
            
            # 원본 데이터 병합
            result_df = pd.merge(
                gr_buildings,
                result_df,
                left_on='id',
                right_on='gr_id',
                how='left'
            )
            
            # 매칭 여부 플래그 추가
            result_df['is_fuzzy_matched'] = result_df['is_fuzzy_matched'].fillna(False)
            
            return result_df
        else:
            # 매칭 결과가 없는 경우
            gr_buildings['is_fuzzy_matched'] = False
            return gr_buildings
    
    def match_by_coordinates(self, gr_buildings, hub_buildings, distance_threshold=100):
        """
        좌표(위도/경도) 기반 건축물 매칭
        
        Args:
            gr_buildings (pd.DataFrame): 그린리모델링 대상 건축물 데이터프레임 (위도/경도 포함)
            hub_buildings (pd.DataFrame): 건축HUB API에서 가져온 건축물 데이터프레임 (위도/경도 포함)
            distance_threshold (float, optional): 거리 임계값 (미터)
            
        Returns:
            pd.DataFrame: 매칭 결과 데이터프레임
        """
        from scipy.spatial.distance import cdist
        
        # 필요한 컬럼 확인
        required_cols = ['latitude', 'longitude']
        for df_name, df in [('gr_buildings', gr_buildings), ('hub_buildings', hub_buildings)]:
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"{df_name}에 {col} 컬럼이 없습니다.")
        
        # 결측치 처리
        gr_valid = gr_buildings[gr_buildings['latitude'].notna() & gr_buildings['longitude'].notna()]
        hub_valid = hub_buildings[hub_buildings['latitude'].notna() & hub_buildings['longitude'].notna()]
        
        if gr_valid.empty or hub_valid.empty:
            print("유효한 좌표 데이터가 없습니다.")
            return gr_buildings
        
        # 좌표 배열 생성
        gr_coords = gr_valid[['latitude', 'longitude']].values
        hub_coords = hub_valid[['latitude', 'longitude']].values
        
        # 좌표 간 거리 계산 (하버사인 거리 - 미터 단위)
        def haversine_distances(coords1, coords2):
            earth_radius = 6371000  # 지구 반경 (미터)
            
            lat1, lon1 = np.radians(coords1[:, 0:1]), np.radians(coords1[:, 1:2])
            lat2, lon2 = np.radians(coords2[:, 0:1]), np.radians(coords2[:, 1:2])
            
            dlat = lat2.T - lat1
            dlon = lon2.T - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) @ np.cos(lat2.T) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return earth_radius * c
        
        # 거리 행렬 계산
        distances = haversine_distances(gr_coords, hub_coords)
        
        # 매칭 결과 저장용 리스트
        matches = []
        
        # 각 그린리모델링 건축물에 대해 가장 가까운 HUB 건축물 찾기
        for i, gr_idx in enumerate(gr_valid.index):
            min_dist_idx = np.argmin(distances[i])
            min_dist = distances[i, min_dist_idx]
            
            if min_dist <= distance_threshold:
                hub_idx = hub_valid.index[min_dist_idx]
                
                match = {
                    'gr_id': gr_valid.loc[gr_idx, 'id'],
                    'hub_id': hub_valid.loc[hub_idx, 'id'],
                    'distance': min_dist,
                    'is_coord_matched': True
                }
                matches.append(match)
        
        # 결과 데이터프레임 생성
        if matches:
            result_df = pd.DataFrame(matches)
            
            # 원본 데이터 병합
            result_df = pd.merge(
                gr_buildings,
                result_df,
                left_on='id',
                right_on='gr_id',
                how='left'
            )
            
            # 매칭 여부 플래그 추가
            result_df['is_coord_matched'] = result_df['is_coord_matched'].fillna(False)
            
            return result_df
        else:
            # 매칭 결과가 없는 경우
            gr_buildings['is_coord_matched'] = False
            return gr_buildings
    
    def combine_matching_results(self, exact_match_df, fuzzy_match_df, coord_match_df):
        """
        여러 매칭 방법의 결과를 통합
        
        Args:
            exact_match_df (pd.DataFrame): 정확한 주소 매칭 결과
            fuzzy_match_df (pd.DataFrame): 퍼지 매칭 결과
            coord_match_df (pd.DataFrame): 좌표 기반 매칭 결과
            
        Returns:
            pd.DataFrame: 통합된 매칭 결과
        """
        # 기본 ID 컬럼 확인
        id_col = 'id'
        if id_col not in exact_match_df.columns:
            raise ValueError("매칭 결과 데이터프레임에 'id' 컬럼이 없습니다.")
        
        # 정확한 매칭 결과 복사
        result_df = exact_match_df.copy()
        
        # 퍼지 매칭 결과 병합
        if 'is_fuzzy_matched' in fuzzy_match_df.columns:
            fuzzy_matched = fuzzy_match_df[fuzzy_match_df['is_fuzzy_matched']]
            if not fuzzy_matched.empty:
                # 정확한 매칭이 없는 경우만 퍼지 매칭 결과 사용
                unmatched_ids = result_df[~result_df['is_exact_matched']][id_col].values
                fuzzy_to_add = fuzzy_matched[fuzzy_matched[id_col].isin(unmatched_ids)]
                
                # 필요한 컬럼만 선택하여 병합
                fuzzy_cols = [c for c in fuzzy_to_add.columns if c not in result_df.columns or c == id_col]
                if fuzzy_cols:
                    result_df = pd.merge(
                        result_df,
                        fuzzy_to_add[fuzzy_cols],
                        on=id_col,
                        how='left'
                    )
        
        # 좌표 매칭 결과 병합
        if 'is_coord_matched' in coord_match_df.columns:
            coord_matched = coord_match_df[coord_match_df['is_coord_matched']]
            if not coord_matched.empty:
                # 다른 매칭이 없는 경우만 좌표 매칭 결과 사용
                unmatched_ids = result_df[
                    ~result_df['is_exact_matched'] & 
                    ~result_df.get('is_fuzzy_matched', pd.Series(False, index=result_df.index))
                ][id_col].values
                
                coord_to_add = coord_matched[coord_matched[id_col].isin(unmatched_ids)]
                
                # 필요한 컬럼만 선택하여 병합
                coord_cols = [c for c in coord_to_add.columns if c not in result_df.columns or c == id_col]
                if coord_cols:
                    result_df = pd.merge(
                        result_df,
                        coord_to_add[coord_cols],
                        on=id_col,
                        how='left'
                    )
        
        # 최종 매칭 상태 컬럼 추가
        result_df['is_matched'] = (
            result_df.get('is_exact_matched', False) | 
            result_df.get('is_fuzzy_matched', False) | 
            result_df.get('is_coord_matched', False)
        )
        
        # 매칭 방법 컬럼 추가
        result_df['matching_method'] = 'none'
        
        if 'is_exact_matched' in result_df.columns:
            result_df.loc[result_df['is_exact_matched'], 'matching_method'] = 'exact'
            
        if 'is_fuzzy_matched' in result_df.columns:
            mask = (~result_df['is_exact_matched']) & result_df['is_fuzzy_matched']
            result_df.loc[mask, 'matching_method'] = 'fuzzy'
            
        if 'is_coord_matched' in result_df.columns:
            mask = (~result_df['is_exact_matched']) & (~result_df.get('is_fuzzy_matched', False)) & result_df['is_coord_matched']
            result_df.loc[mask, 'matching_method'] = 'coordinate'
        
        return result_df


# 사용 예시
if __name__ == "__main__":
    # 매처 인스턴스 생성
    matcher = BuildingMatcher()
    
    # 샘플 데이터
    gr_data = pd.DataFrame({
        'id': [1, 2, 3],
        'address': ['서울특별시 강남구 테헤란로 123', '서울특별시 서초구 서초대로 456', '서울특별시 강동구 천호대로 789'],
        'latitude': [37.5012, 37.4831, 37.5384],
        'longitude': [127.0395, 127.0127, 127.1320]
    })
    
    hub_data = pd.DataFrame({
        'id': ['A1', 'A2', 'A3'],
        'address': ['서울특별시 강남구 테헤란로 123', '서울특별시 서초구 서초대로 456번길 10', '인천광역시 부평구 부평대로 123'],
        'latitude': [37.5015, 37.4835, 37.4912],
        'longitude': [127.0392, 127.0125, 126.7235]
    })
    
    # 정확한 주소 매칭
    exact_matches = matcher.match_by_exact_address(gr_data, hub_data)
    print("정확한 주소 매칭 결과:")
    print(exact_matches[['id', 'address', 'is_exact_matched']])
    
    # 퍼지 매칭
    fuzzy_matches = matcher.match_by_fuzzy_address(gr_data, hub_data, threshold=80)
    print("\n퍼지 매칭 결과:")
    print(fuzzy_matches[['id', 'address', 'is_fuzzy_matched']])
    
    # 좌표 기반 매칭
    coord_matches = matcher.match_by_coordinates(gr_data, hub_data, distance_threshold=100)
    print("\n좌표 기반 매칭 결과:")
    print(coord_matches[['id', 'address', 'is_coord_matched']])
    
    # 통합 결과
    combined_results = matcher.combine_matching_results(exact_matches, fuzzy_matches, coord_matches)
    print("\n통합 매칭 결과:")
    print(combined_results[['id', 'address', 'is_matched', 'matching_method']]) 