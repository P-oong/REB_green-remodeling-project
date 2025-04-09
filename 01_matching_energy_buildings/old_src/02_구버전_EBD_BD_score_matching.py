import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # 진행률 표시를 위해 추가
import time  # 실행 시간 측정을 위해 추가

def load_data():
    """
    기존 매칭 결과와 BD 데이터를 로드
    """
    # 기존 매칭 결과 로드
    rule_result_df = pd.read_excel("./result/rule_matching_result_ver3.xlsx")
    
    # BD 데이터 로드
    bd_df = pd.read_excel("./data/BD_data_all.xlsx")
    
    return rule_result_df, bd_df

def extract_unmatched_ebd(rule_result_df):
    """
    MATCH_STAGE가 '미매칭'인 EBD 데이터만 추출
    """
    # EBD 관련 컬럼만 추출 (영문 컬럼과 기본 정보 제외)
    unmatched_df = rule_result_df[rule_result_df['MATCH_STAGE'] == '미매칭'].copy()
    
    # 필요한 컬럼만 선택 (한글 변수명 + 기본 식별 정보)
    ebd_columns = ['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '기관명', '건축물명', '주소', '지상', '지하', 'MGM_BLD_PK', 'MATCH_STAGE', 'EBD_COUNT', 'BD_COUNT', 'EBD_OVER_BD']
    ebd_columns = [col for col in ebd_columns if col in unmatched_df.columns]
    unmatched_ebd = unmatched_df[ebd_columns]
    
    return unmatched_ebd

def preprocess_data(df, columns, is_bd=False):
    """
    데이터 전처리: 숫자 컬럼 변환, 문자열 컬럼 소문자화
    """
    result = df.copy()
    
    # 숫자 컬럼 처리
    numeric_columns = ['연면적', 'TOTAREA', '사용승인연도', 'USE_DATE'] if is_bd else ['연면적', '사용승인연도']
    for col in numeric_columns:
        if col in result.columns:
            # NaN이 아닌 값만 변환 (에러 방지)
            mask = ~pd.isna(result[col])
            try:
                result.loc[mask, col] = pd.to_numeric(result.loc[mask, col], errors='coerce')
            except:
                pass
    
    # 문자열 컬럼 처리
    text_columns = ['BLD_NM', 'DONG_NM'] if is_bd else ['기관명', '건축물명', '주소']
    for col in text_columns:
        if col in result.columns:
            result[col] = result[col].astype(str).str.lower()
    
    return result[columns]

def optimize_score_based_matching(unmatched_ebd, bd_df):
    """
    최적화된 점수 기반 매칭 수행:
    1. 데이터 사전 처리로 반복 연산 최소화
    2. 필요한 컬럼만 사용하여 메모리 효율화
    3. 불필요한 조합은 사전에 필터링
    """
    start_time = time.time()
    print("최적화된 점수 기반 매칭을 시작합니다...")
    
    # 결과 저장을 위한 리스트
    match_results = []
    
    # 데이터 사전 처리
    ebd_columns = ['SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '기관명', '건축물명', '주소']
    bd_columns = ['MGM_BLD_PK', 'RECAP_PK', 'TOTAREA', 'USE_DATE', 'BLD_NM', 'DONG_NM']
    
    # 사용할 컬럼만 선택하고 전처리
    ebd_processed = preprocess_data(unmatched_ebd, ebd_columns)
    bd_processed = preprocess_data(bd_df, bd_columns, is_bd=True)
    
    # 전체 ebd 레코드 수 확인 (진행률 표시용)
    total_ebd_records = len(ebd_processed)
    print(f"처리할 미매칭 EBD 레코드: {total_ebd_records}개")
    
    # RECAP별 BD 데이터 미리 필터링하여 딕셔너리로 저장 (중복 필터링 방지)
    recap_to_bd = {}
    valid_recaps = set(ebd_processed['RECAP_PK'].dropna().unique())
    
    for recap in valid_recaps:
        recap_to_bd[recap] = bd_processed[bd_processed['RECAP_PK'] == recap]
    
    # 결과 저장 딕셔너리
    processed_count = 0
    matched_count = 0
    
    # 데이터프레임을 iterrows로 처리 (tqdm으로 진행률 표시)
    for idx, ebd_row in tqdm(ebd_processed.iterrows(), total=total_ebd_records, desc="EBD 레코드 처리"):
        # 원본 데이터 가져오기
        original_ebd_row = unmatched_ebd.loc[idx].to_dict()
        recap = ebd_row['RECAP_PK']
        
        # RECAP이 없거나 BD 후보가 없는 경우
        if pd.isna(recap) or recap not in recap_to_bd or recap_to_bd[recap].empty:
            reason = '미매칭(RECAP_NA)' if pd.isna(recap) else '미매칭(후보없음)'
            original_ebd_row['MATCH_STAGE'] = reason
            match_results.append(original_ebd_row)
            processed_count += 1
            continue
        
        # BD 후보들
        bd_candidates = recap_to_bd[recap]
        best_score = 0
        best_match = None
        score_counts = {}
        
        # 각 BD 후보에 대해 점수 계산
        for bd_idx, bd_row in bd_candidates.iterrows():
            # 1. 면적 점수
            area_score = 0
            if not pd.isna(ebd_row['연면적']) and not pd.isna(bd_row['TOTAREA']):
                ebd_area = float(ebd_row['연면적'])
                bd_area = float(bd_row['TOTAREA'])
                if ebd_area * 0.95 <= bd_area <= ebd_area * 1.05:
                    area_score = 1
            
            # 2. 날짜 점수
            date_score = 0
            if not pd.isna(ebd_row['사용승인연도']) and not pd.isna(bd_row['USE_DATE']):
                try:
                    ebd_date = str(int(ebd_row['사용승인연도']))
                    bd_date = str(int(bd_row['USE_DATE']))
                    if ebd_date == bd_date:
                        date_score = 1
                except:
                    pass
            
            # 3. 텍스트 점수
            text_score = 0
            # 3.1 기관명이 BLD_NM에 포함
            if str(ebd_row['기관명']) != 'nan' and str(ebd_row['기관명']) in str(bd_row['BLD_NM']):
                text_score += 1
            
            # 3.2 건축물명이 BLD_NM 또는 DONG_NM에 포함
            if str(ebd_row['건축물명']) != 'nan' and (
                str(ebd_row['건축물명']) in str(bd_row['BLD_NM']) or 
                str(ebd_row['건축물명']) in str(bd_row['DONG_NM'])
            ):
                text_score += 1
            
            # 3.3 결합 텍스트 비교
            ebd_combined = (str(ebd_row['기관명']) + " " + str(ebd_row['건축물명'])).strip()
            bd_combined = (str(bd_row['BLD_NM']) + " " + str(bd_row['DONG_NM'])).strip()
            
            if ebd_combined != " " and ebd_combined in bd_combined:
                text_score += 1
            
            # 총점 계산
            total_score = area_score + date_score + text_score
            
            # 최고 점수 갱신
            if total_score > best_score:
                best_score = total_score
                best_match = {
                    'bd_row': bd_df.loc[bd_idx].to_dict(),  # 원본 BD 데이터 사용
                    'area_score': area_score,
                    'date_score': date_score,
                    'text_score': text_score,
                    'total_score': total_score
                }
            
            # 점수 카운트 (중복 여부 확인용)
            score_counts[total_score] = score_counts.get(total_score, 0) + 1
        
        # 매칭 여부 결정
        if best_score >= 2 and score_counts[best_score] == 1:
            # 매칭 결과 생성
            result = original_ebd_row.copy()
            
            # BD 정보 추가
            for key, value in best_match['bd_row'].items():
                if key not in result:  # 중복 컬럼 처리
                    result[key] = value
            
            # 점수 정보 추가
            result['AREA_SCORE'] = best_match['area_score']
            result['DATE_SCORE'] = best_match['date_score']
            result['TEXT_SCORE'] = best_match['text_score']
            result['TOTAL_SCORE'] = best_match['total_score']
            result['MATCH_STAGE'] = '4차'
            
            matched_count += 1
        else:
            # 매칭되지 않은 경우
            result = original_ebd_row.copy()
            
            if best_score < 2:
                result['MATCH_STAGE'] = '미매칭(점수미달)'
            else:
                result['MATCH_STAGE'] = '미매칭(중복후보)'
            
            # 점수 정보 추가 (분석용)
            if best_match:
                result['AREA_SCORE'] = best_match['area_score']
                result['DATE_SCORE'] = best_match['date_score']
                result['TEXT_SCORE'] = best_match['text_score']
                result['TOTAL_SCORE'] = best_match['total_score']
        
        match_results.append(result)
        processed_count += 1
        
        # 진행 상황 업데이트 (100개 단위)
        if processed_count % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"처리 진행: {processed_count}/{total_ebd_records} 레코드 완료 ({matched_count} 매칭됨) - 경과 시간: {elapsed_time:.2f}초")
    
    # 리스트를 데이터프레임으로 변환
    result_df = pd.DataFrame(match_results)
    
    # 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"매칭 완료: 총 {processed_count}개 중 {matched_count}개가 매칭됨 - 총 소요 시간: {elapsed_time:.2f}초")
    
    return result_df

def main():
    start_time = time.time()
    print("4차 점수 기반 매칭을 시작합니다...")
    
    # 데이터 로드
    rule_result_df, bd_df = load_data()
    print(f"데이터 로드 완료: EBD {len(rule_result_df)}개, BD {len(bd_df)}개")
    
    # 원본 순서 보존을 위한 컬럼 추가
    rule_result_df['_원본순서'] = range(len(rule_result_df))
    
    # 미매칭 EBD 추출
    unmatched_ebd = extract_unmatched_ebd(rule_result_df)
    print(f"미매칭 EBD 건수: {len(unmatched_ebd)}")
    
    # 최적화된 점수 기반 매칭 수행
    score_result_df = optimize_score_based_matching(unmatched_ebd, bd_df)
    
    # 통계 출력
    stage_counts = score_result_df['MATCH_STAGE'].value_counts()
    print("\n4차 점수 기반 매칭 결과:")
    for stage, count in stage_counts.items():
        print(f"- {stage}: {count}건")
    
    # 기존 매칭 결과에서 미매칭 레코드만 제외
    matched_from_rules = rule_result_df[rule_result_df['MATCH_STAGE'] != '미매칭'].copy()
    
    # 기존 매칭 결과와 새로운 점수 기반 매칭 결과 합치기
    final_result = pd.concat([matched_from_rules, score_result_df], ignore_index=False)
    
    # 원본 순서대로 정렬
    final_result = final_result.sort_values('_원본순서')
    
    # 통계 출력
    final_counts = final_result['MATCH_STAGE'].value_counts()
    print("\n최종 매칭 결과:")
    for stage, count in final_counts.items():
        print(f"- {stage}: {count}건")
    
    # 임시 순서 컬럼 제거
    if '_원본순서' in final_result.columns:
        final_result = final_result.drop('_원본순서', axis=1)
    
    # 결과 저장
    os.makedirs("./result", exist_ok=True)
    final_result.to_excel("./result/score_matching_result_ver1.xlsx", index=False)
    print("\n최종 결과가 './result/score_matching_result_ver1.xlsx'에 저장되었습니다.")
    
    # 점수 기반 매칭 결과만 따로 저장
    score_result_df.to_excel("./result/score_only_matching_result_ver1.xlsx", index=False)
    print("4차 점수 기반 매칭 결과가 './result/score_only_matching_result_ver1.xlsx'에 저장되었습니다.")
    
    # 총 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")
    
    return final_result

if __name__ == "__main__":
    main()
