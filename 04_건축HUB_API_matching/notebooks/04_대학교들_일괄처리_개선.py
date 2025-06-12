import pandas as pd
import os
import glob
import re
from pathlib import Path
from collections import defaultdict

def tokenize_text(text):
    """텍스트를 전처리하고 토큰화하여 집합(set)으로 반환"""
    if pd.isna(text) or text is None:
        return set()
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [token.strip() for token in tokens if token.strip()]
    return set(tokens)

def safe_date_convert(date_value):
    """날짜를 YYYY-MM-DD 형식으로 안전하게 변환"""
    if pd.isna(date_value):
        return None
    try:
        # 이미 datetime 객체인 경우
        if hasattr(date_value, 'strftime'):
            return date_value.strftime('%Y-%m-%d')
        
        # 문자열로 변환
        date_str = str(date_value).strip()
        
        # 소수점이 있는 경우 제거 (19871228.0 -> 19871228)
        if '.' in date_str:
            date_str = date_str.split('.')[0]
        
        # YYYYMMDD 형태인 경우 (8자리)
        if len(date_str) == 8 and date_str.isdigit():
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            formatted_date = f"{year}-{month}-{day}"
            return pd.to_datetime(formatted_date).strftime('%Y-%m-%d')
        
        # 기타 형태는 pandas가 자동으로 파싱하도록 시도
        return pd.to_datetime(date_value).strftime('%Y-%m-%d')
        
    except:
        return None

def safe_equals(a, b):
    """안전한 동등 비교"""
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return float(a) == float(b)
    except:
        return str(a) == str(b)

def is_within_percentage(a, b, percentage):
    """두 숫자가 지정된 백분율 이내인지 확인"""
    if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
        return False
    min_val = min(a, b)
    max_val = max(a, b)
    return ((max_val - min_val) / min_val) <= (percentage / 100)

def check_match_condition(mc_row, combined_row, step):
    """각 단계별 매칭 조건 확인"""
    if step == 1:  # 연면적 + 사용승인날짜 정확히 일치
        area_match = safe_equals(mc_row['연면적'], combined_row['연면적(㎡)'])
        mc_date = safe_date_convert(mc_row['사용승인연도'])
        combined_date = safe_date_convert(combined_row['사용승인일'])
        date_match = mc_date == combined_date and mc_date is not None
        return area_match and date_match
    
    elif step == 2:  # 연면적 + 텍스트 토큰 일치
        area_match = safe_equals(mc_row['연면적'], combined_row['연면적(㎡)'])
        token_match = len(mc_row['mc_tokens'] & combined_row['combined_tokens']) > 0
        return area_match and token_match
    
    elif step == 3:  # 연면적 정확히 일치
        return safe_equals(mc_row['연면적'], combined_row['연면적(㎡)'])
    
    elif step == 4:  # 연면적 1% 이내 + 사용승인날짜 일치
        area_match = is_within_percentage(mc_row['연면적'], combined_row['연면적(㎡)'], 1)
        mc_date = safe_date_convert(mc_row['사용승인연도'])
        combined_date = safe_date_convert(combined_row['사용승인일'])
        date_match = mc_date == combined_date and mc_date is not None
        return area_match and date_match
    
    elif step == 5:  # 연면적 1% 이내 + 텍스트 토큰 일치
        area_match = is_within_percentage(mc_row['연면적'], combined_row['연면적(㎡)'], 1)
        token_match = len(mc_row['mc_tokens'] & combined_row['combined_tokens']) > 0
        return area_match and token_match
    
    elif step == 6:  # 연면적 1% 이내 일치
        return is_within_percentage(mc_row['연면적'], combined_row['연면적(㎡)'], 1)
    
    elif step == 7:  # 텍스트 토큰 일치
        return len(mc_row['mc_tokens'] & combined_row['combined_tokens']) > 0
    
    return False

def load_university_data():
    """대학교 CSV 파일들을 로드하고 통합"""
    print("=== 대학교 CSV 파일 로드 시작 ===")
    
    # 데이터 폴더 경로 설정
    data_folder = '../data'
    
    # 대학교 관련 csv 파일들
    university_csv_files = [
        '전북대_금암동18-1.csv',
        '전북대_금암동634-18.csv', 
        '전북대_금암동663.csv',
        '전북대_덕진동1가664-1.csv',
        '서울대_신림동56-1.csv',
        '서울대_봉천동4-2.csv',
        '서울대_봉천동4-1.csv',
        '부경대_559-1.csv',
        '부산대_장전동40.csv',
        '부산대_장전동30.csv',
        '경북대학교.csv'
    ]
    
    all_dataframes = []
    
    for file in university_csv_files:
        file_path = os.path.join(data_folder, file)
        
        try:
            # csv 파일 읽기
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp949')
            
            # 파일명에서 대학교명과 지역 정보 추출
            university_name = file.split('_')[0]
            location_info = file.replace('.csv', '').split('_')[1] if '_' in file else ''
            
            # 데이터프레임에 출처 정보 추가
            df['대학교명'] = university_name
            df['지역정보'] = location_info
            df['원본파일명'] = file
            
            all_dataframes.append(df)
            print(f"✓ {file}: {len(df)}건 로드 완료")
            
        except Exception as e:
            print(f"✗ {file} 읽기 실패: {str(e)}")
    
    # 모든 데이터프레임을 하나로 통합
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 대학교명 변경
    university_name_mapping = {
        '서울대': '서울대학교',
        '전북대': '전북대학교', 
        '부경대': '부경대학교',
        '부산대': '부산대학교',
        '경북대학교.csv': '경북대학교'
    }
    combined_df['대학교명'] = combined_df['대학교명'].map(university_name_mapping)
    
    # 관리건축물대장PK 컬럼명을 new_MGM으로 변경
    combined_df = combined_df.rename(columns={'관리건축물대장PK': 'new_MGM'})
    
    # MGM_BLD_PK 변수 생성
    def create_mgm_bld_pk(row):
        시군구코드 = str(row['시군구코드'])
        new_mgm = str(row['new_MGM'])
        
        if len(new_mgm) >= 20:
            processed_mgm = new_mgm
        else:
            processed_mgm = new_mgm[5:] if len(new_mgm) > 5 else new_mgm
        
        return f"{시군구코드}-{processed_mgm}"
    
    combined_df['MGM_BLD_PK'] = combined_df.apply(create_mgm_bld_pk, axis=1)
    
    print(f"총 {len(combined_df)}건의 대학교 데이터 통합 완료")
    return combined_df

def load_matching_data():
    """매칭용 엑셀 파일 로드"""
    print("=== 매칭 데이터 로드 ===")
    mc_df = pd.read_excel('../data/batch_stage_matching_result_final.xlsx')
    print(f"매칭 데이터 로드 완료: {len(mc_df):,}건")
    return mc_df

def check_date_conversion(mc_df, combined_df, target_universities):
    """사용승인일 변환 결과 확인"""
    print("=== 사용승인일 변환 결과 확인 ===")

    # mc_df의 사용승인연도 변환 결과 샘플
    print("\n--- mc_df 사용승인연도 변환 결과 (샘플 10개) ---")
    mc_sample = mc_df[mc_df['기관명'].isin(target_universities)].head(10)
    for idx, row in mc_sample.iterrows():
        original = row['사용승인연도']
        converted = safe_date_convert(original)
        print(f"원본: {original} → 변환: {converted}")

    # combined_df의 사용승인일 변환 결과 샘플  
    print("\n--- combined_df 사용승인일 변환 결과 (샘플 10개) ---")
    combined_sample = combined_df.head(10)
    for idx, row in combined_sample.iterrows():
        original = row['사용승인일']
        converted = safe_date_convert(original)
        print(f"원본: {original} → 변환: {converted}")

    # 각 대학교별 날짜 변환 통계
    print("\n--- 대학교별 날짜 변환 통계 ---")
    for university in target_universities:
        mc_univ = mc_df[mc_df['기관명'] == university]
        combined_univ = combined_df[combined_df['대학교명'] == university]
        
        # mc_df 날짜 변환 성공률
        mc_dates = mc_univ['사용승인연도'].apply(safe_date_convert)
        mc_valid = mc_dates.notna().sum()
        mc_total = len(mc_univ)
        
        # combined_df 날짜 변환 성공률
        combined_dates = combined_univ['사용승인일'].apply(safe_date_convert)
        combined_valid = combined_dates.notna().sum()
        combined_total = len(combined_univ)
        
        print(f"{university}:")
        print(f"  mc_df: {mc_valid}/{mc_total} ({mc_valid/mc_total*100:.1f}%)")
        print(f"  combined_df: {combined_valid}/{combined_total} ({combined_valid/combined_total*100:.1f}%)")

def match_universities_improved(mc_df, combined_df):
    """개선된 대학교 매칭 수행 - 1:1 매칭만 허용, 매칭된 MGM 제외"""
    print("\n=== 개선된 대학교 데이터 매칭 시작 ===")
    
    target_universities = ['서울대학교', '전북대학교', '부경대학교', '경북대학교', '부산대학교']
    
    # matching_step 컬럼 초기화
    if 'matching_step' not in mc_df.columns:
        mc_df['matching_step'] = None
    
    # 날짜 변환 결과 확인
    check_date_conversion(mc_df, combined_df, target_universities)
    
    # mc_df에 토큰 컬럼 추가
    mc_df['mc_tokens'] = mc_df['건축물명'].apply(tokenize_text)
    
    # combined_df에 토큰 컬럼 추가
    combined_df['combined_tokens'] = combined_df.apply(
        lambda row: tokenize_text(str(row['건물명']) + ' ' + str(row['동명칭'])), axis=1
    )
    
    matching_results = []
    total_matched = 0
    
    # 이미 매칭된 combined_df 인덱스들을 추적
    matched_combined_indices = set()
    
    for university in target_universities:
        print(f"\n--- {university} 매칭 시작 ---")
        
        # 해당 대학교의 mc_df 데이터에서 MGM_BLD_PK가 비어있는 데이터만 필터링
        mc_univ_data = mc_df[mc_df['기관명'] == university].copy()
        empty_mgm_mask = mc_univ_data['MGM_BLD_PK'].isna() | (mc_univ_data['MGM_BLD_PK'] == '')
        mc_univ_data = mc_univ_data[empty_mgm_mask]
        
        print(f"{university} 매칭 대상: {len(mc_univ_data)}건")
        
        if len(mc_univ_data) == 0:
            print(f"{university}: 매칭할 데이터가 없습니다.")
            continue
        
        # 해당 대학교의 combined_df 데이터
        combined_univ_data = combined_df[combined_df['대학교명'] == university]
        print(f"{university} 후보군: {len(combined_univ_data)}건")
        
        if len(combined_univ_data) == 0:
            print(f"{university}: 후보군이 없습니다.")
            continue
        
        univ_matched = 0
        
        # 7단계 순차적 매칭
        for step in range(1, 8):
            print(f"\n  {step}단계 매칭 시작...")
            step_matched = 0
            
            # 이번 단계에서 매칭될 mc 인덱스들을 저장
            step_matches = []
            
            # 매칭되지 않은 mc 데이터들만 대상으로 함 (수정된 부분)
            unmatched_mask = mc_df['matching_step'].isna()
            available_mc_indices = mc_univ_data.index.intersection(mc_df[unmatched_mask].index)
            available_mc_data = mc_univ_data.loc[available_mc_indices]
            
            # 매칭되지 않은 combined 데이터들만 대상으로 함
            available_combined_data = combined_univ_data[~combined_univ_data.index.isin(matched_combined_indices)]
            
            if len(available_mc_data) == 0 or len(available_combined_data) == 0:
                print(f"    사용 가능한 데이터가 없습니다. (MC: {len(available_mc_data)}, Combined: {len(available_combined_data)})")
                continue
            
            print(f"    사용 가능한 MC 데이터: {len(available_mc_data)}건, Combined 데이터: {len(available_combined_data)}건")
            
            # 각 mc 데이터에 대해 후보 찾기
            for mc_idx, mc_row in available_mc_data.iterrows():
                candidates = []
                
                # 모든 사용 가능한 후보군과 비교
                for combined_idx, combined_row in available_combined_data.iterrows():
                    if check_match_condition(mc_row, combined_row, step):
                        candidates.append((combined_idx, combined_row))
                
                # 유일한 후보가 있는 경우만 매칭 예약
                if len(candidates) == 1:
                    combined_idx, combined_row = candidates[0]
                    step_matches.append((mc_idx, mc_row, combined_idx, combined_row))
            
            # 1:1 매칭 확인 및 실제 매칭 수행
            final_matches = []
            used_combined_indices = set()
            
            for mc_idx, mc_row, combined_idx, combined_row in step_matches:
                # 이미 다른 mc에 의해 사용된 combined_idx가 아닌지 확인
                if combined_idx not in used_combined_indices:
                    # 해당 combined_idx가 다른 mc의 유일한 후보이기도 한지 확인
                    conflict = False
                    for other_mc_idx, _, other_combined_idx, _ in step_matches:
                        if other_mc_idx != mc_idx and other_combined_idx == combined_idx:
                            conflict = True
                            break
                    
                    if not conflict:
                        final_matches.append((mc_idx, mc_row, combined_idx, combined_row))
                        used_combined_indices.add(combined_idx)
            
            # 최종 매칭 적용
            for mc_idx, mc_row, combined_idx, combined_row in final_matches:
                # 매칭 결과 저장 (타입 변환 추가)
                mc_df.at[mc_idx, 'new_MGM'] = str(combined_row['new_MGM'])
                mc_df.at[mc_idx, 'MGM_BLD_PK'] = str(combined_row['MGM_BLD_PK'])
                mc_df.at[mc_idx, 'matching_step'] = f"{step}차"
                
                # 매칭에 사용된 변수들 저장
                if step in [1, 4]:  # 날짜 포함 단계
                    mc_df.at[mc_idx, 'matched_date'] = safe_date_convert(combined_row['사용승인일'])
                if step in [2, 5, 7]:  # 토큰 포함 단계
                    matched_tokens = mc_row['mc_tokens'] & combined_row['combined_tokens']
                    mc_df.at[mc_idx, 'matched_tokens'] = ', '.join(matched_tokens)
                
                mc_df.at[mc_idx, 'matched_area'] = float(combined_row['연면적(㎡)'])
                
                matching_results.append({
                    'university': university,
                    'mc_index': mc_idx,
                    'combined_index': combined_idx,
                    'step': step,
                    'mc_area': mc_row['연면적'],
                    'combined_area': combined_row['연면적(㎡)'],
                    'building_name': mc_row['건축물명']
                })
                
                # 매칭된 combined 인덱스를 전체 매칭된 인덱스 집합에 추가
                matched_combined_indices.add(combined_idx)
                step_matched += 1
                univ_matched += 1
            
            print(f"    {step}단계 매칭 완료: {step_matched}건")
        
        print(f"{university} 총 매칭 완료: {univ_matched}건")
        total_matched += univ_matched
    
    print(f"\n=== 전체 매칭 완료 ===")
    print(f"총 매칭된 건수: {total_matched}건")
    print(f"총 사용된 combined 인덱스: {len(matched_combined_indices)}개")
    
    # 단계별 매칭 결과 요약
    if 'matching_step' in mc_df.columns:
        step_summary = mc_df['matching_step'].value_counts().sort_index()
        print("\n=== 단계별 매칭 결과 ===")
        for step, count in step_summary.items():
            print(f"{step}: {count}건")
    
    return matching_results, mc_df

def print_matching_summary(mc_df):
    """매칭 결과 요약 출력"""
    target_universities = ['서울대학교', '전북대학교', '부경대학교', '경북대학교', '부산대학교']
    
    print("\n=== 각 대학교별 MGM_BLD_PK 현황 ===")
    
    summary_data = []
    
    for university in target_universities:
        # 해당 대학교 전체 데이터
        univ_total = mc_df[mc_df['기관명'] == university]
        total_count = len(univ_total)
        
        # 이번에 매칭된 건수 (matching_step이 있는 것들)
        if 'matching_step' in mc_df.columns:
            newly_matched = len(univ_total[univ_total['matching_step'].notna()])
        else:
            newly_matched = 0
        
        # 현재 비어있는 건수 (매칭 후)
        currently_empty = (univ_total['MGM_BLD_PK'].isna() | (univ_total['MGM_BLD_PK'] == '')).sum()
        
        # 기존에 채워져 있던 건수 = 전체 - 이번 매칭 - 현재 빈 건수
        previously_filled = total_count - newly_matched - currently_empty
        
        # 매칭 전 비어있던 건수 = 이번 매칭 + 현재 빈 건수
        originally_empty = newly_matched + currently_empty
        
        summary_data.append({
            'university': university,
            'total': total_count,
            'previously_filled': previously_filled, 
            'originally_empty': originally_empty,
            'newly_matched': newly_matched,
            'still_empty': currently_empty
        })
        
        print(f"\n{university}:")
        print(f"  - 전체 건수: {total_count}건")
        print(f"  - 기존에 채워진 건수: {previously_filled}건") 
        print(f"  - 매칭 전 비어있던 건수: {originally_empty}건")
        print(f"  - 이번에 매칭된 건수: {newly_matched}건")
        print(f"  - 여전히 비어있는 건수: {currently_empty}건")
        print(f"  - 매칭률: {newly_matched/originally_empty*100:.1f}%" if originally_empty > 0 else "  - 매칭률: 100.0% (비어있는 데이터 없음)")

    # 전체 요약
    total_all = sum([data['total'] for data in summary_data])
    total_originally_empty = sum([data['originally_empty'] for data in summary_data])
    total_newly_matched = sum([data['newly_matched'] for data in summary_data])
    total_still_empty = sum([data['still_empty'] for data in summary_data])

    print(f"\n=== 전체 요약 ===")
    print(f"5개 대학교 전체 건수: {total_all}건")
    print(f"매칭 전 총 빈 건수: {total_originally_empty}건")
    print(f"이번에 매칭된 건수: {total_newly_matched}건")
    print(f"여전히 빈 건수: {total_still_empty}건")
    print(f"전체 매칭률: {total_newly_matched/total_originally_empty*100:.1f}%" if total_originally_empty > 0 else "전체 매칭률: 100.0%")
    
    # 검증: 전체 = 기존 채워진 + 이번 매칭 + 여전히 빈 것
    total_previously_filled = sum([data['previously_filled'] for data in summary_data])
    calculated_total = total_previously_filled + total_newly_matched + total_still_empty
    print(f"\n=== 검증 ===")
    print(f"계산된 전체: {calculated_total}건 (실제 전체: {total_all}건)")
    if calculated_total == total_all:
        print("✅ 계산이 정확합니다!")
    else:
        print("❌ 계산에 오류가 있습니다!")

def main():
    """메인 실행 함수"""
    try:
        # 1. 대학교 데이터 로드
        combined_df = load_university_data()
        
        # 2. 매칭 데이터 로드
        mc_df = load_matching_data()
        
        # 3. 개선된 매칭 수행
        results, updated_mc_df = match_universities_improved(mc_df, combined_df)
        
        # 4. 결과 요약 출력
        print_matching_summary(updated_mc_df)
        
        # 5. 결과 저장
        output_file = '../data/대학교_매칭_결과_개선_ver3.xlsx'
        updated_mc_df.to_excel(output_file, index=False)
        print(f"\n결과가 저장되었습니다: {output_file}")
        
        return updated_mc_df, results
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    mc_df, results = main() 