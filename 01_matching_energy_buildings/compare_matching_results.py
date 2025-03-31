import os
import pandas as pd
import numpy as np

def compare_matching_results(file1, file2, output_file):
    """
    두 매칭 결과 파일을 비교하여 매칭 단계가 달라진 레코드만 분석하고 결과를 저장합니다.
    
    Args:
        file1: 첫 번째 파일 경로 (비교 기준)
        file2: 두 번째 파일 경로 (비교 대상)
        output_file: 결과 저장 파일 경로
    """
    # 파일 읽기
    print(f"파일 읽는 중: {file1}")
    df1 = pd.read_excel(file1)
    print(f"파일 읽는 중: {file2}")
    df2 = pd.read_excel(file2)
    
    # 기본 정보 출력
    print(f"\n=== 파일 기본 정보 ===")
    print(f"파일1: {file1}, 레코드 수: {len(df1)}")
    print(f"파일2: {file2}, 레코드 수: {len(df2)}")
    
    # 두 파일에 모두 있는 SEQ_NO 확인
    if 'SEQ_NO' in df1.columns and 'SEQ_NO' in df2.columns:
        seq_no1 = set(df1['SEQ_NO'])
        seq_no2 = set(df2['SEQ_NO'])
        common_seq_no = seq_no1.intersection(seq_no2)
        
        print(f"\n공통 SEQ_NO 수: {len(common_seq_no)} / {len(seq_no1)} (파일1), {len(seq_no2)} (파일2)")
        
        # 공통 SEQ_NO에 대해서만 비교
        df1_common = df1[df1['SEQ_NO'].isin(common_seq_no)].copy()
        df2_common = df2[df2['SEQ_NO'].isin(common_seq_no)].copy()
        
        # SEQ_NO로 인덱스 설정
        df1_common.set_index('SEQ_NO', inplace=True)
        df2_common.set_index('SEQ_NO', inplace=True)
        
        # 매칭 단계 비교
        if 'MATCH_STAGE' in df1.columns and 'MATCH_STAGE' in df2.columns:
            diff_matches = []
            
            for seq_no in common_seq_no:
                stage1 = df1_common.loc[seq_no, 'MATCH_STAGE'] if seq_no in df1_common.index else None
                stage2 = df2_common.loc[seq_no, 'MATCH_STAGE'] if seq_no in df2_common.index else None
                
                if stage1 != stage2:
                    # 주요 데이터 추출
                    row_data = {
                        'SEQ_NO': seq_no,
                        'RECAP_PK': df1_common.loc[seq_no, 'RECAP_PK'] if 'RECAP_PK' in df1_common.columns else None,
                        'MATCH_STAGE_OLD': stage1,
                        'MATCH_STAGE_NEW': stage2,
                    }
                    
                    # 비교에 중요한 컬럼들 추가
                    for col in ['연면적', 'TOTAREA', '사용승인연도', 'USE_DATE', 'MGM_BLD_PK', 'EBD_COUNT', 'BD_COUNT']:
                        if col in df1_common.columns:
                            row_data[f'{col}_OLD'] = df1_common.loc[seq_no, col]
                        if col in df2_common.columns:
                            row_data[f'{col}_NEW'] = df2_common.loc[seq_no, col]
                    
                    # 데이터 타입 또는 값을 분석하기 위한 정보 추가
                    for col in ['연면적', 'TOTAREA', '사용승인연도', 'USE_DATE']:
                        if col in df1_common.columns:
                            row_data[f'{col}_OLD_TYPE'] = type(df1_common.loc[seq_no, col]).__name__
                        if col in df2_common.columns:
                            row_data[f'{col}_NEW_TYPE'] = type(df2_common.loc[seq_no, col]).__name__
                    
                    # 매칭이 달라진 원인 분석
                    if '연면적' in df1_common.columns and 'TOTAREA' in df1_common.columns and '연면적' in df2_common.columns and 'TOTAREA' in df2_common.columns:
                        area1_old = df1_common.loc[seq_no, '연면적']
                        totarea_old = df1_common.loc[seq_no, 'TOTAREA']
                        area1_new = df2_common.loc[seq_no, '연면적']
                        totarea_new = df2_common.loc[seq_no, 'TOTAREA']
                        
                        # 연면적 일치 여부 분석
                        area_match_old = pd.isna(area1_old) and pd.isna(totarea_old) or (not pd.isna(area1_old) and not pd.isna(totarea_old) and abs(float(area1_old) - float(totarea_old)) < 0.01)
                        area_match_new = pd.isna(area1_new) and pd.isna(totarea_new) or (not pd.isna(area1_new) and not pd.isna(totarea_new) and abs(float(area1_new) - float(totarea_new)) < 0.01)
                        
                        row_data['연면적일치_OLD'] = "일치" if area_match_old else "불일치"
                        row_data['연면적일치_NEW'] = "일치" if area_match_new else "불일치"
                        row_data['연면적변화'] = "있음" if area_match_old != area_match_new else "없음"
                    
                    # 승인일자 일치 여부 분석
                    if '사용승인연도' in df1_common.columns and 'USE_DATE' in df1_common.columns and '사용승인연도' in df2_common.columns and 'USE_DATE' in df2_common.columns:
                        date1_old = df1_common.loc[seq_no, '사용승인연도']
                        date2_old = df1_common.loc[seq_no, 'USE_DATE']
                        date1_new = df2_common.loc[seq_no, '사용승인연도']
                        date2_new = df2_common.loc[seq_no, 'USE_DATE']
                        
                        # 날짜 일치 여부 분석
                        date_match_old = pd.isna(date1_old) and pd.isna(date2_old) or (not pd.isna(date1_old) and not pd.isna(date2_old) and date1_old == date2_old)
                        date_match_new = pd.isna(date1_new) and pd.isna(date2_new) or (not pd.isna(date1_new) and not pd.isna(date2_new) and date1_new == date2_new)
                        
                        row_data['날짜일치_OLD'] = "일치" if date_match_old else "불일치"
                        row_data['날짜일치_NEW'] = "일치" if date_match_new else "불일치"
                        row_data['날짜변화'] = "있음" if date_match_old != date_match_new else "없음"
                    
                    diff_matches.append(row_data)
            
            # 매칭 단계별 변화 통계
            change_stats = {}
            for row in diff_matches:
                key = f"{row['MATCH_STAGE_OLD']} -> {row['MATCH_STAGE_NEW']}"
                change_stats[key] = change_stats.get(key, 0) + 1
            
            print("\n=== 매칭 단계 변화 통계 ===")
            for change, count in change_stats.items():
                print(f"{change}: {count}건")
            
            # 매칭 단계 변화 상세 정보 저장
            if diff_matches:
                diff_df = pd.DataFrame(diff_matches)
                
                # 원인별 개수 파악
                causes = {
                    '연면적변화_있음': len(diff_df[diff_df['연면적변화'] == '있음']) if '연면적변화' in diff_df.columns else 0,
                    '날짜변화_있음': len(diff_df[diff_df['날짜변화'] == '있음']) if '날짜변화' in diff_df.columns else 0
                }
                
                print("\n=== 매칭 단계 변화 원인 분석 ===")
                for cause, count in causes.items():
                    if count > 0:
                        print(f"{cause}: {count}건 ({count/len(diff_df)*100:.1f}%)")
                
                # 원인별 대표 사례 출력
                if '연면적변화' in diff_df.columns and len(diff_df[diff_df['연면적변화'] == '있음']) > 0:
                    sample = diff_df[diff_df['연면적변화'] == '있음'].iloc[0]
                    print(f"\n[연면적 변화 대표 사례] SEQ_NO: {sample['SEQ_NO']}")
                    print(f"  - 이전: 연면적={sample.get('연면적_OLD')}, TOTAREA={sample.get('TOTAREA_OLD')}, 일치여부={sample.get('연면적일치_OLD')}")
                    print(f"  - 이후: 연면적={sample.get('연면적_NEW')}, TOTAREA={sample.get('TOTAREA_NEW')}, 일치여부={sample.get('연면적일치_NEW')}")
                
                if '날짜변화' in diff_df.columns and len(diff_df[diff_df['날짜변화'] == '있음']) > 0:
                    sample = diff_df[diff_df['날짜변화'] == '있음'].iloc[0]
                    print(f"\n[날짜 변화 대표 사례] SEQ_NO: {sample['SEQ_NO']}")
                    print(f"  - 이전: 사용승인연도={sample.get('사용승인연도_OLD')}, USE_DATE={sample.get('USE_DATE_OLD')}, 일치여부={sample.get('날짜일치_OLD')}")
                    print(f"  - 이후: 사용승인연도={sample.get('사용승인연도_NEW')}, USE_DATE={sample.get('USE_DATE_NEW')}, 일치여부={sample.get('날짜일치_NEW')}")
                
                diff_df.to_excel(output_file, index=False)
                print(f"\n매칭 단계 변화 상세 정보가 {output_file}에 저장되었습니다. (총 {len(diff_df)}건)")
            else:
                print("\n두 파일의 매칭 단계 결과가 완전히 동일합니다.")
        else:
            print("MATCH_STAGE 컬럼이 존재하지 않아 비교할 수 없습니다.")
    else:
        print("SEQ_NO 컬럼이 존재하지 않아 비교할 수 없습니다.")

def main():
    # 결과 디렉토리 확인
    result_dir = "./result"
    if not os.path.exists(result_dir):
        result_dir = "../result"
    if not os.path.exists(result_dir):
        result_dir = "./01_matching_energy_buildings/result"
    
    # 파일 경로 설정
    file1 = os.path.join(result_dir, "rule_matching_result_ver3.xlsx")
    file2 = os.path.join(result_dir, "rule_matching_result_ver5.xlsx")
    output_file = os.path.join(result_dir, "matching_changes_v3_vs_v5.xlsx")
    
    # 파일 존재 확인
    if not os.path.exists(file1):
        print(f"오류: {file1} 파일이 존재하지 않습니다.")
        return
    
    if not os.path.exists(file2):
        print(f"오류: {file2} 파일이 존재하지 않습니다.")
        return
    
    # 비교 실행
    compare_matching_results(file1, file2, output_file)

if __name__ == "__main__":
    main() 