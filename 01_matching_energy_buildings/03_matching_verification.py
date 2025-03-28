import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_matching_results(result_df):
    """
    매칭 결과에 대한 기본 통계 분석
    """
    # 매칭 단계별 통계
    match_stats = result_df['MATCH_STAGE'].value_counts()
    print("매칭 단계별 건수:")
    for stage, count in match_stats.items():
        print(f"- {stage}: {count}건 ({count/len(result_df)*100:.1f}%)")
    
    # 점수 분포 분석
    if 'TOTAL_SCORE' in result_df.columns:
        matched_df = result_df[~result_df['MATCH_STAGE'].str.contains('미매칭')]
        if not matched_df.empty:
            print("\n매칭된 결과의 점수 분포:")
            print(f"- 평균: {matched_df['TOTAL_SCORE'].mean():.2f}")
            print(f"- 중앙값: {matched_df['TOTAL_SCORE'].median():.2f}")
            print(f"- 최소값: {matched_df['TOTAL_SCORE'].min():.2f}")
            print(f"- 최대값: {matched_df['TOTAL_SCORE'].max():.2f}")
            
            # 점수 범위별 분포
            bins = [0, 1.8, 2.0, 2.5, 3.0, 3.8]
            labels = ['< 1.8', '1.8-2.0', '2.0-2.5', '2.5-3.0', '> 3.0']
            score_dist = pd.cut(matched_df['TOTAL_SCORE'], bins=bins, labels=labels).value_counts().sort_index()
            for label, count in score_dist.items():
                print(f"- {label}: {count}건 ({count/len(matched_df)*100:.1f}%)")
    
    # 점수 컴포넌트 분석
    if all(col in result_df.columns for col in ['AREA_SCORE', 'DATE_SCORE', 'BLD_SCORE', 'DONG_SCORE']):
        matched_df = result_df[~result_df['MATCH_STAGE'].str.contains('미매칭')]
        if not matched_df.empty:
            print("\n매칭된 결과의 점수 컴포넌트 통계:")
            for col in ['AREA_SCORE', 'DATE_SCORE', 'BLD_SCORE', 'DONG_SCORE']:
                mean_val = matched_df[col].mean()
                print(f"- {col} 평균: {mean_val:.2f}")

def verify_area_matching(result_df):
    """
    연면적 매칭 결과 검증
    """
    # 연면적이 있는 매칭된 레코드만 선택
    matched_with_area = result_df[
        (~result_df['MATCH_STAGE'].str.contains('미매칭')) & 
        (~pd.isna(result_df['연면적'])) & 
        (~pd.isna(result_df['TOTAREA']))
    ]
    
    if not matched_with_area.empty:
        # 연면적 차이 계산
        matched_with_area['면적차이비율'] = abs(matched_with_area['연면적'] - matched_with_area['TOTAREA']) / matched_with_area['연면적']
        
        # 면적 차이 분포
        print("연면적 차이 분포:")
        bins = [0, 0.01, 0.05, 0.1, 0.2, float('inf')]
        labels = ['1% 이내', '1-5%', '5-10%', '10-20%', '20% 초과']
        area_diff_dist = pd.cut(matched_with_area['면적차이비율'], bins=bins, labels=labels).value_counts().sort_index()
        
        for label, count in area_diff_dist.items():
            print(f"- {label}: {count}건 ({count/len(matched_with_area)*100:.1f}%)")
        
        # 면적 차이가 큰 상위 10개 케이스 출력
        print("\n면적 차이가 큰 상위 10개 매칭:")
        large_diff = matched_with_area.nlargest(10, '면적차이비율')
        for _, row in large_diff.iterrows():
            print(f"- SEQ_NO: {row.get('SEQ_NO', 'N/A')}, EBD 연면적: {row['연면적']:.1f}, BD 연면적: {row['TOTAREA']:.1f}, " 
                  f"차이: {row['면적차이비율']*100:.1f}%, 총점: {row['TOTAL_SCORE']:.1f}")
        
        # 결과를 엑셀 파일로 저장 (면적 차이가 20% 이상인 케이스)
        large_diff_all = matched_with_area[matched_with_area['면적차이비율'] > 0.2]
        if not large_diff_all.empty:
            os.makedirs("./verification", exist_ok=True)
            large_diff_all.to_excel("./verification/large_area_diff_cases.xlsx", index=False)
            print(f"\n면적 차이가 20% 이상인 {len(large_diff_all)}개 케이스를 './verification/large_area_diff_cases.xlsx'에 저장했습니다.")

def analyze_recap_matching_patterns(result_df):
    """
    RECAP별 매칭 패턴 분석
    """
    if 'RECAP_PK' not in result_df.columns:
        print("RECAP_PK 컬럼이 데이터에 없습니다.")
        return
    
    # RECAP이 없는 레코드 제외
    result_with_recap = result_df[~pd.isna(result_df['RECAP_PK'])]
    
    # RECAP별 매칭 결과 분석
    recap_stats = result_with_recap.groupby('RECAP_PK')['MATCH_STAGE'].value_counts().unstack(fill_value=0)
    
    # RECAP 내 EBD와 BD 개수 비교
    if all(col in result_with_recap.columns for col in ['EBD_COUNT', 'BD_COUNT']):
        recap_counts = result_with_recap.groupby('RECAP_PK').agg({
            'EBD_COUNT': 'first',
            'BD_COUNT': 'first'
        })
        
        # RECAP별 매칭률 계산
        recap_match_stats = pd.DataFrame()
        recap_match_stats['총EBD건수'] = recap_stats.sum(axis=1)
        
        # 매칭된 건수 계산 (미매칭으로 시작하지 않는 모든 MATCH_STAGE)
        match_columns = [col for col in recap_stats.columns if not col.startswith('미매칭')]
        if match_columns:
            recap_match_stats['매칭건수'] = recap_stats[match_columns].sum(axis=1)
            recap_match_stats['매칭률'] = recap_match_stats['매칭건수'] / recap_match_stats['총EBD건수']
            
            # EBD/BD 개수 추가
            if not recap_counts.empty:
                recap_match_stats = recap_match_stats.join(recap_counts)
            
            # 매칭률 높은 순으로 정렬하여 상위 10개 RECAP 추출
            top_match_recap = recap_match_stats.nlargest(10, '매칭률')
            print("매칭률이 높은 상위 10개 RECAP:")
            for recap, row in top_match_recap.iterrows():
                print(f"- RECAP_PK: {recap}, 매칭률: {row['매칭률']*100:.1f}%, " 
                      f"총건수: {int(row['총EBD건수'])}, EBD: {row.get('EBD_COUNT', 'N/A')}, BD: {row.get('BD_COUNT', 'N/A')}")
            
            # 매칭률 낮은 순으로 정렬하여 하위 10개 RECAP 추출
            recap_match_stats_has_bd = recap_match_stats[recap_match_stats.get('BD_COUNT', 0) > 0]
            if not recap_match_stats_has_bd.empty:
                low_match_recap = recap_match_stats_has_bd.nsmallest(10, '매칭률')
                print("\nBD가 있지만 매칭률이 낮은 하위 10개 RECAP:")
                for recap, row in low_match_recap.iterrows():
                    print(f"- RECAP_PK: {recap}, 매칭률: {row['매칭률']*100:.1f}%, " 
                          f"총건수: {int(row['총EBD건수'])}, EBD: {row.get('EBD_COUNT', 'N/A')}, BD: {row.get('BD_COUNT', 'N/A')}")
            
            # 결과를 엑셀 파일로 저장
            os.makedirs("./verification", exist_ok=True)
            recap_match_stats.sort_values('매칭률', ascending=False).to_excel("./verification/recap_matching_stats.xlsx")
            print("\nRECAP별 매칭 통계를 './verification/recap_matching_stats.xlsx'에 저장했습니다.")

def validate_text_matching(result_df, raw_ebd_df=None, raw_bd_df=None):
    """
    텍스트 매칭 결과 검증
    원본 EBD 및 BD 데이터가 제공되면 추가 검증 수행
    """
    # 매칭된 결과만 선택
    matched_df = result_df[~result_df['MATCH_STAGE'].str.contains('미매칭')]
    
    if 'BLD_SCORE' in matched_df.columns and 'DONG_SCORE' in matched_df.columns:
        # 텍스트 점수 분포
        print("텍스트 점수 분포:")
        
        # BLD_SCORE 분포
        bld_counts = matched_df['BLD_SCORE'].value_counts().sort_index()
        print("BLD_SCORE 분포:")
        for score, count in bld_counts.items():
            print(f"- {score}점: {count}건 ({count/len(matched_df)*100:.1f}%)")
        
        # DONG_SCORE 분포
        dong_counts = matched_df['DONG_SCORE'].value_counts().sort_index()
        print("\nDONG_SCORE 분포:")
        for score, count in dong_counts.items():
            print(f"- {score}점: {count}건 ({count/len(matched_df)*100:.1f}%)")
        
        # 텍스트 점수 조합 분포
        text_combo_counts = matched_df.groupby(['BLD_SCORE', 'DONG_SCORE']).size().reset_index(name='count')
        text_combo_counts['total'] = text_combo_counts['BLD_SCORE'] + text_combo_counts['DONG_SCORE']
        text_combo_counts['percentage'] = text_combo_counts['count'] / len(matched_df) * 100
        
        print("\n텍스트 점수 조합 분포:")
        for _, row in text_combo_counts.sort_values('total', ascending=False).iterrows():
            print(f"- BLD: {row['BLD_SCORE']}점, DONG: {row['DONG_SCORE']}점 (합계: {row['total']}점): "
                  f"{row['count']}건 ({row['percentage']:.1f}%)")
    
    # 원본 데이터가 제공된 경우 추가 검증
    if raw_ebd_df is not None and raw_bd_df is not None and 'MGM_BLD_PK' in matched_df.columns:
        print("\n원본 데이터를 이용한 텍스트 매칭 검증:")
        
        # 랜덤 샘플 10개를 선택하여 원본 텍스트 비교
        sample_matches = matched_df.sample(min(10, len(matched_df)))
        
        for _, row in sample_matches.iterrows():
            if 'SEQ_NO' in row and 'MGM_BLD_PK' in row:
                ebd_row = raw_ebd_df[raw_ebd_df['SEQ_NO'] == row['SEQ_NO']].iloc[0] if not raw_ebd_df[raw_ebd_df['SEQ_NO'] == row['SEQ_NO']].empty else None
                bd_row = raw_bd_df[raw_bd_df['MGM_BLD_PK'] == row['MGM_BLD_PK']].iloc[0] if not raw_bd_df[raw_bd_df['MGM_BLD_PK'] == row['MGM_BLD_PK']].empty else None
                
                if ebd_row is not None and bd_row is not None:
                    ebd_text = f"{ebd_row.get('기관명', '')} {ebd_row.get('건축물명', '')} {ebd_row.get('주소', '')}"
                    bd_text = f"{bd_row.get('BLD_NM', '')} {bd_row.get('DONG_NM', '')}"
                    
                    print(f"\nSEQ_NO: {row['SEQ_NO']}, MGM_BLD_PK: {row['MGM_BLD_PK']}")
                    print(f"EBD: {ebd_text}")
                    print(f"BD: {bd_text}")
                    print(f"BLD_SCORE: {row.get('BLD_SCORE', 'N/A')}, DONG_SCORE: {row.get('DONG_SCORE', 'N/A')}")

def verify_date_matching(result_df):
    """
    사용승인연도 매칭 검증
    """
    # 매칭된 결과 중 사용승인연도 정보가 있는 레코드만 선택
    matched_with_date = result_df[
        (~result_df['MATCH_STAGE'].str.contains('미매칭')) & 
        (~pd.isna(result_df['사용승인연도'])) & 
        (~pd.isna(result_df['USE_DATE']))
    ]
    
    if not matched_with_date.empty:
        # 연도를 문자열로 변환하여 비교
        matched_with_date['EBD_YEAR'] = matched_with_date['사용승인연도'].astype(str).str.split('.').str[0]
        matched_with_date['BD_YEAR'] = matched_with_date['USE_DATE'].astype(str).str.split('.').str[0]
        
        # 일치 여부 확인
        matched_with_date['연도일치'] = matched_with_date['EBD_YEAR'] == matched_with_date['BD_YEAR']
        
        # 통계 출력
        match_count = matched_with_date['연도일치'].sum()
        total_count = len(matched_with_date)
        match_rate = match_count / total_count * 100
        
        print(f"사용승인연도 일치 분석 (총 {total_count}건):")
        print(f"- 일치: {match_count}건 ({match_rate:.1f}%)")
        print(f"- 불일치: {total_count - match_count}건 ({100 - match_rate:.1f}%)")
        
        # 불일치 케이스 확인
        if total_count > match_count:
            mismatched_dates = matched_with_date[~matched_with_date['연도일치']]
            print("\n연도가 일치하지 않는 케이스 (상위 10개):")
            for _, row in mismatched_dates.head(10).iterrows():
                print(f"- SEQ_NO: {row.get('SEQ_NO', 'N/A')}, EBD 연도: {row['EBD_YEAR']}, BD 연도: {row['BD_YEAR']}, "
                      f"DATE_SCORE: {row.get('DATE_SCORE', 'N/A')}, TOTAL_SCORE: {row.get('TOTAL_SCORE', 'N/A')}")
            
            # 연도 불일치 케이스 저장
            os.makedirs("./verification", exist_ok=True)
            mismatched_dates.to_excel("./verification/date_mismatched_cases.xlsx", index=False)
            print(f"\n연도가 일치하지 않는 {len(mismatched_dates)}개 케이스를 './verification/date_mismatched_cases.xlsx'에 저장했습니다.")

def validate_multiple_ebd_matching(result_df):
    """
    하나의 BD에 여러 EBD가 매칭된 케이스 분석
    """
    # 매칭된 결과만 선택
    matched_df = result_df[~result_df['MATCH_STAGE'].str.contains('미매칭')]
    
    if 'MGM_BLD_PK' in matched_df.columns:
        # 각 BD(MGM_BLD_PK)별 매칭된 EBD 개수
        bd_match_counts = matched_df['MGM_BLD_PK'].value_counts()
        
        # 복수 매칭된 BD 개수
        multiple_match_count = (bd_match_counts > 1).sum()
        
        if multiple_match_count > 0:
            print(f"하나의 BD에 여러 EBD가 매칭된 케이스: {multiple_match_count}개 BD")
            
            # 복수 매칭된 BD의 상위 10개 출력
            top_multiple = bd_match_counts[bd_match_counts > 1].nlargest(10)
            print("\n가장 많은 EBD가 매칭된 상위 10개 BD:")
            for bd_pk, count in top_multiple.items():
                print(f"- MGM_BLD_PK: {bd_pk}, 매칭된 EBD 수: {count}건")
            
            # 복수 매칭된 BD의 매칭 상세 정보
            multiple_matched_bd_pks = bd_match_counts[bd_match_counts > 1].index
            multiple_matches = matched_df[matched_df['MGM_BLD_PK'].isin(multiple_matched_bd_pks)]
            
            os.makedirs("./verification", exist_ok=True)
            multiple_matches.to_excel("./verification/multiple_ebd_matching_cases.xlsx", index=False)
            print(f"\n복수 매칭된 {len(multiple_matches)}개 케이스를 './verification/multiple_ebd_matching_cases.xlsx'에 저장했습니다.")
            
            # 복수 매칭된 케이스의 점수 분포
            if 'TOTAL_SCORE' in multiple_matches.columns:
                avg_score = multiple_matches['TOTAL_SCORE'].mean()
                min_score = multiple_matches['TOTAL_SCORE'].min()
                print(f"\n복수 매칭된 케이스의 평균 점수: {avg_score:.2f}, 최소 점수: {min_score:.2f}")

def create_verification_report(result_df, output_file="./verification/verification_report.xlsx"):
    """
    매칭 검증 보고서 생성
    """
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 매칭 단계별 통계
    match_stage_stats = pd.DataFrame(result_df['MATCH_STAGE'].value_counts()).reset_index()
    match_stage_stats.columns = ['매칭단계', '건수']
    match_stage_stats['비율(%)'] = match_stage_stats['건수'] / len(result_df) * 100
    
    # 점수 분포 (매칭된 결과만)
    score_stats = None
    if 'TOTAL_SCORE' in result_df.columns:
        matched_df = result_df[~result_df['MATCH_STAGE'].str.contains('미매칭')]
        if not matched_df.empty:
            # 점수 범위별 분포
            bins = [0, 1.8, 2.0, 2.5, 3.0, 3.8]
            labels = ['< 1.8', '1.8-2.0', '2.0-2.5', '2.5-3.0', '> 3.0']
            matched_df['점수범위'] = pd.cut(matched_df['TOTAL_SCORE'], bins=bins, labels=labels)
            score_stats = pd.DataFrame(matched_df['점수범위'].value_counts()).reset_index()
            score_stats.columns = ['점수범위', '건수']
            score_stats['비율(%)'] = score_stats['건수'] / len(matched_df) * 100
    
    # 연면적 차이 분포 (매칭된 결과만)
    area_diff_stats = None
    matched_with_area = result_df[
        (~result_df['MATCH_STAGE'].str.contains('미매칭')) & 
        (~pd.isna(result_df['연면적'])) & 
        (~pd.isna(result_df['TOTAREA']))
    ]
    
    if not matched_with_area.empty:
        # 연면적 차이 계산
        matched_with_area['면적차이비율'] = abs(matched_with_area['연면적'] - matched_with_area['TOTAREA']) / matched_with_area['연면적']
        
        # 면적 차이 분포
        bins = [0, 0.01, 0.05, 0.1, 0.2, float('inf')]
        labels = ['1% 이내', '1-5%', '5-10%', '10-20%', '20% 초과']
        matched_with_area['면적차이범위'] = pd.cut(matched_with_area['면적차이비율'], bins=bins, labels=labels)
        area_diff_stats = pd.DataFrame(matched_with_area['면적차이범위'].value_counts()).reset_index()
        area_diff_stats.columns = ['면적차이범위', '건수']
        area_diff_stats['비율(%)'] = area_diff_stats['건수'] / len(matched_with_area) * 100
    
    # 보고서 저장
    with pd.ExcelWriter(output_file) as writer:
        match_stage_stats.to_excel(writer, sheet_name='매칭단계별통계', index=False)
        
        if score_stats is not None:
            score_stats.to_excel(writer, sheet_name='점수분포', index=False)
        
        if area_diff_stats is not None:
            area_diff_stats.to_excel(writer, sheet_name='연면적차이분포', index=False)
        
        # 점수 컴포넌트 통계 (매칭된 결과만)
        if all(col in result_df.columns for col in ['AREA_SCORE', 'DATE_SCORE', 'BLD_SCORE', 'DONG_SCORE']):
            matched_df = result_df[~result_df['MATCH_STAGE'].str.contains('미매칭')]
            if not matched_df.empty:
                score_components = matched_df[['AREA_SCORE', 'DATE_SCORE', 'BLD_SCORE', 'DONG_SCORE', 'TOTAL_SCORE']].describe()
                score_components.to_excel(writer, sheet_name='점수컴포넌트통계')
    
    print(f"매칭 검증 보고서가 '{output_file}'에 저장되었습니다.")

def generate_visualizations(result_df, output_dir="./verification/visualizations"):
    """
    매칭 결과 시각화 생성
    """
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 매칭 단계 분포 파이 차트
    plt.figure(figsize=(10, 7))
    match_counts = result_df['MATCH_STAGE'].value_counts()
    plt.pie(match_counts, labels=match_counts.index, autopct='%1.1f%%')
    plt.title('매칭 단계별 분포')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/match_stage_pie.png")
    plt.close()
    
    # 매칭된 결과의 점수 분포 히스토그램
    matched_df = result_df[~result_df['MATCH_STAGE'].str.contains('미매칭')]
    if 'TOTAL_SCORE' in matched_df.columns and not matched_df.empty:
        plt.figure(figsize=(12, 7))
        sns.histplot(data=matched_df, x='TOTAL_SCORE', bins=20)
        plt.axvline(x=1.8, color='r', linestyle='--', label='최소 매칭 점수 (1.8)')
        plt.title('매칭된 결과의 총점 분포')
        plt.xlabel('총점 (TOTAL_SCORE)')
        plt.ylabel('빈도')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/total_score_histogram.png")
        plt.close()
    
    # 점수 컴포넌트 분포 (박스플롯)
    score_cols = ['AREA_SCORE', 'DATE_SCORE', 'BLD_SCORE', 'DONG_SCORE']
    if all(col in matched_df.columns for col in score_cols) and not matched_df.empty:
        plt.figure(figsize=(12, 7))
        score_data = matched_df[score_cols].melt(var_name='Score Component', value_name='Score')
        sns.boxplot(data=score_data, x='Score Component', y='Score')
        plt.title('매칭된 결과의 점수 컴포넌트 분포')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_components_boxplot.png")
        plt.close()
    
    # 연면적 차이 분포 (연면적 차이율에 따른 매칭 점수 분포)
    matched_with_area = matched_df[~pd.isna(matched_df['연면적']) & ~pd.isna(matched_df['TOTAREA'])].copy()
    if not matched_with_area.empty and 'TOTAL_SCORE' in matched_with_area.columns:
        matched_with_area['면적차이비율'] = abs(matched_with_area['연면적'] - matched_with_area['TOTAREA']) / matched_with_area['연면적']
        
        plt.figure(figsize=(12, 7))
        plt.scatter(matched_with_area['면적차이비율'] * 100, matched_with_area['TOTAL_SCORE'], alpha=0.5)
        plt.axhline(y=1.8, color='r', linestyle='--', label='최소 매칭 점수 (1.8)')
        plt.title('연면적 차이율에 따른 매칭 점수 분포')
        plt.xlabel('연면적 차이율 (%)')
        plt.ylabel('총점 (TOTAL_SCORE)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/area_diff_vs_score.png")
        plt.close()
    
    print(f"매칭 결과 시각화가 '{output_dir}' 디렉토리에 저장되었습니다.")

def extract_verification_samples(result_df, output_file="./verification/verification_samples.xlsx", sample_size=50):
    """
    검증용 샘플 추출
    매칭 단계별로 일정 수의 샘플을 추출하여 수동 검증을 위한 파일 생성
    """
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 매칭 단계 목록
    match_stages = result_df['MATCH_STAGE'].unique()
    
    all_samples = []
    
    for stage in match_stages:
        stage_df = result_df[result_df['MATCH_STAGE'] == stage]
        
        # 매칭 단계별 샘플 크기 결정 (비례 배분 또는 일정 수)
        stage_sample_size = min(sample_size, len(stage_df))
        
        if stage_sample_size > 0:
            # 랜덤 샘플 추출
            stage_samples = stage_df.sample(stage_sample_size)
            all_samples.append(stage_samples)
    
    if all_samples:
        # 모든 샘플 결합
        samples_df = pd.concat(all_samples)
        
        # 샘플에 검증 결과 컬럼 추가
        samples_df['검증결과'] = ''
        samples_df['검증자코멘트'] = ''
        
        # 엑셀 파일로 저장
        samples_df.to_excel(output_file, index=False)
        print(f"검증용 샘플 {len(samples_df)}개가 '{output_file}'에 저장되었습니다.")
        print("이 파일을 이용하여 수동 검증을 진행할 수 있습니다.")

def comprehensive_verification(result_file, bd_file=None, ebd_file=None):
    """
    종합적인 매칭 검증 수행
    """
    print("=== 매칭 결과 종합 검증 시작 ===")
    
    # 결과 파일 로드
    result_df = pd.read_excel(result_file)
    print(f"매칭 결과 파일을 로드했습니다: {len(result_df)}개 레코드")
    
    # BD 데이터 로드 (제공된 경우)
    bd_df = None
    if bd_file:
        bd_df = pd.read_excel(bd_file)
        print(f"BD 데이터 파일을 로드했습니다: {len(bd_df)}개 레코드")
    
    # EBD 데이터 로드 (제공된 경우)
    ebd_df = None
    if ebd_file:
        ebd_df = pd.read_excel(ebd_file)
        print(f"EBD 데이터 파일을 로드했습니다: {len(ebd_df)}개 레코드")
    
    print("\n1. 매칭 결과 기본 통계 분석")
    analyze_matching_results(result_df)
    
    print("\n2. 연면적 매칭 검증")
    verify_area_matching(result_df)
    
    print("\n3. 사용승인연도 매칭 검증")
    verify_date_matching(result_df)
    
    print("\n4. RECAP별 매칭 패턴 분석")
    analyze_recap_matching_patterns(result_df)
    
    print("\n5. 텍스트 매칭 결과 검증")
    validate_text_matching(result_df, ebd_df, bd_df)
    
    print("\n6. 복수 EBD 매칭 검증")
    validate_multiple_ebd_matching(result_df)
    
    print("\n7. 검증 보고서 생성")
    create_verification_report(result_df)
    
    print("\n8. 매칭 결과 시각화 생성")
    generate_visualizations(result_df)
    
    print("\n9. 검증용 샘플 추출")
    extract_verification_samples(result_df)
    
    print("\n=== 매칭 결과 종합 검증 완료 ===")

if __name__ == "__main__":
    # 검증 실행 예시
    print("EBD-BD 매칭 결과 검증 도구")
    print("사용 방법:")
    print("1. 매칭 결과 파일을 지정하여 검증 실행")
    print("   comprehensive_verification('./result/score_matching_result_ver1.xlsx')")
    print("2. BD 및 EBD 원본 데이터도 함께 제공하여 더 자세한 검증 실행")
    print("   comprehensive_verification('./result/score_matching_result_ver1.xlsx', './data/BD_data_all.xlsx', './data/EBD_data.xlsx')")
    
    # 실제 검증 실행
    # 아래 주석을 해제하고 파일 경로를 수정하여 실행하세요
    # comprehensive_verification("./result/score_matching_result_ver1.xlsx") 