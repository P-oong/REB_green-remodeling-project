from data_loader import load_data, save_results
from preprocess import preprocess_data, tokenize_text
from batch_matcher import match_ebd_bd_batch

def print_matching_stats(results):
    stage_counts = results['MATCH_STAGE'].value_counts()
    total_count = len(results)
    matched_count = 0
    print("\n매칭 단계별 통계:")
    for stage, count in stage_counts.items():
        percentage = (count / total_count) * 100
        print(f"- {stage}: {count}건 ({percentage:.2f}%)")
        if not stage.startswith('미매칭'):
            matched_count += count
    match_percentage = (matched_count / total_count) * 100
    print(f"\n총 EBD: {total_count}건")
    print(f"매칭 성공: {matched_count}건 ({match_percentage:.2f}%)")
    print(f"미매칭: {total_count - matched_count}건 ({100 - match_percentage:.2f}%)")

def main():
    # 데이터 로드
    ebd_df, bd_df = load_data()
    # 데이터 전처리
    ebd_processed, bd_processed = preprocess_data(ebd_df, bd_df)
    # 배치 매칭 (1~10차 자동 매칭 포함)
    matching_results = match_ebd_bd_batch(ebd_processed, bd_processed)
    # 매칭 통계 출력 (10차까지 모두 포함)
    print_matching_stats(matching_results)
    # 원하는 컬럼 순서 정의
    desired_columns = [
        'SEQ_NO', 'RECAP_PK', '연면적', '사용승인연도', '건축물명', '주소', '지상', '지하',
        'TOTAREA', 'BLD_NM', 'DONG_NM', 'USE_DATE', 'MGM_BLD_PK', 'MATCH_STAGE',
        'EBD_COUNT', 'BD_COUNT'
    ]
    # 실제 존재하는 컬럼만 추출
    final_columns = [col for col in desired_columns if col in matching_results.columns]
    # 누락된 컬럼은 뒤에 추가
    for col in matching_results.columns:
        if col not in final_columns:
            final_columns.append(col)
    final_results = matching_results[final_columns]
    # 결과 저장 (10차 매칭까지 포함)
    save_results(final_results, './result/batch_stage_matching_result_final.xlsx')

    # ================== 테스트 코드 및 중복 검사 등은 기존대로 유지 ==================
    print("\n[디버그] 30200-100183053 RECAP_PK 매칭 추적 결과")
    target_recap = '30200-100183053'
    ebd_rows = ebd_processed[ebd_processed['RECAP_PK'] == target_recap]
    print(f"EBD 건물 개수: {len(ebd_rows)}")
    for i, ebd_row in ebd_rows.iterrows():
        print(f"\nEBD[{i}] 건축물명: {ebd_row.get('건축물명')}")
        print(f"EBD[{i}] 토큰: {tokenize_text(ebd_row.get('건축물명'))}")
    bd_rows = bd_processed[bd_processed['RECAP_PK'] == target_recap]
    print(f"\nBD 후보 개수: {len(bd_rows)}")
    for j, bd_row in bd_rows.iterrows():
        print(f"BD[{j}] DONG_NM: {bd_row.get('DONG_NM')}\nBD[{j}] DONG_NM 토큰: {tokenize_text(bd_row.get('DONG_NM'))}")
    print("\n[4,7,8차 매칭 교집합 개수 및 동점 여부]")
    for i, ebd_row in ebd_rows.iterrows():
        ebd_tokens = tokenize_text(ebd_row.get('건축물명'))
        scores = []
        for j, bd_row in bd_rows.iterrows():
            bd_tokens = tokenize_text(bd_row.get('DONG_NM'))
            inter = ebd_tokens & bd_tokens
            scores.append((j, len(inter), bd_row.get('DONG_NM'), inter))
        if scores:
            max_score = max([s[1] for s in scores])
            max_candidates = [s for s in scores if s[1] == max_score]
            print(f"EBD[{i}] 최대 교집합 개수: {max_score}, 동점 후보 수: {len(max_candidates)}")
            for s in max_candidates:
                print(f"  BD[{s[0]}] DONG_NM: {s[2]}, 교집합: {s[3]}")
        else:
            print(f"EBD[{i}] BD 후보 없음")
    print("[디버그 끝]\n")

    print("[중복 검사] MGM_BLD_PK 중복 여부 검사:")
    mgm_col = 'MGM_BLD_PK'
    if mgm_col in final_results.columns:
        mgm_series = final_results[mgm_col].dropna()
        duplicated = mgm_series[mgm_series.duplicated(keep=False)]
        if not duplicated.empty:
            print(f"중복된 MGM_BLD_PK가 {duplicated.nunique()}개 존재합니다. 전체 중복 건수: {len(duplicated)}")
            print("중복된 MGM_BLD_PK 목록:")
            print(duplicated.value_counts())
        else:
            print("MGM_BLD_PK 중복 없음: 모든 EBD에 대해 BD가 한 개씩만 붙었습니다.")
    else:
        print("MGM_BLD_PK 컬럼이 존재하지 않습니다.")

if __name__ == '__main__':
    main() 