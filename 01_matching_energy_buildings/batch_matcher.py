from matching_rules import check_match_condition, find_unique_matches
import pandas as pd
from tqdm import tqdm

def match_ebd_bd_batch(ebd_df, bd_df):
    print("단계별 일괄 EBD-BD 매칭 시작...")
    start_time = pd.Timestamp.now()
    results = ebd_df.copy()
    results['MATCH_STAGE'] = '미매칭'
    results['MGM_BLD_PK'] = None
    results['TOTAREA'] = None
    results['BLD_NM'] = None
    results['DONG_NM'] = None
    results['USE_DATE'] = None
    ebd_counts = ebd_df.groupby('RECAP_PK').size().to_dict()
    bd_counts = bd_df.groupby('RECAP_PK').size().to_dict()
    unmatched_ebd_indices = set(ebd_df.index)
    unmatched_bd_pks = set(bd_df['MGM_BLD_PK'])
    for step in tqdm(range(1, 10), desc="단계별 매칭"):
        print(f"\n{step}차 매칭 시작...")
        possible_matches = []
        to_remove_ebd_indices = []
        for i in tqdm(list(unmatched_ebd_indices), desc=f"{step}차 EBD 평가", leave=False):
            ebd_row = ebd_df.loc[i]
            if pd.isna(ebd_row['RECAP_PK']):
                results.loc[i, 'MATCH_STAGE'] = '미매칭(RECAP없음)'
                to_remove_ebd_indices.append(i)
                continue
            bd_candidates = bd_df[(bd_df['RECAP_PK'] == ebd_row['RECAP_PK']) & (bd_df['MGM_BLD_PK'].isin(unmatched_bd_pks))]
            if bd_candidates.empty:
                if results.loc[i, 'MATCH_STAGE'] == '미매칭':
                    results.loc[i, 'MATCH_STAGE'] = '미매칭(후보없음)'
                continue
            if step in [4, 7, 8]:
                candidate_scores = []
                for _, bd_row in bd_candidates.iterrows():
                    score = check_match_condition(ebd_row, bd_row, step)
                    if score > 0:
                        candidate_scores.append((bd_row['MGM_BLD_PK'], score))
                if not candidate_scores:
                    continue
                max_score = max([s for _, s in candidate_scores])
                if max_score == 0:
                    continue
                max_score_candidates = [pk for pk, s in candidate_scores if s == max_score]
                if len(max_score_candidates) == 1:
                    possible_matches.append((i, max_score_candidates[0]))
            else:
                for _, bd_row in bd_candidates.iterrows():
                    if check_match_condition(ebd_row, bd_row, step):
                        possible_matches.append((i, bd_row['MGM_BLD_PK']))
        for idx in to_remove_ebd_indices:
            unmatched_ebd_indices.remove(idx)
        confirmed_matches = find_unique_matches(possible_matches)
        print(f"{step}차 가능한 매칭: {len(possible_matches)}건, 확정된 매칭: {len(confirmed_matches)}건")
        for ebd_idx, bd_pk in confirmed_matches:
            bd_row = bd_df[bd_df['MGM_BLD_PK'] == bd_pk].iloc[0]
            results.loc[ebd_idx, 'MGM_BLD_PK'] = bd_pk
            results.loc[ebd_idx, 'TOTAREA'] = bd_row['TOTAREA']
            results.loc[ebd_idx, 'BLD_NM'] = bd_row['BLD_NM']
            results.loc[ebd_idx, 'DONG_NM'] = bd_row['DONG_NM']
            results.loc[ebd_idx, 'USE_DATE'] = bd_row['USE_DATE']
            results.loc[ebd_idx, 'MATCH_STAGE'] = f'{step}차'
        to_remove_ebd_indices = [ebd_idx for ebd_idx, _ in confirmed_matches]
        to_remove_bd_pks = [bd_pk for _, bd_pk in confirmed_matches]
        for idx in to_remove_ebd_indices:
            if idx in unmatched_ebd_indices:
                unmatched_ebd_indices.remove(idx)
        for pk in to_remove_bd_pks:
            if pk in unmatched_bd_pks:
                unmatched_bd_pks.remove(pk)
    # 10차 매칭: RECAP 단위로 EBD와 BD가 각각 1건씩인 경우
    print("\n10차 매칭 시작 (RECAP 단위 1:1 매칭)...")
    unmatched_ebd_by_recap = {}
    for i in unmatched_ebd_indices:
        ebd_row = ebd_df.loc[i]
        recap_pk = ebd_row['RECAP_PK']
        if pd.isna(recap_pk):
            continue
        if recap_pk not in unmatched_ebd_by_recap:
            unmatched_ebd_by_recap[recap_pk] = []
        unmatched_ebd_by_recap[recap_pk].append(i)
    unmatched_bd_by_recap = {}
    for pk in unmatched_bd_pks:
        bd_row = bd_df[bd_df['MGM_BLD_PK'] == pk].iloc[0]
        recap_pk = bd_row['RECAP_PK']
        if pd.isna(recap_pk):
            continue
        if recap_pk not in unmatched_bd_by_recap:
            unmatched_bd_by_recap[recap_pk] = []
        unmatched_bd_by_recap[recap_pk].append(pk)
    tenth_stage_count = 0
    to_remove_ebd_indices = []
    to_remove_bd_pks = []
    for recap_pk in unmatched_ebd_by_recap.keys():
        if recap_pk in unmatched_bd_by_recap and len(unmatched_ebd_by_recap[recap_pk]) == 1 and len(unmatched_bd_by_recap[recap_pk]) == 1:
            ebd_idx = unmatched_ebd_by_recap[recap_pk][0]
            bd_pk = unmatched_bd_by_recap[recap_pk][0]
            bd_row = bd_df[bd_df['MGM_BLD_PK'] == bd_pk].iloc[0]
            results.loc[ebd_idx, 'MGM_BLD_PK'] = bd_pk
            results.loc[ebd_idx, 'TOTAREA'] = bd_row['TOTAREA']
            results.loc[ebd_idx, 'BLD_NM'] = bd_row['BLD_NM']
            results.loc[ebd_idx, 'DONG_NM'] = bd_row['DONG_NM']
            results.loc[ebd_idx, 'USE_DATE'] = bd_row['USE_DATE']
            results.loc[ebd_idx, 'MATCH_STAGE'] = '10차'
            to_remove_ebd_indices.append(ebd_idx)
            to_remove_bd_pks.append(bd_pk)
            tenth_stage_count += 1
    print(f"10차 매칭 완료: {tenth_stage_count}건")
    return results 