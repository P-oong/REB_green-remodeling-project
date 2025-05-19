import pandas as pd

def safe_equals(a, b, strict=False):
    if pd.isna(a) and pd.isna(b):
        return True
    elif pd.isna(a) or pd.isna(b):
        return False
    if isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if strict:
            return a == b
        else:
            return abs(a - b) < 0.01
    return a == b

def is_within_percentage(a, b, percentage):
    if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
        return False
    min_val = min(a, b)
    max_val = max(a, b)
    diff_percentage = ((max_val - min_val) / min_val) * 100
    return diff_percentage <= percentage

def find_unique_matches(possible_matches):
    from collections import defaultdict
    # BD -> EBD 매핑
    bd_to_ebds = defaultdict(list)
    for ebd_idx, bd_pk in possible_matches:
        bd_to_ebds[bd_pk].append(ebd_idx)
    
    # 각 BD에 대해 첫 번째 EBD만 선택
    confirmed = []
    for bd_pk, ebd_list in bd_to_ebds.items():
        if ebd_list:
            confirmed.append((ebd_list[0], bd_pk))
    
    return confirmed

# check_match_condition: 1~9차 매칭 조건만 처리 (10차는 batch_matcher.py에서 RECAP 단위 1:1로 처리)
def check_match_condition(ebd_row, bd_row, step):
    if step == 1:
        return safe_equals(ebd_row['연면적'], bd_row['TOTAREA'], strict=True) and safe_equals(ebd_row['사용승인연도'], bd_row['USE_DATE'], strict=True)
    elif step == 2:
        return safe_equals(ebd_row['연면적'], bd_row['TOTAREA'], strict=True)
    elif step == 3:
        return abs(ebd_row['연면적'] - bd_row['TOTAREA']) <= 0.01 and safe_equals(ebd_row['사용승인연도'], bd_row['USE_DATE'], strict=True)
    elif step == 4:
        # 연면적 ±1% 이내 + 텍스트 토큰 교집합 개수 반환
        area_match = abs(ebd_row['연면적'] - bd_row['TOTAREA']) / max(ebd_row['연면적'], 1) <= 0.01
        if not area_match:
            return 0
        ebd_tokens = ebd_row.get('ebd_tokens', set())
        dong_tokens = bd_row.get('dong_tokens', set())
        return len(ebd_tokens & dong_tokens)
    elif step == 5:
        return abs(ebd_row['연면적'] - bd_row['TOTAREA']) / max(ebd_row['연면적'], 1) <= 0.01
    elif step == 6:
        area_match = abs(ebd_row['연면적'] - bd_row['TOTAREA']) / max(ebd_row['연면적'], 1) <= 0.05
        year_match = safe_equals(ebd_row['사용승인연도'], bd_row['USE_DATE'], strict=True)
        return area_match and year_match
    elif step == 7:
        area_match = abs(ebd_row['연면적'] - bd_row['TOTAREA']) / max(ebd_row['연면적'], 1) <= 0.05
        if not area_match:
            return 0
        ebd_tokens = ebd_row.get('ebd_tokens', set())
        dong_tokens = bd_row.get('dong_tokens', set())
        return len(ebd_tokens & dong_tokens)
    elif step == 8:
        ebd_tokens = ebd_row.get('ebd_tokens', set())
        dong_tokens = bd_row.get('dong_tokens', set())
        return len(ebd_tokens & dong_tokens)
    elif step == 9:
        return abs(ebd_row['연면적'] - bd_row['TOTAREA']) / max(ebd_row['연면적'], 1) <= 0.05
    # 10차는 batch_matcher.py에서 처리
    return False 