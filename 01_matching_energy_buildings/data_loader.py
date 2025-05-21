# data_loader.py
# 이 모듈은 데이터 로드/저장만 담당. 1~10차 매칭은 main.py/batch_matcher.py에서 자동 처리됨.

import pandas as pd
import os

def load_data():
    """
    EBD와 BD 데이터 로드
    """
    print("데이터 로드 중...")
    ebd_df = pd.read_excel("./data/EBD_new_3.xlsx")
    print(f"EBD 데이터 로드: {len(ebd_df)}건")
    bd_df = pd.read_excel("./data/BD_data_all.xlsx")
    print(f"BD 데이터 로드: {len(bd_df)}건")
    return ebd_df, bd_df

def save_results(final_results, filename):
    os.makedirs("./result", exist_ok=True)
    try:
        final_results.to_excel(filename, index=False)
        print(f"\n최종 결과가 '{filename}'에 저장되었습니다.")
    except PermissionError:
        try:
            final_results.to_excel(filename, index=False)
            print(f"\n파일 권한 문제로 '{filename}'에 저장되었습니다.")
        except Exception as e:
            print(f"\n파일 저장 중 오류 발생: {e}")
            final_results.to_csv(filename.replace('.xlsx', '.csv'), index=False)
            print(f"\nCSV 형식으로 '{filename.replace('.xlsx', '.csv')}'에 저장되었습니다.") 