import pandas as pd
import os
import glob

def convert_txt_to_csv(input_file, output_file, delimiter='|'):
    """
    '|' 구분자로 된 텍스트 파일을 읽어서 CSV 파일로 변환
    
    Args:
        input_file (str): 입력 텍스트 파일 경로
        output_file (str): 출력 CSV 파일 경로
        delimiter (str): 구분자 (기본값: '|')
    """
    # txt 파일 읽기
    try:
        # 파일 인코딩 자동 감지 시도
        encodings = ['utf-8', 'euc-kr', 'cp949']
        
        for encoding in encodings:
            try:
                # 첫 줄만 읽어서 헤더 확인
                with open(input_file, 'r', encoding=encoding) as f:
                    header = f.readline().strip()
                
                # 구분자가 있는지 확인
                if delimiter in header:
                    # 파일 전체 읽기
                    df = pd.read_csv(input_file, delimiter=delimiter, encoding=encoding)
                    print(f"파일 '{input_file}'을 '{encoding}' 인코딩으로 성공적으로 읽었습니다.")
                    print(f"데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
                    break
            except UnicodeDecodeError:
                continue
        else:
            # 모든 인코딩 시도 실패
            print(f"파일 '{input_file}'을 읽을 수 있는 인코딩을 찾지 못했습니다.")
            return False
    except Exception as e:
        print(f"파일 읽기 오류: {str(e)}")
        return False
    
    # CSV 파일로 저장
    try:
        # 절대 경로로 변환
        abs_output_file = os.path.abspath(output_file)
        
        # CSV 파일로 저장
        df.to_csv(abs_output_file, index=False, encoding='utf-8-sig')
        print(f"파일이 '{abs_output_file}'로 성공적으로 변환되었습니다.")
        return True
    except Exception as e:
        print(f"파일 저장 오류: {str(e)}")
        return False


def process_all_txt_files(input_dir, output_dir, delimiter='|'):
    """
    지정된 디렉토리에 있는 모든 txt 파일을 CSV로 변환
    
    Args:
        input_dir (str): 입력 텍스트 파일이 있는 디렉토리
        output_dir (str): 출력 CSV 파일을 저장할 디렉토리
        delimiter (str): 구분자 (기본값: '|')
    """
    # 절대 경로로 변환
    abs_input_dir = os.path.abspath(input_dir)
    abs_output_dir = os.path.abspath(output_dir)
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)
    
    # 모든 txt 파일 찾기
    txt_files = glob.glob(os.path.join(abs_input_dir, "*.txt"))
    
    if not txt_files:
        print(f"'{abs_input_dir}' 디렉토리에 txt 파일이 없습니다.")
        return
    
    success_count = 0
    fail_count = 0
    
    # 각 txt 파일에 대해 변환 실행
    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        csv_file = os.path.join(abs_output_dir, os.path.splitext(file_name)[0] + '.csv')
        
        print(f"\n'{file_name}' 처리 중...")
        
        if convert_txt_to_csv(txt_file, csv_file, delimiter):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n처리 완료: {len(txt_files)}개 파일 중 {success_count}개 성공, {fail_count}개 실패")


if __name__ == "__main__":
    # 지정된 결과 디렉토리 설정
    output_dir = r"C:\Users\HyeonPoong Lee\REB_GR\REB_green-remodeling-project\02_matching_Econsumption_meter\result"
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 두 개의 특정 txt 파일 지정 (현재 작업 디렉토리 기준)
    txt_files = [
        "../data/사용량정보_23.txt"
    ]
    
    # 각 파일 CSV로 변환
    for txt_file in txt_files:
        # 상대 경로를 절대 경로로 변환
        abs_txt_file = os.path.abspath(txt_file)
        
        if os.path.exists(abs_txt_file):
            # 출력 파일명 생성 (확장자를 .csv로 변경)
            file_name = os.path.basename(abs_txt_file)
            csv_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.csv')
            
            print(f"\n'{file_name}' 처리 중...")
            convert_txt_to_csv(abs_txt_file, csv_file)
        else:
            print(f"'{abs_txt_file}' 파일이 존재하지 않습니다.")
    
    print("\n변환 작업 완료") 