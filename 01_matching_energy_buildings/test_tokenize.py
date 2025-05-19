import re

def tokenize_text_02(text):
    """
    02_text_based_EBD_BD_matching.py 스타일의 토큰화
    """
    if text is None:
        return []
    
    # 문자열로 변환
    text = str(text).lower()
    
    # 특수문자를 공백으로 치환
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 공백으로 분할하여 토큰화
    tokens = text.split()
    
    # 빈 문자열이나 공백만 있는 토큰 제거
    tokens = [token.strip() for token in tokens if token.strip()]
    
    return tokens

def tokenize_text_06(text):
    """
    06_batch_stage_EBD_BD_matching.py 스타일의 토큰화
    """
    if text is None:
        return set()
    
    # 문자열로 변환 및 소문자화
    text = str(text).lower()
    
    # 특수문자를 공백으로 치환 (02와 동일한 정규식 사용)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 공백으로 분할하여 토큰화
    tokens = text.split()
    
    # 빈 문자열이나 공백만 있는 토큰 제거
    tokens = [token.strip() for token in tokens if token.strip()]
    
    # 리스트를 집합으로 변환
    return set(tokens)

def tokenize_text_custom(text):
    """
    커스텀 토큰화 함수
    """
    if text is None:
        return set()
    
    # 문자열로 변환 및 소문자화
    text = str(text).lower()
    
    # 한글, 영문자, 숫자만 남기고 나머지는 공백으로 변환
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
    
    # 공백으로 분할하여 토큰화
    tokens = text.split()
    
    # 빈 문자열이나 공백만 있는 토큰 제거
    tokens = [token.strip() for token in tokens if token.strip()]
    
    return set(tokens)

# 테스트할 문자열
test_strings = [
    "과학기술전시체험센터",
    "과학 기술 전시 체험 센터",
    "과학기술 전시체험센터",
    "서울 과학기술전시체험센터",
    "SCIENCE 과학기술전시체험센터"
]

print("02 스타일 토큰화 결과:")
for s in test_strings:
    print(f"{s} -> {tokenize_text_02(s)}")

print("\n06 스타일 토큰화 결과 (수정 후):")
for s in test_strings:
    print(f"{s} -> {tokenize_text_06(s)}")

print("\n커스텀 토큰화 결과:")
for s in test_strings:
    print(f"{s} -> {tokenize_text_custom(s)}") 