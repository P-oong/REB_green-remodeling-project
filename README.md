# 한국부동산원 그린리모델링 데이터 분석 프로젝트

## 개요
한국부동산원 그린리모델링 지원 사업의 데이터 기반 의사결정을 위한 종합 데이터 분석 프로젝트입니다. 건물 에너지 성능 향상과 탄소중립 목표 달성을 위해 건물별 에너지 소비량 데이터를 체계적으로 분석하고 관리하는 시스템을 구축합니다.

- **인턴 기간**: 2025년 3월 4일 ~ 2025년 6월 23일 (4개월)

## 프로젝트 목표
- 건물별 에너지 소비량 데이터의 체계적 관리 시스템 구축
- 그린리모델링 성과 평가를 위한 데이터 기반 분석 체계 확립
- 건물 에너지 성능 향상을 위한 데이터 기반 의사결정 지원
- 탄소중립 목표 달성을 위한 건물 에너지 효율 개선 가이드라인 수립

## 작업 목록
| 번호 | 작업명 | 설명 | 주요 기술 |
|------|--------|------|------------|
| 01 | [에너지소비량-건물 매칭](./01_matching_energy_buildings) | 에너지소비량 보고 건물과 개별표제부 매칭 작업 | Python, Pandas, NLP, GPT-4 |
| 02 | [에너지소비량-계량기 매칭](./02_matching_Econsumption_meter) | 건물별 계량기 정보와 에너지 소비량 데이터 매칭 | Python, Data Validation, Time Series Analysis |
| 03 | [통합 원가명세서 자동화](./03_Integrated_cost_statement_automation) | 그린리모델링 사업 원가명세서 작성 자동화 | Python, VBA |
| 04 | [추후 작업 예정] | - 건물 에너지 성능 예측 모델 개발<br>- 실시간 모니터링 시스템 구축<br>- 에너지 효율 개선 가이드라인 수립 | - Machine Learning<br>- IoT, Real-time Processing<br>- Data Analytics |

## 주요 기술 스택
### 데이터 처리 및 분석
- Python 3.9+
- Pandas, NumPy: 데이터 처리 및 분석
- Scikit-learn: 머신러닝 및 데이터 검증
- Jupyter Notebook: 데이터 분석 및 시각화

### 자연어 처리 및 매칭
- GPT-api: 텍스트 매칭 및 데이터 검증
- NLTK, SpaCy: 텍스트 전처리 및 분석
- 정규표현식: 패턴 매칭

### 데이터 시각화
- Matplotlib, Seaborn: 기본 시각화
- Plotly: 인터랙티브 시각화
- Folium: 지리 데이터 시각화

### 데이터베이스
- SQLite: 로컬 데이터 저장
- PostgreSQL: 대용량 데이터 처리

### 문서 처리 및 자동화
- OCR (Tesseract, EasyOCR): 문서 텍스트 추출
- PDF Processing (PyPDF2, pdf2image): PDF 파일 처리
- Automation (Selenium, PyAutoGUI): 웹 자동화

## 프로젝트 구조
```bash
REB_green-remodeling-project
│
├── 00_environment/                # 공통 환경 파일 및 가이드
│   ├── environment.yml            # Conda 환경 파일
│   ├── requirements.txt           # pip 패키지 목록
│   └── README.md                  # 환경 세팅 가이드
│
├── 01_matching_energy_buildings/  # 에너지소비량-건물 매칭
│   ├── data/                      # 원본/가공 데이터
│   ├── notebooks/                 # 분석/가공 노트북
│   ├── src/                       # Python 모듈
│   ├── README.md                  # 작업 설명
│   └── result/                    # 산출물
│
├── 02_matching_Econsumption_meter/ # 에너지소비량-계량기 매칭
│   ├── data/                      # 원본/가공 데이터
│   ├── notebooks/                 # 분석/가공 노트북
│   ├── src/                       # Python 모듈
│   ├── README.md                  # 작업 설명
│   └── result/                    # 산출물
│
├── 03_Integrated_cost_statement_automation/ # 원가명세서 자동화
│   ├── data/                      # 원본/가공 데이터
│   ├── notebooks/                 # 분석/가공 노트북
│   ├── src/                       # Python 모듈
│   ├── README.md                  # 작업 설명
│   └── result/                    # 산출물
│
├── .gitignore                     # Git 무시 파일
└── README.md                      # 프로젝트 설명
```

## 환경 구축
- Conda 환경 파일: [00_environment/environment.yml](./00_environment/environment.yml)
- pip 패키지 파일: [00_environment/requirements.txt](./00_environment/requirements.txt)

```bash
# Conda 환경 설치
conda env create -f 00_environment/environment.yml

# pip 패키지 설치
pip install -r 00_environment/requirements.txt
```

## 주요 성과
1. 에너지소비량-건물 매칭
   - 룰 기반 매칭 (60%) → 텍스트 토큰 기반 매칭 (75%) → 점수 가중치 기반 매칭 (85%)
   - GPT 기반 자동화 매칭 시스템 구축
   - 건물 식별 정확도 향상

2. 에너지소비량-계량기 매칭
   - 다중 계량기 건물 처리 시스템 구축
   - 월간/연간 에너지 소비량 데이터 검증 체계 확립

3. 통합 원가명세서 자동화
   - 자동화된 원가명세서 통합 시스템 구축


## 주의 사항
- **데이터 파일(`data/`)**과 **결과 파일(`result/`)**은 `.gitignore`로 Git에 포함되지 않습니다.
- **API 키**, 비밀 설정 등은 `.env` 파일로 관리하며 Git에는 포함하지 않습니다.
- 데이터 처리 시 개인정보 보호를 위해 익명화 처리 필수

