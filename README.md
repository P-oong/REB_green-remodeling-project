# 한국부동산원 그린리모델링 데이터 분석 프로젝트

## 개요
한국부동산원 그린리모델링 지원 관련 데이터를 분석하고, 다양한 과제를 수행하기 위한 프로젝트입니다.

- **인턴 기간**: 2025년 3월 4일 ~ 2025년 6월 23일 (4개월)

## 작업 목록
| 번호 | 작업명                                           | 설명                                              |
|------|------------------------------------------------|-------------------------------------------------|
| 01   | [에너지소비량-건물 매칭](./01_matching_energy_buildings) | 에너지소비량 보고 건물과 개별표제부 매칭 작업                  |
| 02   | [추후 작업 예시](./02_other_task_example)         | 새로운 데이터 분석 작업 예시                                  |

## 프로젝트 구조
```bash
REB_green-remodeling-project
│
├── 00_environment/                # 공통 환경 파일 및 가이드
│   ├── environment.yml            # Conda 환경 파일
│   ├── requirements.txt           # pip 패키지 목록 (선택)
│   └── README.md                  # 환경 세팅 가이드
│
├── 01_matching_energy_buildings/  # (현재 진행 중인 작업) 에너지소비량-건물 매칭
│   ├── data/                      # (Git에 포함되지 않는 원본/가공 데이터)
│   ├── notebooks/                 # 분석/가공을 위한 Jupyter 노트북
│   ├── src/                       # Python 코드 (예: 데이터 처리 함수)
│   ├── README.md                  # 작업 설명 (예: 데이터 출처, 매칭 방법)
│   └── result/                    # 산출물, 가공된 데이터 (Git 제외 대상 권장)
│
├── 02_other_task_example/         # 추후 다른 작업 폴더
│   ├── data/                      # 데이터 폴더
│   ├── notebooks/                 # 노트북 폴더
│   ├── src/                       # Python 코드
│   ├── README.md                  # 작업 설명
│   └── result/                    # 결과물 저장
│
├── .gitignore                     # 전역 무시 파일
└── README.md                      # 프로젝트 전체 설명 (현재 파일)
```

## 환경 구축
- Conda 환경 파일: [00_environment/environment.yml](./00_environment/environment.yml)
- pip 패키지 파일 (선택): [00_environment/requirements.txt](./00_environment/requirements.txt)

```bash
# Conda 환경 설치
conda env create -f 00_environment/environment.yml

# (선택) pip로 설치할 경우
pip install -r 00_environment/requirements.txt
```

## 주의 사항
- **데이터 파일(`data/`)**과 **결과 파일(`result/`)**은 `.gitignore`로 Git에 포함되지 않도록 설정되어 있습니다.
- **API 키**, 비밀 설정 등은 `.env` 파일로 관리하며 Git에는 포함하지 않습니다.
```
