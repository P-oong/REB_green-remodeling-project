# 에너지보고대상 건물과 건축물대장 매칭

## 개요

이 디렉토리는 에너지보고대상(EBD) 건물과 건축물대장(BD) 표제부 데이터를 매칭하는 알고리즘을 구현한 코드를 포함합니다. 매칭은 여러 단계로 진행되며, 단계별로 다양한 조건을 적용하여 정확도를 높였습니다.

### 매칭 작업의 어려움

에너지보고대상 건물과 건축물대장의 개별 표제부를 매칭하는 작업은 다음과 같은 어려움이 있습니다:

- **데이터 구조의 차이**: 건축물대장은 총괄표제부와 개별표제부로 분리되어 있으나, 에너지 보고서는 단일 건물 단위로 작성됨
- **식별 변수 부재**: 두 데이터 간 공통적인 식별자(ID)가 없어 직접적인 매칭이 불가능
- **다중 건물 문제**: 하나의 총괄표제부 내에 여러 개별표제부가 존재하며, 에너지 보고대상 건물이 정확히 어떤 개별표제부에 해당하는지 식별하기 어려움
- **대규모 데이터**: 다량의 건물 데이터를 수작업으로 매칭하는 것은 인력과 시간이 과도하게 소요됨
- **데이터 불일치**: 건물명, 주소, 면적 등의 정보가 두 데이터 소스 간에 불일치하는 경우가 많음

### 매칭 작업의 중요성

이러한 어려움에도 불구하고 매칭 작업은 다음과 같은 이유로 매우 중요합니다:

- **정확한 에너지 계량기 DB 구축**: 건물별 에너지 소비량을 정확히 파악하기 위한 기초 데이터 확보
- **그린리모델링 대상 선정**: 효과적인 그린리모델링 사업 대상 건물 선정을 위한 정확한 건물 정보 필요
- **성과 평가 기반 마련**: 그린리모델링 전후 성과를 정량적으로 평가하기 위한 기준점 설정
- **데이터 기반 의사결정**: 시계열 모델링, 에너지 시뮬레이션 등을 통한 효과적인 의사결정 지원
- **탄소중립 기여**: 정확한 건물 에너지 소비 데이터를 통해 탄소중립 정책의 효과적 이행 지원

이러한 매칭 작업의 자동화는 인력과 시간을 크게 절약하면서도 높은 정확도를 유지하여 그린리모델링 사업의 성공적 수행에 기여합니다.

## 사용 데이터

- `./data/EBD_new_3.xlsx`: 에너지보고대상 건물 데이터
- `./data/BD_data_all.xlsx`: 건축물대장 표제부 데이터

## 주요 파일 설명

- **01_EBD_BD_rule_matching.py**: 초기 규칙 기반 매칭 (1~3차)
  - 연면적과 사용승인연도를 기준으로 한 기본 매칭 로직
  - 계층적 매칭 단계 적용 (완전 일치 → 연면적 일치 → 1:1 구조 매칭)

- **02_text_based_EBD_BD_matching.py**: 텍스트 기반 매칭 추가 (4차)
  - 건축물명과 주소 토큰화 기반 텍스트 매칭
  - 기존 미매칭 데이터에 대해 추가 매칭 시도

- **05_multi_stage_EBD_BD_matching.py**: 최종 통합 10단계 매칭 구현
  - 종합적인 10단계 매칭 알고리즘 구현
  - 연면적 오차 범위 확대 및 텍스트 토큰 매칭 조합
  - 자동 매칭 후 수동 매칭을 위한 데이터 준비

- **GPT 기반 자동화 매칭 실험**:
  - 프롬프트 엔지니어링을 통한 매칭 자동화 시도
  - 인간 추론이 필요한 복잡한 케이스 처리
  - 텍스트 전처리 및 모델링 작업에서 LLM 활용 가능성 검증
  - 정형 데이터 처리에서는 규칙 기반 전처리 로직이 더 우수한 성능을 보여 최종 적용하지 않음

## 매칭 단계 및 결과

### 자동 매칭 단계 (1~10단계)

| 단계 | 조건 설명 | 건수 | 비율 |
|------|-----------|------|------|
| 1차 | 연면적 완전 일치 & 사용승인연도 일치 | 1,633건 | 38.94% |
| 2차 | 연면적 완전 일치 | 145건 | 3.46% |
| 3차 | 연면적 ±1% 이내 & 사용승인연도 일치 | 732건 | 17.45% |
| 4차 | 연면적 ±1% 이내 & (동명칭 토큰 매칭 또는 동명칭 NaN 시 건축물명 토큰 매칭) | 66건 | 1.57% |
| 5차 | 연면적 ±1% 이내 | 69건 | 1.65% |
| 6차 | 연면적 ±5% 이내 & 사용승인연도 일치 | 223건 | 5.32% |
| 7차 | 연면적 ±5% 이내 & (동명칭 토큰 매칭 또는 동명칭 NaN 시 건축물명 토큰 매칭) | 24건 | 0.57% |
| 8차 | 연면적 ±5% 이내 | 51건 | 1.22% |
| 9차 | (연면적·연도 무관) 텍스트 토큰 매칭만 | 226건 | 5.39% |
| 10차 | 해당 RECAP_PK 내 EBD·BD가 각각 1건만 존재하는 1:1 구조적 매칭 | 250건 | 5.96% |
| 미매칭 (RECAP 없음) | 동일 RECAP_PK를 가진 BD 후보가 아예 존재하지 않음 | 178건 | 4.24% |
| 미매칭 (조건 불충족) | 1~10차 모든 조건을 거쳤으나 매칭되지 않음 | 597건 | 14.23% |

### 수동 매칭 단계 (11~12단계)

- **11단계: 건축HUB 조회 매칭**
  - 미매칭 775건(RECAP 없음 178건 + 조건 불충족 597건)을 건축물대장 HUB 포털에서 수동 매칭 시도
  - 주소 보정, 지도 조회, 건물명 검색 등 다양한 방법 활용
  - [`04_건축HUB_API_matching`](../04_건축HUB_API_matching) 폴더의 API 활용 스크립트를 통한 수동 매칭 지원
    - 건축HUB Open API를 활용한 효율적인 건축물 정보 조회
    - 대량 API 호출 자동화 스크립트로 수동 매칭 작업 효율화

- **12단계: 최종 미매칭**
  - 11단계 이후에도 매칭되지 않은 건물을 최종 미매칭으로 분류

## 텍스트 전처리 방식

1. **EBD 텍스트 토큰**:
   - 건축물명과 주소 컬럼에서 특수문자 제거
   - 공백 기준으로 토큰화
   - 주소의 앞 3개 토큰(행정구역명 등) 제외
   - 중복 제거된 통합 토큰 집합 생성

2. **BD 텍스트 토큰**:
   - 건축물명(BLD_NM)과 동명칭(DONG_NM) 각각 토큰화
   - 특수문자 제거 및 공백 기준 분리

## GPT 기반 매칭 실험

데이터 매칭 과정에서 LLM(Large Language Model)의 활용 가능성을 검증하기 위해 GPT 기반 자동화 매칭 실험을 진행했습니다:

- **프롬프트 엔지니어링**:
  - 건축물명, 주소, 면적 등 건물 특성 정보를 포함한 프롬프트 설계
  - 다양한 방식의 유사도 판단 지시문 실험

- **실험 결과 분석**:
  - 복잡한 텍스트 비교에서는 LLM이 인간과 유사한 추론 방식 보임
  - 주소 변경, 건물명 변경 등 변형이 있는 케이스에서 적응력 확인
  - 그러나 정형화된 규칙 기반 매칭에 비해 일관성과 정확도가 떨어짐

- **한계점**:
  - API 비용 및 처리 시간 증가
  - 대규모 데이터셋 처리 시 스케일링 문제
  - 결과 예측 및 재현성 부족

이러한 실험을 통해 LLM은 텍스트 전처리 및 특수 케이스 처리에 보조적으로 활용할 수 있지만, 정형화된 데이터 매칭에서는 기존의 규칙 기반 알고리즘이 더 효율적임을 확인했습니다.

## 실행 방법

최종 매칭 알고리즘 실행:
```bash
python 05_multi_stage_EBD_BD_matching.py
```

## 결과 파일

- `./result/multi_stage_matching_result.xlsx`: 최종 매칭 결과
  - `MATCH_STAGE` 컬럼에 매칭 단계 기록
  - 성공 매칭의 경우 `MGM_BLD_PK`에 건축물대장 고유 ID 저장
  - `EBD_COUNT`, `BD_COUNT` 컬럼에 RECAP_PK별 개수 정보 제공

## 매칭 성공률

- **자동 매칭 성공률**: 81.53% (3,419건)
- **미매칭 비율**: 18.47% (775건)

## 후속 작업

- 11단계 수동 매칭 작업 진행
- 최종 매칭 결과 검증 및 통계 분석

