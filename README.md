# 1 LLM vs. 100 Persona-given sLLMs

**Expert vs. Crowd Intelligence**: 단일 고성능 LLM과 100개의 페르소나 기반 소형 LLM의 주식 투자 의사결정 비교 연구

## 프로젝트 개요

이 프로젝트는 뉴스 이벤트 기반 주가 예측에서 **전문가(Expert) 모델** 1개와 **집단지성(Crowd) 모델** 100개의 성능을 비교하는 연구입니다.

- **Expert**: 단일 고성능 LLM (gpt-5)
- **Crowd**: 100개의 다양한 페르소나를 가진 소형 LLM (gpt-5-nano)
- **데이터**: S&P 100 기업의 뉴스 이벤트 및 주가 데이터
- **예측 기간**: 1일, 3일, 7일, 15일, 30일 후 주가 등락

## 빠른 시작

### 1. 환경 설정

```bash
# Poetry로 의존성 설치
poetry install

# 환경 변수 설정 (.env 파일 생성)
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

### 2. 데이터 수집 파이프라인 실행

```bash
cd yujin
python 00_run_pipeline.py
```

뉴스 수집부터 기사 본문 크롤링까지 자동으로 실행됩니다.
자세한 내용은 [yujin/README.md](yujin/README.md) 참고.

### 3. LLM 예측 수행

```bash
# 프로젝트 루트로 돌아가기
cd ..

# 기본 버전 (순차 처리)
python baseline.py

# 병렬 처리 버전 (더 빠름)
python baseline_v2.py
```

**출력**:
- `prediction_results_YYYYMMDD_HHMMSS.csv`: 예측 결과
- `prediction_failures_YYYYMMDD_HHMMSS.csv`: 실패 로그
- `checkpoint_YYYYMMDD_HHMMSS.json`: 중단 시 복구용

### 4. 결과 평가

```bash
cd yujin

# 예측 정확도 평가
python 08_score_predictions.py

# 통계적 유의성 검정
python 09_statistical_tests.py
```

**출력** (`data/answer/`):
- `scoring_comparison_*.csv`: Expert vs Crowd 비교
- `scoring_summary_*.csv`: 종합 요약
- `stats_test_results_*.csv`: 통계 검정 결과

## 프로젝트 구조

```
.
├── baseline.py                 # LLM 예측 프레임워크 (순차)
├── baseline_v2.py              # LLM 예측 프레임워크 (병렬)
├── CLAUDE.md                   # 프로젝트 상세 문서 (아키텍처, 명령어)
├── .env                        # API 키 설정
├── pyproject.toml              # Poetry 의존성 관리
│
├── yujin/                      # 데이터 처리 파이프라인
│   ├── README.md               # 파이프라인 상세 가이드
│   ├── 00_run_pipeline.py      # 파이프라인 자동 실행
│   ├── 01-06_*.py              # 데이터 수집 스크립트
│   ├── 08_score_predictions.py # 결과 채점
│   └── 09_statistical_tests.py # 통계 분석
│
└── data/
    ├── articles/csv/           # 수집된 뉴스 데이터
    ├── answer/                 # 예측 및 평가 결과
    ├── ESS11/persona.json      # 페르소나 프로필 (100개)
    └── *.csv                   # 이벤트/주식 데이터
```

## 핵심 컴포넌트

### 1. 데이터 수집 (yujin/)

뉴스 크롤링, 시가총액 추가, 본문 추출 등 데이터 준비 과정.
상세 가이드: [yujin/README.md](yujin/README.md)

### 2. LLM 예측 (baseline.py)

- Expert 에이전트: 단일 고성능 모델
- Crowd 에이전트: 100개의 페르소나 기반 모델
- 신뢰도 가중 투표로 집단지성 집계
- 체크포인트 기반 중단/재개 지원

### 3. 결과 평가 (yujin/08, 09)

- 실제 주가 데이터와 비교 (yfinance)
- 거래일 기준 날짜 보정 (미국 공휴일)
- 통계적 유의성 검정 (T-test, Mann-Whitney U-test)

## 주요 특징

- **페르소나 기반 다양성**: ESS11 설문 데이터로 생성된 100개의 현실적인 투자자 프로필
- **체크포인트 시스템**: 실행 중단 시 이어서 재개 가능
- **병렬 처리**: baseline_v2.py로 빠른 실행 (10배 속도 향상)
- **상세한 실패 로그**: 모든 예측 실패를 CSV로 기록
- **통계적 검증**: T-test, U-test로 성능 차이 유의성 검정

## 실험 설정

- **Expert 모델**: `gpt-5` (effort: medium)
- **Crowd 모델**: `gpt-5-nano` (effort: medium)
- **페르소나 수**: 100개 (ESS11 설문 기반)
- **예측 형식**: JSON (decision: up/down, confidence: 0-100, reason: 텍스트)
- **집계 방식**: 신뢰도 가중 투표 (confidence-weighted voting)

## 출력 데이터

### 예측 결과

- `prediction_results_*.csv`: expert, crowd_1~100, crowd_average 행
- `prediction_failures_*.csv`: 실패한 예측 상세 로그

### 평가 결과 (data/answer/)

- `scoring_comparison_*.csv`: 기간별 Expert vs Crowd 비교
- `scoring_summary_*.csv`: 전체 정확도 및 가중 점수
- `stats_test_results_*.csv`: 통계 검정 결과 (p-value)

## 의존성

**핵심 라이브러리**:
- `pandas`, `numpy`: 데이터 처리
- `openai`: OpenRouter API 클라이언트
- `yfinance`: 주가 데이터
- `scipy`, `scikit-learn`: 통계 분석

**데이터 수집** (yujin/):
- `selenium`: 동적 웹 크롤링
- `newspaper3k`: 기사 본문 추출
- `gnews`: 뉴스 헤드라인 수집

자세한 내용은 `pyproject.toml` 참고.

## 주의사항

- **API 키 필수**: `.env` 파일에 `OPENROUTER_API_KEY` 설정
- **ChromeDriver 필요**: Selenium 사용 스크립트 실행 시 (yujin/04, 06)
- **실행 순서**: 데이터 수집 -> 예측 -> 평가 순서로 실행
- **비용 주의**: OpenRouter API 호출 비용 발생 (특히 Crowd 100개 실행 시)

## 문서

- **CLAUDE.md**: 프로젝트 아키텍처, 명령어, 상세 설명
- **yujin/README.md**: 데이터 파이프라인 가이드
- **yujin/I_stats_test_explanation.md**: 통계 분석 설명

## 라이선스 및 기여

연구 프로젝트입니다. 문의사항은 이슈로 남겨주세요.
