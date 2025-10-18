# yujin/ Directory - Data Processing Pipeline

이 디렉토리는 뉴스 데이터 수집부터 LLM 예측 결과 평가까지의 전체 파이프라인을 담고 있습니다.

## 파일 구조

### 핵심 파이프라인 (실행 순서)

```
00_run_pipeline.py          -> 전체 파이프라인 자동 실행 (단계 1-6)
├── 01_generate_event_data.py     -> S&P 100 이벤트 데이터 생성
├── 02_crawl_news_headlines.py    -> 뉴스 헤드라인 수집 (GNews API)
├── 03_attach_market_data.py      -> 시가총액/섹터 정보 추가
├── 04_resolve_article_urls.py    -> 실제 URL 획득 (Selenium)
├── 05_extract_article_body.py    -> 기사 본문 크롤링
└── 06_retry_failed_articles.py   -> 실패한 기사 재수집

baseline.py (루트)          -> LLM 에이전트 예측 수행

08_score_predictions.py     -> 예측 결과 채점 및 평가
09_statistical_tests.py     -> 통계적 유의성 검정
```

### 유틸리티

- **07_format_agent_data.py**: 데이터 포맷팅 함수 (baseline.py에 통합됨)

### 아카이브

`_archived/` 디렉토리:
- `crawl_news_basic.py`: 기본 뉴스 크롤러 (02번으로 대체)
- `resolve_urls_v1.py`: 구버전 URL 해결 스크립트 (04번으로 대체)
- `call_openrouter.py`: OpenRouter API 예제 (baseline.py로 대체)
- `check_crawling.py`: 미완성 검증 스크립트
- `result_refinement.py`: 레거시 결과 처리

## 빠른 시작

### 1. 데이터 수집 파이프라인 실행

```bash
cd yujin
python 00_run_pipeline.py
```

이 명령어는 다음을 순차적으로 실행합니다:
1. 이벤트 데이터 생성
2. 뉴스 헤드라인 수집
3. 시가총액 정보 추가
4. 실제 URL 획득
5. 기사 본문 크롤링
6. 실패 기사 재수집

**출력**: `data/articles/csv/news_with_article_body_retry.csv`

### 2. LLM 예측 실행

```bash
cd ..  # 프로젝트 루트로 이동
python baseline.py
```

**출력**:
- `prediction_results_YYYYMMDD_HHMMSS.csv`
- `prediction_failures_YYYYMMDD_HHMMSS.csv`
- `checkpoint_YYYYMMDD_HHMMSS.json` (실행 중단 시)

### 3. 결과 평가

```bash
cd yujin
python 08_score_predictions.py
```

**출력** (`data/answer/`):
- `scoring_results_detailed_*.csv`: 개별 예측 상세 결과
- `scoring_results_crowd_*.csv`: 집단지성 집계 결과
- `scoring_comparison_*.csv`: 기간별 Expert vs Crowd 비교
- `scoring_summary_*.csv`: 종합 요약

### 4. 통계 분석

```bash
python 09_statistical_tests.py
```

**출력** (`data/answer/`):
- `classification_metrics_*.csv`: Precision, Recall, F1-score
- `confidence_performance_*.csv`: 신뢰도 분석
- `stats_test_results_*.csv`: T-test, Mann-Whitney U-test 결과

## 파일 네이밍 규칙

- **00-09**: 실행 순서를 나타내는 번호
  - `00`: 오케스트레이션 스크립트
  - `01-06`: 데이터 수집 파이프라인
  - `07`: 유틸리티
  - `08-09`: 평가 및 분석
- **_archived/**: 더 이상 사용하지 않는 스크립트

## 주요 의존성

- **pandas**: 데이터 처리
- **yfinance**: 주가 및 시가총액 데이터
- **selenium**: 동적 웹 크롤링
- **newspaper3k**: 기사 본문 추출
- **gnews**: 뉴스 헤드라인 수집
- **scipy**: 통계 테스트
- **scikit-learn**: 분류 메트릭

## 주의사항

1. **00_run_pipeline.py** 실행 시 한 단계라도 실패하면 전체가 중단됩니다
2. **08_score_predictions.py** 실행 전에 반드시 `baseline.py`로 예측 결과를 생성해야 합니다
3. **Selenium** 사용 스크립트(04, 06)는 ChromeDriver가 필요합니다
4. **API 키** 필요: OpenRouter API (`.env` 파일에 `OPENROUTER_API_KEY` 설정)
