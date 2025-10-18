"""
=============================================================================
00_run_pipeline.py - 전체 데이터 수집 파이프라인 통합 실행 스크립트
=============================================================================

실행 순서:
1. 01_generate_event_data.py           - 이벤트 & 주식 데이터 생성
2. 02_crawl_news_headlines.py    - 뉴스 헤드라인 수집
3. 03_attach_market_data.py         - 시가총액 & 섹터 정보 추가
4. 04_resolve_article_urls.py       - 실제 URL 획득 (리디렉션)
5. 05_extract_article_body.py        - 기사 본문 크롤링
6. 06_retry_failed_articles.py        - 실패한 본문 재수집

주의:
- baseline.py (LLM 예측)와 08_score_predictions.py (평가)는 별도 실행
- 한 단계 실패 시 전체 파이프라인 중단
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import glob
import pandas as pd

# =============================================================================
# 설정
# =============================================================================

# 기본 경로
BASE_DIR = Path(__file__).parent.parent  # one-vs-a-hundred 디렉토리
YUJIN_DIR = BASE_DIR / "yujin"
DATA_DIR = BASE_DIR / "data"
ARTICLES_CSV_DIR = DATA_DIR / "articles" / "csv"

# 파이프라인 단계 정의
PIPELINE_STEPS = [
    {
        "name": "01_generate_event_data",
        "script": "01_generate_event_data.py",
        "description": "S&P 100 이벤트 & 주식 데이터 생성",
        "input_files": ["../data/df_cond1_tidy.csv"],
        "output_files": ["../data/stock_data.csv", "../data/event_data.csv"],
    },
    {
        "name": "02_crawl_news_headlines",
        "script": "02_crawl_news_headlines.py",
        "description": "뉴스 헤드라인 수집 (GNews API)",
        "input_files": ["../data/event_data.csv", "../data/stock_data.csv"],
        "output_pattern": "../data/articles/csv/filtered_news_*.csv",
    },
    {
        "name": "03_attach_market_data",
        "script": "03_attach_market_data.py",
        "description": "시가총액 & 섹터 정보 추가",
        "input_pattern": "filtered_news_*.csv",
        "output_pattern": "../data/articles/csv/news_with_market_cap_*.csv",
        "needs_modification": True,
    },
    {
        "name": "04_resolve_article_urls",
        "script": "04_resolve_article_urls.py",
        "description": "실제 URL 획득 (Selenium 리디렉션)",
        "input_pattern": "news_with_market_cap_*.csv",
        "output_file": "../data/articles/csv/news_with_market_cap_with_actual_url.csv",
        "needs_modification": True,
    },
    {
        "name": "05_extract_article_body",
        "script": "05_extract_article_body.py",
        "description": "기사 본문 크롤링 (newspaper3k)",
        "input_file": "../data/articles/csv/news_with_market_cap_with_actual_url.csv",
        "output_file": "../data/articles/csv/news_with_article_body.csv",
    },
    {
        "name": "06_retry_failed_articles",
        "script": "06_retry_failed_articles.py",
        "description": "실패한 본문 재수집 (newspaper + Selenium)",
        "input_file": "../data/articles/csv/news_with_article_body.csv",
        "output_file": "../data/articles/csv/news_with_article_body_retry.csv",
    },
]

# =============================================================================
# 유틸리티 함수
# =============================================================================

def print_header(text: str):
    """섹션 헤더 출력"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_step(step_num: int, total: int, description: str):
    """단계 정보 출력"""
    print(f"\n[단계 {step_num}/{total}] {description}")
    print("-" * 80)

def find_latest_file(pattern: str, base_dir: Path = ARTICLES_CSV_DIR) -> Optional[Path]:
    """
    패턴에 맞는 가장 최신 파일 찾기

    Args:
        pattern: 파일명 패턴 (예: "filtered_news_*.csv")
        base_dir: 검색할 디렉토리

    Returns:
        가장 최신 파일 경로 or None
    """
    search_path = base_dir / pattern
    files = glob.glob(str(search_path))

    if not files:
        return None

    # 수정 시간 기준 정렬
    latest_file = max(files, key=os.path.getmtime)
    return Path(latest_file)

def verify_file_exists(filepath: Path, description: str = "파일") -> bool:
    """
    파일 존재 확인

    Args:
        filepath: 확인할 파일 경로
        description: 파일 설명 (에러 메시지용)

    Returns:
        파일 존재 여부
    """
    if not filepath.exists():
        print(f"❌ 에러: {description}을(를) 찾을 수 없습니다: {filepath}")
        return False

    print(f"✓ {description} 확인: {filepath.name}")
    return True

def run_script(script_path: Path, step_name: str) -> bool:
    """
    Python 스크립트 실행

    Args:
        script_path: 실행할 스크립트 경로
        step_name: 단계 이름 (로깅용)

    Returns:
        성공 여부
    """
    print(f"\n▶ 실행 중: {script_path.name}")
    print(f"  경로: {script_path}")

    start_time = time.time()

    try:
        # 스크립트가 위치한 디렉토리에서 실행 (상대 경로 문제 해결)
        result = subprocess.run(
            [sys.executable, script_path.name],
            cwd=script_path.parent,
            capture_output=False,  # 실시간 출력
            text=True,
            check=True,
        )

        elapsed_time = time.time() - start_time
        print(f"\n✅ 완료: {step_name} (소요 시간: {elapsed_time:.1f}초)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ 실패: {step_name} (소요 시간: {elapsed_time:.1f}초)")
        print(f"  에러 코드: {e.returncode}")
        return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ 예상치 못한 에러: {step_name}")
        print(f"  에러 메시지: {str(e)}")
        return False

def modify_script_input(script_path: Path, input_file: Path) -> bool:
    """
    스크립트의 하드코딩된 입력 파일 경로를 수정

    Args:
        script_path: 수정할 스크립트 경로
        input_file: 새로운 입력 파일 경로

    Returns:
        수정 성공 여부
    """
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 백업 생성
        backup_path = script_path.with_suffix('.py.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # 03_attach_market_data.py의 input_file 변경
        if script_path.name == "03_attach_market_data.py":
            # input_file = '../data/articles/csv/filtered_news_20250924_043743.csv'
            import re
            pattern = r"input_file\s*=\s*['\"].*?['\"]"
            replacement = f"input_file = '{input_file.relative_to(BASE_DIR)}'"
            content = re.sub(pattern, replacement, content)

        # 04_resolve_article_urls.py의 df 로드 경로 변경
        elif script_path.name == "04_resolve_article_urls.py":
            import re
            pattern = r"df\s*=\s*pd\.read_csv\(['\"].*?['\"]"
            replacement = f"df = pd.read_csv('{input_file.relative_to(BASE_DIR)}'"
            content = re.sub(pattern, replacement, content)

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  ✓ 스크립트 입력 파일 수정: {input_file.name}")
        return True

    except Exception as e:
        print(f"  ⚠ 스크립트 수정 실패: {str(e)}")
        return False

def restore_script_backup(script_path: Path):
    """스크립트 백업 복원"""
    backup_path = script_path.with_suffix('.py.bak')
    if backup_path.exists():
        try:
            backup_path.replace(script_path)
            print(f"  ✓ 백업에서 복원: {script_path.name}")
        except Exception as e:
            print(f"  ⚠ 복원 실패: {str(e)}")

# =============================================================================
# 메인 파이프라인
# =============================================================================

def run_pipeline():
    """전체 파이프라인 실행"""

    print_header("📊 데이터 수집 파이프라인 시작")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"기본 디렉토리: {BASE_DIR}")

    total_steps = len(PIPELINE_STEPS)
    pipeline_start_time = time.time()

    # 단계별 실행
    for idx, step in enumerate(PIPELINE_STEPS, 1):
        print_step(idx, total_steps, step['description'])

        script_path = YUJIN_DIR / step['script']

        # 스크립트 존재 확인
        if not script_path.exists():
            print(f"❌ 에러: 스크립트 파일을 찾을 수 없습니다: {script_path}")
            print("파이프라인을 중단합니다.")
            return False

        # 입력 파일 확인 및 수정
        if step.get('needs_modification'):
            if step['name'] == "D_attach_marketcap":
                # 가장 최신 filtered_news_*.csv 찾기
                input_file = find_latest_file(step['input_pattern'])
                if not input_file:
                    print(f"❌ 에러: 입력 파일을 찾을 수 없습니다: {step['input_pattern']}")
                    print("파이프라인을 중단합니다.")
                    return False

                print(f"  → 입력 파일 감지: {input_file.name}")
                modify_script_input(script_path, input_file)

            elif step['name'] == "CA_v2_get_actual_url":
                # 가장 최신 news_with_market_cap_*.csv 찾기
                input_file = find_latest_file(step['input_pattern'])
                if not input_file:
                    print(f"❌ 에러: 입력 파일을 찾을 수 없습니다: {step['input_pattern']}")
                    print("파이프라인을 중단합니다.")
                    return False

                print(f"  → 입력 파일 감지: {input_file.name}")
                modify_script_input(script_path, input_file)

        # 스크립트 실행
        success = run_script(script_path, step['name'])

        # 백업 복원
        if step.get('needs_modification'):
            restore_script_backup(script_path)

        if not success:
            print("\n" + "=" * 80)
            print("❌ 파이프라인 실패: 단계 실행 중 에러 발생")
            print("=" * 80)
            return False

        # 단계 간 짧은 대기
        if idx < total_steps:
            time.sleep(1)

    # 완료
    pipeline_elapsed = time.time() - pipeline_start_time
    print_header("✅ 파이프라인 완료!")
    print(f"총 소요 시간: {pipeline_elapsed:.1f}초 ({pipeline_elapsed/60:.1f}분)")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 최종 출력 파일 확인
    print("\n📁 생성된 주요 파일:")
    final_file = ARTICLES_CSV_DIR / "news_with_article_body_retry.csv"
    if final_file.exists():
        df = pd.read_csv(final_file)
        print(f"  ✓ {final_file.name}")
        print(f"    - 총 기사 수: {len(df):,}")
        print(f"    - 본문 수집 성공: {df['article_body'].notna().sum():,}")
        print(f"    - 본문 수집 실패: {df['article_body'].isna().sum():,}")

    print("\n다음 단계:")
    print("  1. baseline.py 실행으로 LLM 예측 수행")
    print("  2. 08_score_predictions.py로 예측 결과 평가
  3. 09_statistical_tests.py로 통계 분석 수행")

    return True

# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    데이터 수집 파이프라인 자동 실행                       ║
║                                                                           ║
║  이 스크립트는 다음 단계를 순차적으로 실행합니다:                         ║
║  1. 이벤트 데이터 생성                                                    ║
║  2. 뉴스 헤드라인 수집                                                    ║
║  3. 시가총액 정보 추가                                                    ║
║  4. 실제 URL 획득                                                         ║
║  5. 기사 본문 수집                                                        ║
║  6. 실패 기사 재수집                                                      ║
║                                                                           ║
║  주의: 한 단계라도 실패하면 파이프라인이 중단됩니다.                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        success = run_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
