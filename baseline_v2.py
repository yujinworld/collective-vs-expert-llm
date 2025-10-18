import json
import os
import random
from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Optional, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd
import time
import csv
import threading

# .env 파일에서 환경 변수 로드 (API 저장 변수 = OPENROUTER_API_KEY)
load_dotenv()

# --- 응답 형식 정의 ---
class PeriodPrediction(TypedDict):
    decision: str # 오를지 내릴지
    confidence: int # 신뢰도
    reason: Optional[str] # API 응답에 포함되는 reason 필드 처리를 위해 추가

class AgentPrediction(TypedDict):
    day_1: PeriodPrediction
    day_3: PeriodPrediction
    day_7: PeriodPrediction
    day_15: PeriodPrediction
    day_30: PeriodPrediction

# --- 페르소나 데이터 로딩 함수 ---
def load_persona_data(file_path: str) -> List[Dict]:
    """페르소나 데이터를 로딩합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"페르소나 데이터 로딩 오류: {e}")
        return []

def select_random_personas(persona_data: List[Dict], num_personas: int) -> List[Dict]:
    """페르소나 데이터에서 랜덤하게 선택합니다."""
    return random.sample(persona_data, min(num_personas, len(persona_data)))

def format_persona_profile(persona: Dict) -> str:
    """페르소나 프로필을 문자열로 포맷팅합니다."""
    profile = persona.get("PERSONA_PROFILE", {})
    
    identity = profile.get("Identity", {})
    socio = profile.get("Socio_Economics", {})
    outlook = profile.get("Outlook_and_Trust", {})
    info_habit = profile.get("Information_Habit", {})
    
    persona_text = f"""당신은 다음 페르소나를 가진 투자자입니다:
- 국적: {identity.get('Nationality', 'Unknown')}
- 나이: {identity.get('Age', 'Unknown')}
- 성별: {identity.get('Gender', 'Unknown')}
- 교육수준: {identity.get('Education', 'Unknown')}
- 직업: {socio.get('Occupation', 'Unknown')}
- 소득수준: {socio.get('Income_Decile', 'Unknown')}
- 정치성향: {socio.get('Left_Right_Position', 'Unknown')}
- 전반적 행복도: {outlook.get('Overall_Happiness', 'Unknown')}
- 국가경제 전망: {outlook.get('Natl_Econ_View', 'Unknown')}
- 가계재정 전망: {outlook.get('House_Fin_View', 'Unknown')}
- 제도신뢰도: {outlook.get('Inst_Trust', 'Unknown')}
- 대인신뢰도: {outlook.get('Interpersonal_Trust', 'Unknown')}
- 정치관심도: {info_habit.get('Political_Interest', 'Unknown')}
- 뉴스빈도: {info_habit.get('News_Frequency', 'Unknown')}
- 인터넷사용빈도: {info_habit.get('Internet_Frequency', 'Unknown')}"""
    
    return persona_text

# --- 프롬프트 생성 함수 ---
def create_stock_prompt(stock_data: str, persona_profile: Optional[str] = None, is_expert: bool = False) -> str:
    """주식 예측을 위한 표준 프롬프트를 생성합니다."""
    # 다중 기간 주가 예측을 위한 프롬프트
    base_prompt = (
        "Based on the provided stock news data (including article titles, descriptions, and full article content), predict whether the stock price will go 'up' or 'down' for different time periods: 1 day, 3 days, 7 days, 15 days, and 30 days from the data date. "
        "Consider all available information: news titles, descriptions, and detailed article content to make informed predictions. "
        "For each period, provide your confidence level as a single integer from 0 to 100 and a brief reason for your prediction. "
        "Your response MUST be a JSON object with five keys: 'day_1', 'day_3', 'day_7', 'day_15', 'day_30'. "
        "Each period should have 'decision' (up/down), 'confidence' (0-100), and 'reason' (brief explanation).\n\n"
        "Example: {\"day_1\": {\"decision\": \"up\", \"confidence\": 75, \"reason\": \"Strong Q4 results\"}, \"day_3\": {\"decision\": \"down\", \"confidence\": 60, \"reason\": \"Market volatility\"}, ...}\n\n"
        "Respond ONLY with a valid JSON object. Do not include any extra text.\n\n"
    )

    if is_expert:
        expert_prefix = "You are an expert financial analyst with deep knowledge of market trends, technical analysis, and fundamental analysis. "
        base_prompt = expert_prefix + base_prompt
    elif persona_profile:
        base_prompt = f"{persona_profile}\n\n{base_prompt}"

    return f"{base_prompt}Data: {stock_data}"

# --- 주식 심볼 추출 함수 ---
def extract_stock_symbol(stock_data: str) -> str:
    """stock_data에서 주식 심볼을 추출합니다."""
    try:
        data_dict = json.loads(stock_data)
        # descriptions나 titles에서 NASDAQ:SYMBOL 형태를 찾아 추출
        text_to_search = data_dict.get('descriptions', '') + ' ' + data_dict.get('titles', '')
        
        import re
        # NASDAQ:SYMBOL 패턴 찾기
        nasdaq_match = re.search(r'NASDAQ:([A-Z]+)', text_to_search)
        if nasdaq_match:
            return nasdaq_match.group(1)
        
        # 괄호 안의 심볼 찾기 (ADBE) 형태
        paren_match = re.search(r'\(([A-Z]{2,5})\)', text_to_search)
        if paren_match:
            return paren_match.group(1)
            
        return "UNKNOWN"
    except:
        return "UNKNOWN"

# --- 날짜 계산 함수 ---
def calculate_prediction_dates(search_date: str) -> Dict[str, str]:
    """search_date를 기준으로 각 예측 기간의 정확한 날짜를 계산합니다."""
    try:
        base_date = datetime.strptime(search_date, '%Y-%m-%d')

        prediction_dates = {}
        periods = {
            'day_1': 1,
            'day_3': 3,
            'day_7': 7,
            'day_15': 15,
            'day_30': 30
        }

        for period_name, days in periods.items():
            prediction_date = base_date + timedelta(days=days)
            prediction_dates[period_name] = prediction_date.strftime('%Y-%m-%d')

        return prediction_dates
    except Exception as e:
        print(f"날짜 계산 오류: {e}")
        # 오류 발생 시 기본값 반환
        return {
            'day_1': 'UNKNOWN',
            'day_3': 'UNKNOWN',
            'day_7': 'UNKNOWN',
            'day_15': 'UNKNOWN',
            'day_30': 'UNKNOWN'
        }

# --- CSV 데이터 로딩 함수 ---
def load_events_from_csv(csv_path: str, symbols: Optional[List[str]] = None, dates: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    CSV 파일에서 이벤트 목록(symbol, search_date 조합)을 추출합니다.

    Args:
        csv_path: CSV 파일 경로
        symbols: 필터링할 종목 리스트 (None이면 전체)
        dates: 필터링할 날짜 리스트 (None이면 전체)

    Returns:
        [(symbol, search_date), ...] 형태의 유니크한 이벤트 튜플 리스트
    """
    try:
        df = pd.read_csv(csv_path)

        # 필터링
        if symbols is not None:
            df = df[df['symbol'].isin(symbols)]
        if dates is not None:
            df = df[df['search_date'].isin(dates)]

        # (symbol, search_date) 조합의 유니크한 값만 추출
        events = df[['symbol', 'search_date']].drop_duplicates().values.tolist()

        return [(str(symbol), str(date)) for symbol, date in events]
    except Exception as e:
        print(f"CSV 로딩 오류: {e}")
        return []

def convert_df_to_agent_format(csv_path: str, symbol: str, search_date: str) -> Dict[str, str]:
    """
    CSV 파일을 읽어서 특정 symbol과 search_date의 데이터를 LLM Agent에 전달할 수 있는 딕셔너리 형태로 변환
    (yujin/E_data_form_making.py의 함수를 baseline.py에 통합)

    Args:
        csv_path: CSV 파일 경로
        symbol: 필터링할 종목 심볼 (예: 'AAPL', 'MSFT')
        search_date: 필터링할 검색 날짜 (예: '2024-03-14')

    Returns:
        dict: LLM Agent에 전달할 데이터 딕셔너리
            - symbol: 종목 심볼
            - search_date: 검색 날짜
            - titles: title 열의 모든 값을 ' / '로 연결한 문자열
            - descriptions: description 열의 모든 값을 ' / '로 연결한 문자열
            - article_body: article_body 열의 모든 값을 ' / '로 연결한 문자열
            - sector: 섹터 정보 (없으면 빈 문자열)
    """
    try:
        df = pd.read_csv(csv_path)

        # symbol과 search_date로 필터링
        df_filtered = df[(df['symbol'] == symbol) & (df['search_date'] == search_date)]

        # 데이터가 없는 경우 처리
        if len(df_filtered) == 0:
            return {
                'symbol': symbol,
                'search_date': search_date,
                'titles': "",
                'descriptions': "",
                'article_body': "",
                'sector': "",
            }

        # NaN 값을 빈 문자열로 처리
        df_filtered = df_filtered.fillna('')

        # 딕셔너리 생성
        agent_data = {
            'symbol': symbol,
            'search_date': search_date,
            'titles': ' / '.join(df_filtered['title'].astype(str).tolist()),
            'descriptions': ' / '.join(df_filtered['description'].astype(str).tolist()),
            'article_body': ' / '.join(df_filtered['article_body'].astype(str).tolist()) if 'article_body' in df_filtered.columns else "",
            'sector': df_filtered['Sector'].iloc[0] if 'Sector' in df_filtered.columns else ""
        }

        return agent_data
    except Exception as e:
        print(f"데이터 포맷팅 오류 (symbol={symbol}, date={search_date}): {e}")
        return {
            'symbol': symbol,
            'search_date': search_date,
            'titles': "",
            'descriptions': "",
            'article_body': "",
            'sector': ""
        }

# --- 에이전트 설계 ---
class BaseModelAgent(ABC):
    @abstractmethod
    def predict(self, stock_data: str) -> AgentPrediction:
        pass

class OpenAIAgent(BaseModelAgent):
    """OpenRouter를 통해 API를 사용하는 표준 에이전트"""

    def __init__(self, model_name: str, client: OpenAI, site_url: str, app_name: str, persona_profile: Optional[str] = None, is_expert: bool = False):
        self.model_name = model_name
        self.client = client
        self.site_url = site_url
        self.app_name = app_name
        self.persona_profile = persona_profile
        self.is_expert = is_expert
        # self.conversation_history = []  # 대화 히스토리 저장 (주석처리: 독립 호출 모드)

    def predict(self, stock_data: str) -> Optional[AgentPrediction]:
        # ========== 독립 호출 모드 (현재 활성화) ==========
        prompt = create_stock_prompt(stock_data, self.persona_profile, self.is_expert)
        effort_level = "medium"

        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.app_name,
                },
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                        "effort" : effort_level
                    }
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            # 응답이 비어있는 경우 처리
            if not content or content.strip() == "":
                print(f"⚠ Empty response from model '{self.model_name}'")
                return None

            # JSON 파싱 전에 내용 정제
            content = content.strip()

            # Markdown 코드 블록 제거
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # JSON 추출 시도 (텍스트가 섞여있을 경우 대비)
            if not content.startswith("{"):
                # 첫 번째 { 찾기
                start_idx = content.find("{")
                if start_idx != -1:
                    content = content[start_idx:]

            if not content.endswith("}"):
                # 마지막 } 찾기
                end_idx = content.rfind("}")
                if end_idx != -1:
                    content = content[:end_idx+1]

            result: Dict[str, Any] = json.loads(content)

            # 다중 기간 예측 결과 파싱
            prediction = {}
            for period in ["day_1", "day_3", "day_7", "day_15", "day_30"]:
                if period in result:
                    period_data = result[period]
                    prediction[period] = {
                        "decision": period_data.get("decision", "up"),
                        "confidence": int(period_data.get("confidence", 50)),
                        "reason": period_data.get("reason", "")
                    }
                else:
                    prediction[period] = {"decision": "up", "confidence": 50, "reason": "Missing period data"}

            return AgentPrediction(**prediction)
        except json.JSONDecodeError as e:
            print(f"\n{'='*60}")
            print(f"❌ JSON DECODE ERROR")
            print(f"{'='*60}")
            print(f"Model: {self.model_name}")
            print(f"Error: {e}")
            print(f"Error position: line {e.lineno}, column {e.colno}, char {e.pos}")
            print(f"\nFull response ({len(content)} chars):")
            print(f"{'-'*60}")
            print(content)
            print(f"{'-'*60}")
            if len(content) > 500:
                print(f"\nResponse around error position (char {e.pos}):")
                start = max(0, e.pos - 100)
                end = min(len(content), e.pos + 100)
                print(f"...{content[start:end]}...")
            print(f"{'='*60}\n")
            return None
        except KeyError as e:
            print(f"\n{'='*60}")
            print(f"❌ KEY ERROR")
            print(f"{'='*60}")
            print(f"Model: {self.model_name}")
            print(f"Missing key: {e}")
            print(f"Available keys in result: {list(result.keys()) if 'result' in locals() else 'N/A'}")
            print(f"Result content: {result if 'result' in locals() else 'N/A'}")
            print(f"{'='*60}\n")
            return None
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ UNEXPECTED ERROR")
            print(f"{'='*60}")
            print(f"Model: {self.model_name}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            import traceback
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            print(f"{'='*60}\n")
            return None  # 오류 발생 시 None 반환하여 집계에서 제외

# --- 예측 시스템 ---
class StockPredictor:
    def __init__(self, expert_agent: OpenAIAgent, crowd_agents: List[OpenAIAgent], crowd_agent_personas: Optional[List[str]] = None, fail_log_callback=None):
        self.expert_agent = expert_agent
        self.crowd_agents = crowd_agents
        self.crowd_agent_personas = crowd_agent_personas or ["Unknown"] * len(crowd_agents)
        self.fail_log_callback = fail_log_callback

    def predict(self, stock_data: str, symbol: str = "", search_date: str = "") -> Dict:
        # 전문가 예측은 단일 요청이므로 그냥 실행
        expert_result = self.expert_agent.predict(stock_data)

        # 전문가 예측이 실패한 경우 처리
        if expert_result is None:
            print("⚠ 전문가 모델 예측 실패. 전문가 예측 없이 크라우드 예측만 집계합니다.")
            # Expert 실패 로그 기록
            if self.fail_log_callback:
                self.fail_log_callback(
                    symbol=symbol,
                    search_date=search_date,
                    model_name=self.expert_agent.model_name,
                    model_type="expert",
                    persona_id="N/A",
                    agent_index="N/A",
                    error_type="PredictionError",
                    error_message="Expert prediction returned None"
                )

        # crowd_agents 병렬 실행
        crowd_results = []
        crowd_result_indices = []  # 성공한 agent의 인덱스 추적
        with ThreadPoolExecutor(max_workers=len(self.crowd_agents)) as executor:
            futures = {executor.submit(agent.predict, stock_data): idx for idx, agent in enumerate(self.crowd_agents)}
            for future in as_completed(futures):
                agent_idx = futures[future]
                try:
                    res = future.result()
                    if res is not None:  # None이 아닌 결과만 추가
                        crowd_results.append(res)
                        crowd_result_indices.append(agent_idx)
                    else:
                        print(f"⚠ Crowd agent {agent_idx+1} (persona: {self.crowd_agent_personas[agent_idx]}) 예측 실패")
                        # Crowd agent 실패 로그 기록
                        if self.fail_log_callback:
                            self.fail_log_callback(
                                symbol=symbol,
                                search_date=search_date,
                                model_name=self.crowd_agents[agent_idx].model_name,
                                model_type=f"crowd_{agent_idx+1}",
                                persona_id=self.crowd_agent_personas[agent_idx],
                                agent_index=agent_idx+1,
                                error_type="PredictionError",
                                error_message="Crowd agent prediction returned None"
                            )
                except Exception as e:
                    print(f"⚠ Crowd agent {agent_idx+1} (persona: {self.crowd_agent_personas[agent_idx]}) 오류: {e}")
                    # Crowd agent 실패 로그 기록
                    if self.fail_log_callback:
                        self.fail_log_callback(
                            symbol=symbol,
                            search_date=search_date,
                            model_name=self.crowd_agents[agent_idx].model_name,
                            model_type=f"crowd_{agent_idx+1}",
                            persona_id=self.crowd_agent_personas[agent_idx],
                            agent_index=agent_idx+1,
                            error_type=type(e).__name__,
                            error_message=str(e)
                        )

        # --- 각 기간별 집계 ---
        periods = ["day_1", "day_3", "day_7", "day_15", "day_30"]
        aggregated_results = {}

        for period in periods:
            up_cnt = down_cnt = 0
            up_sum = down_sum = 0

            # 전문가 예측 포함 (None이 아닌 경우에만)
            if expert_result is not None:
                expert_decision = expert_result[period]["decision"].strip().lower()
                expert_conf = expert_result[period]["confidence"]
                if expert_decision == "up":
                    up_sum += expert_conf
                    up_cnt += 1
                elif expert_decision == "down":
                    down_sum += expert_conf
                    down_cnt += 1

            # 크라우드 예측 집계
            for res in crowd_results:
                decision = res[period]["decision"].strip().lower()
                confidence = res[period]["confidence"]
                if decision == "up":
                    up_sum += confidence
                    up_cnt += 1
                elif decision == "down":
                    down_sum += confidence
                    down_cnt += 1

            # 최종 결정
            if up_sum > down_sum:
                final_decision = "up"
                avg_confidence = up_sum / up_cnt if up_cnt else 0
            else:
                final_decision = "down"
                avg_confidence = down_sum / down_cnt if down_cnt else 0

            aggregated_results[period] = {
                "up_confidence_sum": up_sum,
                "down_confidence_sum": down_sum,
                "average_confidence": avg_confidence,
                "final_decision": final_decision
            }

        return {
            "expert_prediction": expert_result,
            "crowd_predictions": crowd_results,
            "crowd_result_indices": crowd_result_indices,
            "aggregated_results": aggregated_results
        }


if __name__ == "__main__":
    start_time = time.time()

    # ========== 설정 구간 (사용자가 수정할 부분) ==========
    CSV_PATH = "data/articles/csv/news_with_article_body_test_v1.csv"

    # 모든 이벤트 처리
    FILTER_SYMBOLS = None  # None이면 전체, 특정 종목만:
    FILTER_DATES = None  # None이면 전체, 특정 날짜만:

    EXPERT_MODEL = "gpt-5"
    CROWD_MODEL = "gpt-5-nano"
    NUM_CROWD_AGENTS = 25

    PERSONA_FILE_PATH = "data/ESS11/persona.json"

    YOUR_SITE_URL = "http://localhost:8000"
    YOUR_APP_NAME = "AI Stock Predictor"
    # ===================================================

    # --- 1. OpenRouter API 클라이언트 초기화 ---
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # --- 2. 에이전트 초기화 ---
    expert_agent = OpenAIAgent(model_name=EXPERT_MODEL, client=client, site_url=YOUR_SITE_URL, app_name=YOUR_APP_NAME, is_expert=True)

    # 페르소나 데이터 로딩
    persona_data = load_persona_data(PERSONA_FILE_PATH)

    # 랜덤하게 페르소나 선택
    selected_personas = select_random_personas(persona_data, NUM_CROWD_AGENTS)

    # crowd agents 생성 (각각 다른 페르소나 적용)
    crowd_agents = []
    crowd_agent_personas = []  # 페르소나 ID 저장용
    for i, persona in enumerate(selected_personas):
        persona_profile = format_persona_profile(persona)
        agent = OpenAIAgent(
            model_name=CROWD_MODEL,
            client=client,
            site_url=YOUR_SITE_URL,
            app_name=YOUR_APP_NAME,
            persona_profile=persona_profile
        )
        crowd_agents.append(agent)
        crowd_agent_personas.append(persona.get('PROMPT_ID', 'Unknown'))
        print(f"Crowd Agent {i+1} 페르소나: {persona.get('PROMPT_ID', 'Unknown')} - {persona.get('PERSONA_PROFILE', {}).get('Identity', {}).get('Nationality', 'Unknown')}")

    # Fail 로그 콜백 함수 정의
    def log_failure(symbol, search_date, model_name, model_type, persona_id, agent_index, error_type, error_message):
        """개별 agent 실패를 로깅하는 콜백 함수 (thread-safe)"""
        with file_lock:
            with open(fail_log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fail_fieldnames)
                writer.writerow({
                    "symbol": symbol,
                    "search_date": search_date,
                    "model_name": model_name,
                    "model_type": model_type,
                    "persona_id": persona_id,
                    "agent_index": agent_index,
                    "error_type": error_type,
                    "error_message": error_message,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    predictor = StockPredictor(expert_agent, crowd_agents, crowd_agent_personas, fail_log_callback=log_failure)

    # --- 3. CSV에서 이벤트 목록 로드 ---
    events = load_events_from_csv(CSV_PATH, symbols=FILTER_SYMBOLS, dates=FILTER_DATES)

    if not events:
        print("처리할 이벤트가 없습니다. CSV 경로와 필터 조건을 확인하세요.")
        exit(1)

    print(f"\n총 {len(events)}개의 이벤트를 처리합니다.")
    print(f"이벤트 목록: {events[:5]}{'...' if len(events) > 5 else ''}\n")

    # --- 4. 결과 저장 파일 초기화 및 체크포인트 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"prediction_results_{timestamp}.csv"
    fail_log_file = f"prediction_failures_{timestamp}.csv"
    checkpoint_file = f"checkpoint_{timestamp}.json"

    fieldnames = ["symbol", "search_date", "prediction_date", "model_type", "model_name", "decision", "confidence", "reason"]
    fail_fieldnames = ["symbol", "search_date", "model_name", "model_type", "persona_id", "agent_index", "error_type", "error_message", "timestamp"]

    # 체크포인트 로드 (이전 실행에서 중단된 경우)
    completed_events = set()
    resume_mode = False

    # 기존 체크포인트 파일 찾기
    import glob
    existing_checkpoints = sorted(glob.glob("checkpoint_*.json"), reverse=True)
    if existing_checkpoints and os.path.exists(existing_checkpoints[0]):
        print(f"\n⚠️  기존 체크포인트 발견: {existing_checkpoints[0]}")
        user_input = input("이어서 실행하시겠습니까? (y/n): ").strip().lower()
        if user_input == 'y':
            resume_mode = True
            checkpoint_file = existing_checkpoints[0]
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_events = set(tuple(e) for e in checkpoint_data.get('completed_events', []))
                output_file = checkpoint_data.get('output_file', output_file)
                fail_log_file = checkpoint_data.get('fail_log_file', fail_log_file)
                timestamp = checkpoint_data.get('timestamp', timestamp)
            print(f"✅ 체크포인트 로드 완료: {len(completed_events)}개 이벤트 이미 처리됨")
            print(f"   출력 파일: {output_file}")
            print(f"   실패 로그: {fail_log_file}")

    if not resume_mode:
        # CSV 파일 헤더 작성 (새로 시작하는 경우만)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        # Fail 로그 파일 헤더 작성 (새로 시작하는 경우만)
        with open(fail_log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fail_fieldnames)
            writer.writeheader()

        # 초기 체크포인트 저장
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'output_file': output_file,
                'fail_log_file': fail_log_file,
                'completed_events': [],
                'total_events': len(events)
            }, f, indent=2)

    # --- 5. 각 이벤트에 대해 예측 실행 (병렬 처리) ---
    periods = ["day_1", "day_3", "day_7", "day_15", "day_30"]

    # Thread-safe 카운터 및 Lock (딕셔너리로 변경)
    counters = {"success": len([e for e in completed_events if e in events]), "fail": 0}
    counter_lock = threading.Lock()
    file_lock = threading.Lock()
    checkpoint_lock = threading.Lock()

    def save_checkpoint(completed_event):
        """체크포인트 저장 (thread-safe)"""
        with checkpoint_lock:
            completed_events.add(completed_event)
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'output_file': output_file,
                    'fail_log_file': fail_log_file,
                    'completed_events': list(completed_events),
                    'total_events': len(events),
                    'completed_count': len(completed_events)
                }, f, indent=2)

    def process_event(event_info):
        """단일 이벤트를 처리하는 함수"""
        event_idx, symbol, search_date = event_info

        # 이미 처리된 이벤트는 스킵
        if (symbol, search_date) in completed_events:
            print(f"\n⏭️  [{event_idx}/{len(events)}] 스킵 (이미 처리됨): {symbol} @ {search_date}")
            return

        event_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"[{event_idx}/{len(events)}] 처리 중: {symbol} @ {search_date}")
        print(f"{'='*60}")

        # 데이터 로드 및 포맷팅
        agent_data = convert_df_to_agent_format(CSV_PATH, symbol, search_date)

        # 데이터가 비어있으면 스킵
        if not agent_data['titles'] and not agent_data['descriptions'] and not agent_data['article_body']:
            error_msg = "뉴스 데이터 없음"
            print(f"  ⚠ {error_msg}. 스킵합니다.")

            # Fail 로그 기록 (thread-safe)
            with file_lock:
                with open(fail_log_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fail_fieldnames)
                    writer.writerow({
                        "symbol": symbol,
                        "search_date": search_date,
                        "model_name": "N/A",
                        "model_type": "data_loading",
                        "persona_id": "N/A",
                        "agent_index": "N/A",
                        "error_type": "DataError",
                        "error_message": error_msg,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
            with counter_lock:
                counters["fail"] += 1
            return

        stock_data = json.dumps(agent_data, ensure_ascii=False)

        # 예측 실행
        try:
            result = predictor.predict(stock_data, symbol=symbol, search_date=search_date)
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ 예측 실패: {error_msg}")

            # Fail 로그 기록 (thread-safe)
            with file_lock:
                with open(fail_log_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fail_fieldnames)
                    writer.writerow({
                        "symbol": symbol,
                        "search_date": search_date,
                        "model_name": "N/A",
                        "model_type": "prediction",
                        "persona_id": "N/A",
                        "agent_index": "N/A",
                        "error_type": type(e).__name__,
                        "error_message": error_msg,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
            with counter_lock:
                counters["fail"] += 1
            return

        # 예측 날짜 계산
        prediction_dates = calculate_prediction_dates(search_date)

        # 결과 출력
        expert_result = result['expert_prediction']
        if expert_result is not None:
            print(f"\n전문가 의견 ({EXPERT_MODEL}):")
            for period in periods:
                expert_pred = expert_result[period]
                print(f"  {period}: {expert_pred['decision'].upper()} (신뢰도: {expert_pred['confidence']})")
        else:
            print(f"\n⚠ 전문가 의견을 가져올 수 없습니다.")

        print(f"\n집계된 최종 결과:")
        for period in periods:
            agg_result = result['aggregated_results'][period]
            print(f"  {period}: {agg_result['final_decision'].upper()} (평균 신뢰도: {agg_result['average_confidence']:.1f})")

        # --- 6. 결과를 CSV에 append ---
        rows = []

        # 전문가 결과 (None이 아닌 경우에만)
        if expert_result is not None:
            for period in periods:
                expert_pred = expert_result[period]
                rows.append({
                    "symbol": symbol,
                    "search_date": search_date,
                    "prediction_date": prediction_dates[period],
                    "model_type": "expert",
                    "model_name": EXPERT_MODEL,
                    "decision": expert_pred["decision"],
                    "confidence": expert_pred["confidence"],
                    "reason": expert_pred.get("reason", "")
                })

        # 크라우드 결과 (실제 agent 인덱스 사용)
        for result_idx, crowd in enumerate(result['crowd_predictions']):
            actual_agent_idx = result['crowd_result_indices'][result_idx]
            for period in periods:
                crowd_pred = crowd[period]
                rows.append({
                    "symbol": symbol,
                    "search_date": search_date,
                    "prediction_date": prediction_dates[period],
                    "model_type": f"crowd_{actual_agent_idx+1}",
                    "model_name": CROWD_MODEL,
                    "decision": crowd_pred["decision"],
                    "confidence": crowd_pred["confidence"],
                    "reason": crowd_pred.get("reason", "")
                })

        # 크라우드 평균 (성공한 예측이 있을 때만)
        if len(result['crowd_predictions']) > 0:
            for period in periods:
                # 크라우드 모델들의 up/down 가중치 합 계산
                up_sum = 0
                down_sum = 0
                for crowd in result['crowd_predictions']:
                    decision = crowd[period]["decision"].strip().lower()
                    confidence = crowd[period]["confidence"]
                    if decision == "up":
                        up_sum += confidence
                    elif decision == "down":
                        down_sum += confidence

                # 최종 결정: 가중치 합이 큰 쪽
                crowd_final_decision = "up" if up_sum > down_sum else "down"

                # 평균 신뢰도 계산
                crowd_confidences = [crowd[period]["confidence"] for crowd in result['crowd_predictions']]
                crowd_avg_confidence = sum(crowd_confidences) / len(crowd_confidences)

                rows.append({
                    "symbol": symbol,
                    "search_date": search_date,
                    "prediction_date": prediction_dates[period],
                    "model_type": "crowd_average",
                    "model_name": "crowd_models",
                    "decision": crowd_final_decision,
                    "confidence": round(crowd_avg_confidence, 1),
                    "reason": f"Aggregated decision from {len(crowd_confidences)} crowd models (up_sum={up_sum}, down_sum={down_sum})"
                })
        else:
            print(f"  ⚠ 성공한 크라우드 예측이 없어 평균을 계산할 수 없습니다.")

        # CSV에 추가 (thread-safe)
        with file_lock:
            with open(output_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(rows)

        with counter_lock:
            counters["success"] += 1

        event_elapsed_time = time.time() - event_start_time
        print(f"  ✅ 결과가 {output_file}에 저장되었습니다.")
        print(f"  ⏱️  소요 시간: {event_elapsed_time:.2f}초")

        # 체크포인트 저장
        save_checkpoint((symbol, search_date))
        print(f"  💾 체크포인트 저장됨 ({len(completed_events)}/{len(events)})")

    # 병렬 실행 (최대 10개의 이벤트 동시 처리)
    event_info_list = [(idx+1, symbol, search_date) for idx, (symbol, search_date) in enumerate(events)]

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_event, event_info) for event_info in event_info_list]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠ 이벤트 처리 중 예상치 못한 오류: {e}")

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"⚠️  사용자에 의해 중단되었습니다 (Ctrl+C)")
        print(f"{'='*60}")
        print(f"진행 상황:")
        print(f"  - 처리 완료: {len(completed_events)}/{len(events)} 이벤트")
        print(f"  - 성공: {counters['success']}개")
        print(f"  - 실패: {counters['fail']}개")
        print(f"\n체크포인트 저장됨: {checkpoint_file}")
        print(f"다음에 실행하면 이어서 진행됩니다.")
        print(f"{'='*60}")
        exit(0)

    end_time = time.time()
    total_elapsed_time = end_time - start_time

    print(f"\n{'='*60}")
    print(f"전체 실험 완료!")
    print(f"  - 총 이벤트: {len(events)}개")
    print(f"  - 성공: {counters['success']}개")
    print(f"  - 실패: {counters['fail']}개")
    print(f"  - 전체 처리 시간: {total_elapsed_time:.2f}초")
    print(f"결과 파일: {output_file}")
    print(f"실패 로그: {fail_log_file}")
    print(f"\n💾 체크포인트 파일 삭제 중...")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"   ✅ {checkpoint_file} 삭제 완료")
    print(f"{'='*60}")