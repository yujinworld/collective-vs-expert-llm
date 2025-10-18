# I_stats_test.py 통계 분석 상세 설명

## 목차
1. [데이터 로드 및 전처리](#1-데이터-로드-및-전처리)
2. [Baseline 분석](#2-baseline-분석)
3. [분류 지표](#3-분류-지표)
4. [수익률 계산](#4-수익률-계산)
5. [통계 검정 1: 수익률 비교](#5-통계-검정-1-수익률-비교)
6. [통계 검정 2: 승률 비교](#6-통계-검정-2-승률-비교)
7. [통계 검정 3: Confidence 비교](#7-통계-검정-3-confidence-비교)
8. [Baseline과의 비교](#8-baseline과의-비교)

---

## 1. 데이터 로드 및 전처리

### 1.1 파일 로드
```python
# G_scoring_answer.py 실행 결과 파일 중 가장 최근 파일 로드
df_detailed = pd.read_csv('scoring_results_detailed_YYYYMMDD_HHMMSS.csv')
df_crowd = pd.read_csv('scoring_results_crowd_YYYYMMDD_HHMMSS.csv')
```

**파일 구조:**
- `df_detailed`: Expert 및 개별 Crowd 모델의 예측 결과 (각 행 = 1개 모델의 1개 예측)
- `df_crowd`: Crowd 집단의 투표 결과 (각 행 = Crowd 전체의 합의된 예측)

### 1.2 데이터 분리 및 병합
```python
# Expert 데이터 추출
df_expert = df_detailed[df_detailed['model_type'] == 'expert']

# Crowd 데이터에 가격 정보 병합
df_crowd_agg = df_crowd.merge(df_prices, on=['symbol', 'search_date', 'prediction_date'])
```

**병합 이유:** `df_crowd`에는 실제 주가 정보(Close_search, Close_pred)가 없으므로, `df_detailed`에서 추출하여 병합

---

## 2. Baseline 분석

### 2.1 Baseline 정의
**가설:** "항상 UP(매수)으로 예측하면 어떻게 될까?"

```python
df_baseline = df_expert.copy()
df_baseline['baseline_prediction'] = 1  # 모든 예측을 UP(1)로 설정
df_baseline['baseline_correct'] = np.where(
    df_baseline['baseline_prediction'] == df_baseline['actual_sign'], 1, 0
)
baseline_accuracy = df_baseline['baseline_correct'].mean()
```

### 2.2 계산 과정
1. Expert와 동일한 데이터셋 사용
2. 모든 예측을 `predicted_sign = 1` (UP)로 강제 설정
3. `actual_sign`과 비교하여 정답률 계산

**의미:**
- 주식 시장이 장기적으로 상승 추세라면 baseline 정답률이 높게 나올 수 있음
- Expert/Crowd가 baseline보다 못하다면 모델의 의미가 없음

---

## 3. 분류 지표

### 3.1 Accuracy (정확도)
```python
accuracy = accuracy_score(y_true, y_pred)
```

**계산식:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**의미:**
- 전체 예측 중 맞춘 비율
- UP/DOWN 2클래스 분류 문제에서 기본 성능 지표

### 3.2 Precision (정밀도)
```python
precision = precision_score(y_true, y_pred, average='weighted')
```

**계산식 (weighted average):**
```
Precision_class = TP / (TP + FP)
Precision_weighted = Σ(Precision_class × support_class) / total_samples
```

**의미:**
- UP으로 예측한 것 중 실제로 UP인 비율
- DOWN으로 예측한 것 중 실제로 DOWN인 비율
- 두 클래스의 precision을 샘플 수에 비례하여 가중평균

### 3.3 Recall (재현율)
```python
recall = recall_score(y_true, y_pred, average='weighted')
```

**계산식 (weighted average):**
```
Recall_class = TP / (TP + FN)
Recall_weighted = Σ(Recall_class × support_class) / total_samples
```

**의미:**
- 실제 UP 중 UP으로 예측한 비율
- 실제 DOWN 중 DOWN으로 예측한 비율

### 3.4 F1-score (F1 점수)
```python
f1 = f1_score(y_true, y_pred, average='weighted')
```

**계산식:**
```
F1_class = 2 × (Precision × Recall) / (Precision + Recall)
F1_weighted = Σ(F1_class × support_class) / total_samples
```

**의미:**
- Precision과 Recall의 조화평균
- 클래스 불균형이 있을 때 accuracy보다 신뢰할 수 있는 지표

---

## 4. 수익률 계산

### 4.1 수익률 전략
```python
df['price_change_pct'] = (df['Close_pred'] - df['Close_search']) / df['Close_search'] * 100
df['strategy_return'] = df['predicted_sign'] * df['price_change_pct']
```

### 4.2 계산 로직

**Case 1: UP 예측 (predicted_sign = 1)**
```
strategy_return = 1 × price_change_pct
- 주가 상승(+5%) → 수익 +5%
- 주가 하락(-3%) → 손실 -3%
```

**Case 2: DOWN 예측 (predicted_sign = -1)**
```
strategy_return = -1 × price_change_pct
- 주가 상승(+5%) → 손실 -5% (공매도 실패)
- 주가 하락(-3%) → 수익 +3% (공매도 성공)
```

### 4.3 집계 지표
```python
total_return = df['strategy_return'].sum()     # 총 누적 수익률
avg_return = df['strategy_return'].mean()      # 평균 수익률
win_rate = (df['strategy_return'] > 0).mean()  # 승률
```

**의미:**
- `total_return`: 모든 거래의 수익률 합계
- `avg_return`: 거래당 평균 수익률
- `win_rate`: 수익을 낸 거래의 비율

---

## 5. 통계 검정 1: 수익률 비교

### 5.1 기초 통계량
```python
expert_returns.mean()              # 평균
expert_returns.median()            # 중앙값
expert_returns.std()               # 표준편차
expert_returns.min() / max()       # 최소/최대
expert_returns.quantile(0.25/0.75) # Q1, Q3
```

**분포 분석:**
- 평균 vs 중앙값: 분포의 치우침(skewness) 파악
- 표준편차: 변동성(위험도) 측정
- 사분위수: 이상치 및 분포 범위 확인

### 5.2 T-검정 (Independent t-test)
```python
t_stat, t_pval = stats.ttest_ind(expert_returns, crowd_returns, equal_var=False)
```

**가설:**
- H₀ (귀무가설): Expert와 Crowd의 평균 수익률이 같다
- H₁ (대립가설): Expert와 Crowd의 평균 수익률이 다르다

**검정 통계량:**
```
t = (μ_expert - μ_crowd) / sqrt(s²_expert/n_expert + s²_crowd/n_crowd)
```

**해석:**
- `equal_var=False`: Welch's t-test 사용 (분산이 같다고 가정하지 않음)
- `p-value < 0.05`: 유의수준 5%에서 귀무가설 기각 → 두 집단의 평균이 유의미하게 다름
- `t_stat > 0`: Expert 평균이 더 높음 / `t_stat < 0`: Crowd 평균이 더 높음

### 5.3 Mann-Whitney U 검정 (비모수 검정)
```python
u_stat, u_pval = stats.mannwhitneyu(expert_returns, crowd_returns, alternative='two-sided')
```

**특징:**
- 정규성 가정 불필요 (수익률 분포가 정규분포가 아닐 수 있음)
- 중앙값 비교 (평균 대신)

**가설:**
- H₀: 두 집단의 분포가 같다
- H₁: 두 집단의 분포가 다르다

**검정 통계량:**
- 두 집단을 합쳐서 순위를 매긴 후, 순위 합 비교
- U-statistic이 작을수록 두 집단의 차이가 큼

**해석:**
- `p-value < 0.05`: 두 집단의 중앙값이 유의미하게 다름
- T-검정과 결과가 다르면 → 분포가 정규분포가 아니거나 이상치 존재

### 5.4 Levene 검정 (분산 동질성 검정)
```python
levene_stat, levene_pval = stats.levene(expert_returns, crowd_returns)
```

**가설:**
- H₀: 두 집단의 분산이 같다
- H₁: 두 집단의 분산이 다르다

**의미:**
- Expert와 Crowd의 **위험도(변동성)**가 다른지 검증
- `p-value < 0.05`: 분산이 유의미하게 다름
  - Expert 분산 > Crowd 분산 → Expert가 더 변동성 높음 (위험함)
  - Crowd 분산 > Expert 분산 → Crowd가 더 변동성 높음

**실전 의미:**
- 같은 수익률이라도 변동성이 낮은 쪽이 더 안정적

---

## 6. 통계 검정 2: 승률 비교

### 6.1 승률 정의
```python
expert_win_rate = (expert_returns > 0).mean()
crowd_win_rate = (crowd_returns > 0).mean()
```

**계산:**
```
win_rate = (수익 거래 수) / (전체 거래 수)
```

### 6.2 카이제곱 검정 (Chi-square test)
```python
contingency_table = [
    [expert_win, expert_loss],
    [crowd_win, crowd_loss]
]
chi2_stat, chi2_pval, dof, expected = chi2_contingency(contingency_table)
```

**분할표 예시:**
|        | Win | Loss | Total |
|--------|-----|------|-------|
| Expert | 398 | 222  | 620   |
| Crowd  | 442 | 178  | 620   |

**가설:**
- H₀: Expert와 Crowd의 승률이 독립적 (차이 없음)
- H₁: Expert와 Crowd의 승률이 독립적이지 않음 (차이 있음)

**검정 통계량:**
```
χ² = Σ[(Observed - Expected)² / Expected]
```

**해석:**
- `p-value < 0.05`: 승률 차이가 통계적으로 유의미함
- `dof = 1`: 자유도 (2×2 테이블에서 1)

---

## 7. 통계 검정 3: Confidence 비교

### 7.1 Confidence란?
- 모델이 자신의 예측에 대해 부여한 신뢰도 (0~100)
- Expert: GPT-5가 부여한 confidence
- Crowd: 투표 결과의 가중 confidence

### 7.2 Confidence T-검정
```python
t_conf_stat, t_conf_pval = stats.ttest_ind(expert_confidence, crowd_confidence, equal_var=False)
```

**가설:**
- H₀: Expert와 Crowd의 평균 confidence가 같다
- H₁: Expert와 Crowd의 평균 confidence가 다르다

**해석:**
- Crowd confidence가 높다 → 집단지성의 합의도가 높음
- Expert confidence가 높다 → 모델의 확신이 강함

### 7.3 Confidence와 수익률의 상관관계
```python
expert_corr = np.corrcoef(df_expert['confidence'], expert_returns)[0, 1]
```

**Pearson 상관계수:**
```
r = Σ[(x_i - x̄)(y_i - ȳ)] / sqrt(Σ(x_i - x̄)² × Σ(y_i - ȳ)²)
```

**해석:**
- `r > 0`: confidence가 높을수록 수익률도 높음 (잘 보정된 모델)
- `r ≈ 0`: confidence와 수익률이 무관 (과신 또는 과소평가)
- `r < 0`: confidence가 높을수록 수익률이 낮음 (역설적, 문제 있음)

### 7.4 Confidence 구간별 승률 분석
```python
conf_bins = [0, 50, 60, 70, 80, 100]
df_model['conf_bin'] = pd.cut(df_model['confidence'], bins=conf_bins)
```

**분석:**
- 각 confidence 구간별로 승률과 평균 수익률 계산
- 이상적인 모델: confidence가 높을수록 승률도 높아야 함

**예시 출력:**
```
Expert:
  50-60: 승률 57.68%, 평균 수익률 +2.83%
  60-70: 승률 70.81%, 평균 수익률 +6.63%
  80-100: 승률 100.00%, 평균 수익률 +15.33%
```

**해석:**
- Expert는 confidence 보정이 잘 되어 있음 (높을수록 승률↑)
- Crowd도 비슷한 패턴이면 → 집단지성이 신뢰도를 잘 반영

---

## 8. Baseline과의 비교

### 8.1 Expert vs Baseline
```python
t_expert_base, p_expert_base = stats.ttest_ind(expert_returns, baseline_returns, equal_var=False)
```

**가설:**
- H₀: Expert와 Baseline의 평균 수익률이 같다
- H₁: Expert와 Baseline의 평균 수익률이 다르다

**해석:**
- `p-value >= 0.05`: Expert가 "무조건 매수" 전략과 차이 없음 → 모델 무용론
- `p-value < 0.05` + `t_stat > 0`: Expert가 Baseline보다 우수
- `p-value < 0.05` + `t_stat < 0`: Expert가 Baseline보다 나쁨 (오히려 역효과)

### 8.2 Crowd vs Baseline
```python
t_crowd_base, p_crowd_base = stats.ttest_ind(crowd_returns, baseline_returns, equal_var=False)
```

**동일한 논리:**
- Crowd가 Baseline보다 유의미하게 좋은지 검증
- 만약 Crowd > Expert > Baseline이면 → 집단지성의 우수성 입증

---

## 결과 저장

### 8.1 통계 검정 결과
```csv
test,statistic,p_value,significant
T-test (Expert vs Crowd) 수익률,-2.6524,0.0081,True
U-test (Expert vs Crowd) 수익률,176441.5000,0.0124,True
Levene test (분산) 수익률,4.8204,0.0283,True
...
```

### 8.2 분류 지표 결과
```csv
model,accuracy,precision,recall,f1_score,win_rate,total_return_pct,avg_return_pct,...
Expert,0.6419,0.6511,0.6419,0.6456,0.6419,2979.68,4.81,...
Crowd,0.7129,0.7036,0.7129,0.7041,0.7129,4567.59,7.37,...
Baseline,0.6435,0.4142,0.6435,0.5040,0.6435,3371.30,5.44,...
```

### 8.3 Confidence 구간별 성과
```csv
model,confidence_range,count,win_rate,avg_return_pct,total_return_pct,std_return_pct
Expert,50-60,397,0.5768,2.83,1122.51,15.23
Expert,60-70,161,0.7081,6.63,1067.43,14.89
...
```

---

## 통계적 유의성 해석 가이드

### p-value 해석
- `p < 0.01`: 매우 강한 증거 (1% 유의수준)
- `p < 0.05`: 유의미한 증거 (5% 유의수준, 일반적 기준)
- `p < 0.10`: 약한 증거 (10% 유의수준)
- `p >= 0.10`: 증거 불충분

### 효과 크기 (Effect Size)
- T-검정의 t-statistic 절댓값이 클수록 효과 크기 큼
- 평균 차이를 표준편차로 나눈 값 (Cohen's d) 계산 가능
- `|d| < 0.2`: 작은 효과 / `0.2 ≤ |d| < 0.8`: 중간 효과 / `|d| ≥ 0.8`: 큰 효과

### 실무적 의미
- 통계적 유의성 ≠ 실무적 유의성
- 예: `p-value = 0.001`이지만 평균 차이가 0.1%라면 → 실무적으로 무의미
- 수익률 차이가 2~3% 이상이고 p < 0.05라면 → 실무적으로 의미 있음

---

## 결론

이 스크립트는 다음을 검증합니다:

1. **성능 비교**: Expert vs Crowd vs Baseline
2. **통계적 유의성**: 차이가 우연인지, 진짜 차이인지
3. **안정성 비교**: 변동성(위험도) 비교
4. **신뢰도 검증**: Confidence가 실제 성과와 일치하는지

**핵심 질문에 대한 답:**
- "집단지성이 전문가보다 나은가?" → T-검정, U-검정으로 검증
- "차이가 우연인가?" → p-value로 판단
- "위험도는 어떤가?" → 분산 검정으로 확인
- "무작정 매수하는 것보다 나은가?" → Baseline 비교로 검증
