# T-검정
# U-검정
# Precision
# F1-score
# recall
# accuracy
# 분산 검정
# 수익률

# 다 올랐다고 찍었을 때?

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. 데이터 로드
# ===========================
# G_scoring_answer.py 실행 후 생성된 결과 파일 중 가장 최근 파일 로드
import glob
import os

data_dir = "../data/answer"
detailed_files = sorted(glob.glob(f"{data_dir}/scoring_results_detailed_*.csv"))
crowd_files = sorted(glob.glob(f"{data_dir}/scoring_results_crowd_*.csv"))

if not detailed_files or not crowd_files:
    raise FileNotFoundError("평가 결과 파일이 없습니다. G_scoring_answer.py를 먼저 실행하세요.")

latest_detailed = detailed_files[-1]
latest_crowd = crowd_files[-1]

print(f"데이터 로드:")
print(f"  - 상세 결과: {os.path.basename(latest_detailed)}")
print(f"  - Crowd 결과: {os.path.basename(latest_crowd)}")

df_detailed = pd.read_csv(latest_detailed)
df_crowd = pd.read_csv(latest_crowd)

# Expert / Crowd 분리
df_expert = df_detailed[df_detailed['model_type'] == 'expert'].copy()

# Crowd 데이터에 가격 정보 병합
df_crowd_agg = df_crowd.copy()
price_columns = ['Close_search', 'Close_pred']
merge_keys = ['symbol', 'search_date', 'prediction_date']

# detailed 파일에서 가격 정보 추출
df_prices = df_detailed[merge_keys + price_columns].drop_duplicates(subset=merge_keys)

# Crowd 데이터와 병합
df_crowd_agg = df_crowd_agg.merge(df_prices, on=merge_keys, how='left')

print(f"\nExpert 예측: {len(df_expert)}건")
print(f"Crowd 예측: {len(df_crowd_agg)}건")
print(f"Crowd 가격 정보 병합: {df_crowd_agg[price_columns].notna().all(axis=1).sum()}/{len(df_crowd_agg)}건")

# ===========================
# 2. Baseline: 전부 UP으로 예측했을 때
# ===========================
print("\n" + "="*80)
print("[Baseline] 전부 UP으로 예측했을 때")
print("="*80)

df_baseline = df_expert.copy()
df_baseline['baseline_prediction'] = 1  # 전부 UP
df_baseline['baseline_correct'] = np.where(df_baseline['baseline_prediction'] == df_baseline['actual_sign'], 1, 0)
baseline_accuracy = df_baseline['baseline_correct'].mean()

print(f"Baseline (전부 UP) 정답률: {baseline_accuracy:.2%}")

# ===========================
# 3. 분류 지표: Precision, Recall, F1-score, Accuracy
# ===========================
print("\n" + "="*80)
print("[분류 지표] Precision, Recall, F1-score, Accuracy")
print("="*80)

def calculate_metrics(df, prefix=""):
    """분류 지표 계산"""
    y_true = df['actual_sign'].values
    y_pred = df['predicted_sign'].values

    # actual_sign이 0인 경우 제외 (가격 변동 없음)
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{prefix}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

expert_metrics = calculate_metrics(df_expert, "Expert")
crowd_metrics = calculate_metrics(df_crowd_agg, "Crowd")
baseline_metrics = calculate_metrics(df_baseline[df_baseline['actual_sign'] != 0].assign(predicted_sign=1), "Baseline (전부 UP)")

# ===========================
# 4. 수익률 계산
# ===========================
print("\n" + "="*80)
print("[수익률 계산]")
print("="*80)

def calculate_returns(df, prefix=""):
    """수익률 계산 (예측에 따라 매수/공매도)"""
    df = df.copy()
    df['price_change_pct'] = (df['Close_pred'] - df['Close_search']) / df['Close_search'] * 100

    # 예측이 UP(1)이면 매수, DOWN(-1)이면 공매도
    # 실제 수익 = predicted_sign * price_change_pct
    df['strategy_return'] = df['predicted_sign'] * df['price_change_pct']

    total_return = df['strategy_return'].sum()
    avg_return = df['strategy_return'].mean()
    win_rate = (df['strategy_return'] > 0).mean()

    print(f"\n{prefix}")
    print(f"  총 수익률: {total_return:+.2f}%")
    print(f"  평균 수익률: {avg_return:+.2f}%")
    print(f"  승률: {win_rate:.2%}")

    return df['strategy_return']

expert_returns = calculate_returns(df_expert, "Expert")
crowd_returns = calculate_returns(df_crowd_agg, "Crowd")

# Baseline: 전부 매수 (UP)
df_baseline_returns = df_expert.copy()
df_baseline_returns['predicted_sign'] = 1
baseline_returns = calculate_returns(df_baseline_returns, "Baseline (전부 UP)")

# ===========================
# 5. 통계 검정 (수익률)
# ===========================
print("\n" + "="*80)
print("[통계 검정 1] 수익률 비교")
print("="*80)

# 기초 통계량
print("\n[수익률 기초 통계량]")
print(f"\nExpert 수익률:")
print(f"  평균: {expert_returns.mean():+.4f}%")
print(f"  중앙값: {expert_returns.median():+.4f}%")
print(f"  표준편차: {expert_returns.std():.4f}%")
print(f"  최소: {expert_returns.min():+.4f}%")
print(f"  최대: {expert_returns.max():+.4f}%")
print(f"  사분위수 (Q1, Q3): {expert_returns.quantile(0.25):+.4f}%, {expert_returns.quantile(0.75):+.4f}%")

print(f"\nCrowd 수익률:")
print(f"  평균: {crowd_returns.mean():+.4f}%")
print(f"  중앙값: {crowd_returns.median():+.4f}%")
print(f"  표준편차: {crowd_returns.std():.4f}%")
print(f"  최소: {crowd_returns.min():+.4f}%")
print(f"  최대: {crowd_returns.max():+.4f}%")
print(f"  사분위수 (Q1, Q3): {crowd_returns.quantile(0.25):+.4f}%, {crowd_returns.quantile(0.75):+.4f}%")

print(f"\nBaseline 수익률:")
print(f"  평균: {baseline_returns.mean():+.4f}%")
print(f"  중앙값: {baseline_returns.median():+.4f}%")
print(f"  표준편차: {baseline_returns.std():.4f}%")

# 5-1. T-검정 (정규성 가정)
print("\n[T-검정] Expert vs Crowd (정규성 가정)")
t_stat, t_pval = stats.ttest_ind(expert_returns.dropna(), crowd_returns.dropna(), equal_var=False)
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pval:.4f}")
print(f"  평균 차이: {expert_returns.mean() - crowd_returns.mean():+.4f}%")
if t_pval < 0.05:
    winner = "Expert" if t_stat > 0 else "Crowd"
    print(f"  ✅ 유의수준 5%에서 {winner}가 통계적으로 유의미하게 우수")
else:
    print(f"  ❌ 유의수준 5%에서 통계적 차이 없음")

# 5-2. Mann-Whitney U 검정 (비모수 검정)
print("\n[Mann-Whitney U 검정] Expert vs Crowd (비모수)")
u_stat, u_pval = stats.mannwhitneyu(expert_returns.dropna(), crowd_returns.dropna(), alternative='two-sided')
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {u_pval:.4f}")
median_expert = expert_returns.median()
median_crowd = crowd_returns.median()
print(f"  중앙값 차이: {median_expert - median_crowd:+.4f}%")
if u_pval < 0.05:
    winner = "Expert" if median_expert > median_crowd else "Crowd"
    print(f"  ✅ 유의수준 5%에서 {winner}가 통계적으로 유의미하게 우수")
else:
    print(f"  ❌ 유의수준 5%에서 통계적 차이 없음")

# 5-3. F-검정 (분산 동질성 검정)
print("\n[Levene 검정] 분산 동질성 검정 (수익률)")
levene_stat, levene_pval = stats.levene(expert_returns.dropna(), crowd_returns.dropna())
print(f"  Levene statistic: {levene_stat:.4f}")
print(f"  p-value: {levene_pval:.4f}")
if levene_pval < 0.05:
    print(f"  ✅ 유의수준 5%에서 두 집단의 분산이 다름")
else:
    print(f"  ❌ 유의수준 5%에서 두 집단의 분산이 같음")

# 분산 비교
expert_var = expert_returns.var()
crowd_var = crowd_returns.var()
print(f"\n  Expert 분산: {expert_var:.4f}")
print(f"  Crowd 분산: {crowd_var:.4f}")
print(f"  분산 비율 (Expert/Crowd): {expert_var/crowd_var:.4f}")

# ===========================
# 5-4. 승률 비교
# ===========================
print("\n" + "="*80)
print("[통계 검정 2] 승률 비교")
print("="*80)

expert_win_rate = (expert_returns > 0).mean()
crowd_win_rate = (crowd_returns > 0).mean()
baseline_win_rate = (baseline_returns > 0).mean()

print(f"\n[승률 (예측이 수익으로 이어진 비율)]")
print(f"  Expert:   {expert_win_rate:.4f} ({(expert_returns > 0).sum()}/{len(expert_returns)})")
print(f"  Crowd:    {crowd_win_rate:.4f} ({(crowd_returns > 0).sum()}/{len(crowd_returns)})")
print(f"  Baseline: {baseline_win_rate:.4f} ({(baseline_returns > 0).sum()}/{len(baseline_returns)})")
print(f"  승률 차이 (Crowd - Expert): {crowd_win_rate - expert_win_rate:+.4f}")

# 승률 카이제곱 검정
from scipy.stats import chi2_contingency

expert_win = (expert_returns > 0).sum()
expert_loss = (expert_returns <= 0).sum()
crowd_win = (crowd_returns > 0).sum()
crowd_loss = (crowd_returns <= 0).sum()

contingency_table = np.array([
    [expert_win, expert_loss],
    [crowd_win, crowd_loss]
])

chi2_stat, chi2_pval, dof, expected = chi2_contingency(contingency_table)

print(f"\n[카이제곱 검정] 승률 차이 검정")
print(f"  χ² statistic: {chi2_stat:.4f}")
print(f"  p-value: {chi2_pval:.4f}")
print(f"  자유도: {dof}")
if chi2_pval < 0.05:
    winner = "Crowd" if crowd_win_rate > expert_win_rate else "Expert"
    print(f"  ✅ 유의수준 5%에서 {winner}의 승률이 통계적으로 유의미하게 높음")
else:
    print(f"  ❌ 유의수준 5%에서 승률 차이가 통계적으로 유의미하지 않음")

# ===========================
# 5-5. Confidence 비교
# ===========================
print("\n" + "="*80)
print("[통계 검정 3] Confidence 비교")
print("="*80)

expert_confidence = df_expert['confidence'].values
crowd_confidence = df_crowd_agg['confidence'].values

print(f"\n[Confidence 기초 통계량]")
print(f"\nExpert Confidence:")
print(f"  평균: {expert_confidence.mean():.4f}")
print(f"  중앙값: {np.median(expert_confidence):.4f}")
print(f"  표준편차: {expert_confidence.std():.4f}")
print(f"  최소: {expert_confidence.min():.4f}")
print(f"  최대: {expert_confidence.max():.4f}")
print(f"  사분위수 (Q1, Q3): {np.percentile(expert_confidence, 25):.4f}, {np.percentile(expert_confidence, 75):.4f}")

print(f"\nCrowd Confidence:")
print(f"  평균: {crowd_confidence.mean():.4f}")
print(f"  중앙값: {np.median(crowd_confidence):.4f}")
print(f"  표준편차: {crowd_confidence.std():.4f}")
print(f"  최소: {crowd_confidence.min():.4f}")
print(f"  최대: {crowd_confidence.max():.4f}")
print(f"  사분위수 (Q1, Q3): {np.percentile(crowd_confidence, 25):.4f}, {np.percentile(crowd_confidence, 75):.4f}")

# Confidence T-검정
print(f"\n[T-검정] Expert vs Crowd Confidence")
t_conf_stat, t_conf_pval = stats.ttest_ind(expert_confidence, crowd_confidence, equal_var=False)
print(f"  t-statistic: {t_conf_stat:.4f}")
print(f"  p-value: {t_conf_pval:.4f}")
print(f"  평균 차이: {expert_confidence.mean() - crowd_confidence.mean():+.4f}")
if t_conf_pval < 0.05:
    winner = "Expert" if t_conf_stat > 0 else "Crowd"
    print(f"  ✅ 유의수준 5%에서 {winner}의 confidence가 통계적으로 유의미하게 높음")
else:
    print(f"  ❌ 유의수준 5%에서 confidence 차이가 통계적으로 유의미하지 않음")

# Confidence U-검정
print(f"\n[Mann-Whitney U 검정] Expert vs Crowd Confidence")
u_conf_stat, u_conf_pval = stats.mannwhitneyu(expert_confidence, crowd_confidence, alternative='two-sided')
print(f"  U-statistic: {u_conf_stat:.4f}")
print(f"  p-value: {u_conf_pval:.4f}")
print(f"  중앙값 차이: {np.median(expert_confidence) - np.median(crowd_confidence):+.4f}")
if u_conf_pval < 0.05:
    winner = "Expert" if np.median(expert_confidence) > np.median(crowd_confidence) else "Crowd"
    print(f"  ✅ 유의수준 5%에서 {winner}의 confidence가 통계적으로 유의미하게 높음")
else:
    print(f"  ❌ 유의수준 5%에서 confidence 차이가 통계적으로 유의미하지 않음")

# Confidence와 실제 성과의 상관관계
print(f"\n[Confidence와 실제 수익률의 상관관계]")
expert_corr = np.corrcoef(df_expert['confidence'], expert_returns)[0, 1]
crowd_corr = np.corrcoef(df_crowd_agg['confidence'], crowd_returns)[0, 1]

print(f"  Expert: {expert_corr:+.4f} (confidence가 높을수록 수익률이 {'높음' if expert_corr > 0 else '낮음'})")
print(f"  Crowd:  {crowd_corr:+.4f} (confidence가 높을수록 수익률이 {'높음' if crowd_corr > 0 else '낮음'})")

# Confidence별 승률 분석
print(f"\n[Confidence 구간별 승률]")
conf_bins = [0, 50, 60, 70, 80, 100]
conf_labels = ['0-50', '50-60', '60-70', '70-80', '80-100']

for model_name, df_model, returns in [('Expert', df_expert, expert_returns),
                                        ('Crowd', df_crowd_agg, crowd_returns)]:
    print(f"\n  {model_name}:")
    df_model = df_model.copy()
    df_model['return'] = returns.values
    df_model['conf_bin'] = pd.cut(df_model['confidence'], bins=conf_bins, labels=conf_labels, include_lowest=True)

    for label in conf_labels:
        subset = df_model[df_model['conf_bin'] == label]
        if len(subset) > 0:
            win_rate = (subset['return'] > 0).mean()
            avg_return = subset['return'].mean()
            print(f"    {label}: 승률 {win_rate:.2%} ({(subset['return'] > 0).sum()}/{len(subset)}), 평균 수익률 {avg_return:+.2f}%")

# ===========================
# 6. Baseline과 비교
# ===========================
print("\n" + "="*80)
print("[Baseline과의 비교]")
print("="*80)

# Expert vs Baseline
print("\n[T-검정] Expert vs Baseline")
t_expert_base, p_expert_base = stats.ttest_ind(expert_returns.dropna(), baseline_returns.dropna(), equal_var=False)
print(f"  t-statistic: {t_expert_base:.4f}")
print(f"  p-value: {p_expert_base:.4f}")
if p_expert_base < 0.05:
    winner = "Expert" if t_expert_base > 0 else "Baseline"
    print(f"  ✅ {winner}가 통계적으로 유의미하게 우수")
else:
    print(f"  ❌ 통계적 차이 없음")

# Crowd vs Baseline
print("\n[T-검정] Crowd vs Baseline")
t_crowd_base, p_crowd_base = stats.ttest_ind(crowd_returns.dropna(), baseline_returns.dropna(), equal_var=False)
print(f"  t-statistic: {t_crowd_base:.4f}")
print(f"  p-value: {p_crowd_base:.4f}")
if p_crowd_base < 0.05:
    winner = "Crowd" if t_crowd_base > 0 else "Baseline"
    print(f"  ✅ {winner}가 통계적으로 유의미하게 우수")
else:
    print(f"  ❌ 통계적 차이 없음")

# ===========================
# 7. 결과 요약 저장
# ===========================
print("\n" + "="*80)
print("[결과 저장]")
print("="*80)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 통계 검정 결과 (수익률)
stats_results = {
    'test': ['T-test (Expert vs Crowd) 수익률',
             'U-test (Expert vs Crowd) 수익률',
             'Levene test (분산) 수익률',
             'T-test (Expert vs Baseline) 수익률',
             'T-test (Crowd vs Baseline) 수익률',
             'Chi2-test (Expert vs Crowd) 승률',
             'T-test (Expert vs Crowd) Confidence',
             'U-test (Expert vs Crowd) Confidence'],
    'statistic': [t_stat, u_stat, levene_stat, t_expert_base, t_crowd_base,
                  chi2_stat, t_conf_stat, u_conf_stat],
    'p_value': [t_pval, u_pval, levene_pval, p_expert_base, p_crowd_base,
                chi2_pval, t_conf_pval, u_conf_pval],
    'significant': [t_pval < 0.05, u_pval < 0.05, levene_pval < 0.05,
                    p_expert_base < 0.05, p_crowd_base < 0.05,
                    chi2_pval < 0.05, t_conf_pval < 0.05, u_conf_pval < 0.05]
}
stats_df = pd.DataFrame(stats_results)
stats_output = f"{data_dir}/stats_test_results_{timestamp}.csv"
stats_df.to_csv(stats_output, index=False, encoding='utf-8')
print(f"통계 검정 결과: {stats_output}")

# 분류 지표 결과
metrics_results = {
    'model': ['Expert', 'Crowd', 'Baseline'],
    'accuracy': [expert_metrics['accuracy'], crowd_metrics['accuracy'], baseline_metrics['accuracy']],
    'precision': [expert_metrics['precision'], crowd_metrics['precision'], baseline_metrics['precision']],
    'recall': [expert_metrics['recall'], crowd_metrics['recall'], baseline_metrics['recall']],
    'f1_score': [expert_metrics['f1'], crowd_metrics['f1'], baseline_metrics['f1']],
    'win_rate': [expert_win_rate, crowd_win_rate, baseline_win_rate],
    'total_return_pct': [expert_returns.sum(), crowd_returns.sum(), baseline_returns.sum()],
    'avg_return_pct': [expert_returns.mean(), crowd_returns.mean(), baseline_returns.mean()],
    'median_return_pct': [expert_returns.median(), crowd_returns.median(), baseline_returns.median()],
    'std_return_pct': [expert_returns.std(), crowd_returns.std(), baseline_returns.std()],
    'variance': [expert_var, crowd_var, baseline_returns.var()],
    'avg_confidence': [expert_confidence.mean(), crowd_confidence.mean(), np.nan],
    'median_confidence': [np.median(expert_confidence), np.median(crowd_confidence), np.nan],
    'std_confidence': [expert_confidence.std(), crowd_confidence.std(), np.nan]
}
metrics_df = pd.DataFrame(metrics_results)
metrics_output = f"{data_dir}/classification_metrics_{timestamp}.csv"
metrics_df.to_csv(metrics_output, index=False, encoding='utf-8')
print(f"분류 지표 결과: {metrics_output}")

# Confidence 구간별 성과 저장
conf_performance = []
for model_name, df_model, returns in [('Expert', df_expert, expert_returns),
                                       ('Crowd', df_crowd_agg, crowd_returns)]:
    df_model = df_model.copy()
    df_model['return'] = returns.values
    df_model['conf_bin'] = pd.cut(df_model['confidence'], bins=conf_bins, labels=conf_labels, include_lowest=True)

    for label in conf_labels:
        subset = df_model[df_model['conf_bin'] == label]
        if len(subset) > 0:
            conf_performance.append({
                'model': model_name,
                'confidence_range': label,
                'count': len(subset),
                'win_rate': (subset['return'] > 0).mean(),
                'avg_return_pct': subset['return'].mean(),
                'total_return_pct': subset['return'].sum(),
                'std_return_pct': subset['return'].std()
            })

conf_perf_df = pd.DataFrame(conf_performance)
conf_output = f"{data_dir}/confidence_performance_{timestamp}.csv"
conf_perf_df.to_csv(conf_output, index=False, encoding='utf-8')
print(f"Confidence 구간별 성과: {conf_output}")

print("\n" + "="*80)
print("분석 완료!")
print("="*80)
