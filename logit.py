import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# -----------------------------------------------------------------
#  Cargar dataset
# -----------------------------------------------------------------
df = pd.read_csv("../dataset_lstm_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------------------------------------------
# Target de regresión y target direccional
# -----------------------------------------------------------------
if "target_t_plus_1" not in df.columns:
    df["target_t_plus_1"] = df["argt_logret"].shift(-1)

df["target_direction"] = (df["target_t_plus_1"] > 0).astype(int)

df = df.dropna().reset_index(drop=True)

# -----------------------------------------------------------------
# Features
# -----------------------------------------------------------------
feature_cols = [
    "argt_logret",
    #"sp500_logret",
    #"petroleo_wti_logret",
    #"usd_ars_logret",
    #"riesgo_pais_logret",
]

target_col = "target_direction"

# -----------------------------------------------------------------
# Split temporal
# -----------------------------------------------------------------
n = len(df)
train_size = int(n * 0.70)
val_size = int(n * 0.15)

train_df = df.iloc[:train_size].copy()
val_df = df.iloc[train_size:train_size + val_size].copy()
test_df = df.iloc[train_size + val_size:].copy()

# -----------------------------------------------------------------
#  Escalado
# -----------------------------------------------------------------
X_scaler = StandardScaler()

X_train_raw = train_df[feature_cols].values
X_val_raw = val_df[feature_cols].values
X_test_raw = test_df[feature_cols].values

y_train_raw = train_df[target_col].values
y_val_raw = val_df[target_col].values
y_test_raw = test_df[target_col].values

X_train_scaled = X_scaler.fit_transform(X_train_raw)
X_val_scaled = X_scaler.transform(X_val_raw)
X_test_scaled = X_scaler.transform(X_test_raw)

# -----------------------------------------------------------------
# Alinear fechas con la LSTM
# La LSTM pierde las primeras 'lookback' filas del test
# -----------------------------------------------------------------
lookback = 30

test_dates = test_df["Date"].iloc[lookback:].reset_index(drop=True)
test_returns = test_df["target_t_plus_1"].iloc[lookback:].reset_index(drop=True)

X_test_scaled = X_test_scaled[lookback:]
y_test = y_test_raw[lookback:]

if len(test_dates) != len(y_test):
    raise ValueError(
        f"No coincide longitud de fechas ({len(test_dates)}) con y_test ({len(y_test)})"
    )

if len(test_returns) != len(y_test):
    raise ValueError(
        f"No coincide longitud de retornos ({len(test_returns)}) con y_test ({len(y_test)})"
    )

# Entrenamos con train + val
#X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
#y_trainval = np.concatenate([y_train_raw, y_val_raw])

# -----------------------------------------------------------------
# Modelo de benchmark: Logistic Regression
# -----------------------------------------------------------------
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=27
)

model.fit(X_train_scaled, y_train_raw)

# -----------------------------------------------------------------
# Predicciones
# -----------------------------------------------------------------
threshold = 0.53
y_prob = model.predict_proba(X_test_scaled)[:, 1] #probabilidad de que la clase sea 1
y_pred = (y_prob >= threshold).astype(int)

# -----------------------------------------------------------------
#  Evaluación clasificación
# -----------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nResultados en test:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC ROC  : {auc:.4f}")

print("\nMatriz de confusión:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================================================
# 10) Baseline clase mayoritaria
# =========================================================
# majority_class = int(pd.Series(y_train_raw).mode()[0])
# baseline_pred = np.full_like(y_test, majority_class)

# baseline_acc = accuracy_score(y_test, baseline_pred)
# baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)

# print("\nBaseline clase mayoritaria:")
# print("Clase predicha siempre:", majority_class)
# print(f"Accuracy: {baseline_acc:.4f}")
# print(f"F1-score: {baseline_f1:.4f}")

# -----------------------------------------------------------
#  Backtest
# y_pred = 1 -> invertido
# y_pred = 0 -> liquidez
# -----------------------------------------------------------------
transaction_cost = 0.00  

bt_df = pd.DataFrame({
    "Date": test_dates.values,
    "y_true_class": y_test,
    "real_logret": test_returns.values,
    "y_prob": y_prob,
    "y_pred": y_pred
})

# Señal
bt_df["signal"] = bt_df["y_pred"]

# Detectar entradas (0 -> 1)
bt_df["entry"] = bt_df["signal"].diff().fillna(bt_df["signal"]).clip(lower=0)

# Retorno de estrategia
bt_df["strategy_logret_gross"] = bt_df["signal"] * bt_df["real_logret"]
bt_df["strategy_logret_net"] = bt_df["strategy_logret_gross"] - bt_df["entry"] * transaction_cost

# Buy & hold
bt_df["buy_hold_logret"] = bt_df["real_logret"]

# Capital acumulado
bt_df["strategy_cum_gross"] = np.exp(bt_df["strategy_logret_gross"].cumsum())
bt_df["strategy_cum_net"] = np.exp(bt_df["strategy_logret_net"].cumsum())
bt_df["buy_hold_cum"] = np.exp(bt_df["buy_hold_logret"].cumsum())

# -----------------------------------------------------------------
#  Métricas del backtest
# -----------------------------------------------------------------
def max_drawdown(wealth_series):
    roll_max = wealth_series.cummax()
    drawdown = wealth_series / roll_max - 1.0
    return drawdown.min()

def annualized_sharpe(logrets, periods_per_year=252):
    mu = logrets.mean()
    sigma = logrets.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return (mu / sigma) * np.sqrt(periods_per_year)

strategy_total_return_gross = bt_df["strategy_cum_gross"].iloc[-1] - 1.0
strategy_total_return_net = bt_df["strategy_cum_net"].iloc[-1] - 1.0
buy_hold_total_return = bt_df["buy_hold_cum"].iloc[-1] - 1.0

days_in_market = bt_df["signal"].mean()
n_entries = int(bt_df["entry"].sum())

invested_mask = bt_df["signal"] == 1
if invested_mask.sum() > 0:
    hit_ratio_invested = (bt_df.loc[invested_mask, "real_logret"] > 0).mean()
else:
    hit_ratio_invested = np.nan

sharpe_strategy = annualized_sharpe(bt_df["strategy_logret_net"])
sharpe_buyhold = annualized_sharpe(bt_df["buy_hold_logret"])

mdd_strategy = max_drawdown(bt_df["strategy_cum_net"])
mdd_buyhold = max_drawdown(bt_df["buy_hold_cum"])

print("\n=== Backtest ===")
print(f"Retorno total estrategia bruto: {strategy_total_return_gross:.4%}")
print(f"Retorno total estrategia neto : {strategy_total_return_net:.4%}")
print(f"Retorno total buy & hold      : {buy_hold_total_return:.4%}")
print(f"% días invertido              : {days_in_market:.2%}")
print(f"Número de entradas            : {n_entries}")
print(f"Hit ratio días invertido      : {hit_ratio_invested:.2%}")
print(f"Sharpe estrategia neta        : {sharpe_strategy:.4f}")
print(f"Sharpe buy & hold             : {sharpe_buyhold:.4f}")
print(f"Max drawdown estrategia       : {mdd_strategy:.4%}")
print(f"Max drawdown buy & hold       : {mdd_buyhold:.4%}")

# -----------------------------------------------------------------
# Guardar predicciones alineadas
# -----------------------------------------------------------------
pred_df = bt_df.copy()
pred_df.to_csv("predicciones_logreg_direccional.csv", index=False)
print("\nArchivo guardado: predicciones_logreg_direccional.csv")

# -----------------------------------------------------------------
#  Guardar coeficientes
# -----------------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coef": model.coef_[0]
})

coef_df["odds_ratio"] = np.exp(coef_df["coef"])
coef_df["abs_coef"] = coef_df["coef"].abs()

coef_df = coef_df.sort_values("abs_coef", ascending=False)

coef_df.to_csv("coeficientes_logreg.csv", index=False)
print("Archivo guardado: coeficientes_logreg.csv")

# -----------------------------------------------------------------
# Gráficas
# -----------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Azar")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(bt_df["Date"], bt_df["buy_hold_cum"], label="Buy & Hold")
plt.plot(bt_df["Date"], bt_df["strategy_cum_gross"], label="Estrategia bruto")
#plt.plot(bt_df["Date"], bt_df["strategy_cum_net"], label="Estrategia neto")
plt.title("Capital acumulado - Benchmark Logistic Regression")
plt.xlabel("Fecha")
plt.ylabel("Capital acumulado")
plt.legend()
plt.tight_layout()
plt.show()
