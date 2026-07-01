import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam


# =========================================================
# 1) SEMILLA
# =========================================================

def set_seed(seed: int):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================================================
# 2) CREAR SECUENCIAS
# =========================================================

def create_sequences(X, y, lookback=20):

    Xs = []
    ys = []

    for i in range(lookback, len(X)):

        Xs.append(X[i - lookback:i])
        ys.append(y[i])

    return np.array(Xs), np.array(ys)


# =========================================================
# 3) BACKTEST
# =========================================================

def run_backtest(
    real_logret,
    y_pred,
    transaction_cost=0.001
):

    bt = pd.DataFrame({

        "real_logret": real_logret,
        "signal": y_pred
    })

    # =====================================
    # ENTRADAS
    # =====================================

    bt["entry"] = (

        bt["signal"]
        .diff()
        .fillna(bt["signal"])
        .clip(lower=0)
    )

    # =====================================
    # RETORNOS
    # =====================================

    bt["strategy_logret_gross"] = (
        bt["signal"] * bt["real_logret"]
    )

    bt["strategy_logret_net"] = (

        bt["strategy_logret_gross"]
        - bt["entry"] * transaction_cost
    )

    bt["buy_hold_logret"] = bt["real_logret"]

    # =====================================
    # CURVAS ACUMULADAS
    # =====================================

    bt["strategy_cum_net"] = np.exp(
        bt["strategy_logret_net"].cumsum()
    )

    bt["buy_hold_cum"] = np.exp(
        bt["buy_hold_logret"].cumsum()
    )

    # =====================================
    # SHARPE
    # =====================================

    def sharpe(x, periods_per_year=252):

        mu = x.mean()

        sigma = x.std(ddof=1)

        if sigma == 0 or np.isnan(sigma):

            return np.nan

        return (
            (mu / sigma)
            * np.sqrt(periods_per_year)
        )

    # =====================================
    # RESULTADOS
    # =====================================

    return {

        "strategy_total_return_net":
            bt["strategy_cum_net"].iloc[-1] - 1.0,

        "buy_hold_total_return":
            bt["buy_hold_cum"].iloc[-1] - 1.0,

        "sharpe_strategy":
            sharpe(bt["strategy_logret_net"]),

        "days_in_market":
            bt["signal"].mean(),

        "strategy_curve":
            bt["strategy_cum_net"].values,

        "buy_hold_curve":
            bt["buy_hold_cum"].values
    }


# =========================================================
# 4) CARGAR DATOS
# =========================================================

df = pd.read_csv("../dataset_lstm_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

# =====================================
# TARGET
# =====================================

if "target_t_plus_1" not in df.columns:

    df["target_t_plus_1"] = (
        df["argt_logret"].shift(-1)
    )

df["target_direction"] = (
    df["target_t_plus_1"] > 0
).astype(int)

df = df.dropna().reset_index(drop=True)

# =====================================
# FEATURES
# =====================================

feature_cols = [

    "argt_logret",
    #"sp500_logret",
    #"petroleo_wti_logret",
    #"usd_ars_logret",
    #"riesgo_pais_logret",
]

target_col = "target_direction"

# =========================================================
# 5) SPLITS
# =========================================================

n = len(df)

train_size = int(n * 0.70)

val_size = int(n * 0.15)

train_df = df.iloc[:train_size].copy()

val_df = df.iloc[
    train_size:train_size + val_size
].copy()

test_df = df.iloc[
    train_size + val_size:
].copy()

# =========================================================
# 6) ESCALADO
# =========================================================

X_scaler = StandardScaler()

X_train_raw = train_df[feature_cols].values
X_val_raw = val_df[feature_cols].values
X_test_raw = test_df[feature_cols].values

y_train_raw = train_df[target_col].values
y_val_raw = val_df[target_col].values
y_test_raw = test_df[target_col].values

X_train_scaled = X_scaler.fit_transform(
    X_train_raw
)

X_val_scaled = X_scaler.transform(
    X_val_raw
)

X_test_scaled = X_scaler.transform(
    X_test_raw
)

# =========================================================
# 7) SECUENCIAS
# =========================================================

lookback = 30

X_train, y_train = create_sequences(
    X_train_scaled,
    y_train_raw,
    lookback
)

X_val, y_val = create_sequences(
    X_val_scaled,
    y_val_raw,
    lookback
)

X_test, y_test = create_sequences(
    X_test_scaled,
    y_test_raw,
    lookback
)

# =====================================
# RETORNOS TEST
# =====================================

test_returns = (

    test_df["target_t_plus_1"]
    .iloc[lookback:]
    .reset_index(drop=True)
    .values
)

# =====================================
# FECHAS TEST
# =====================================

test_dates = (

    test_df["Date"]
    .iloc[lookback:]
    .reset_index(drop=True)
)

# =========================================================
# 8) MONTE CARLO
# =========================================================

n_runs = 30

threshold = 0.53

results = []

all_curves = []

buy_hold_curve = None

# =========================================================
# 9) LOOP
# =========================================================

for seed in range(1, n_runs + 1):

    print(f"\nSimulación {seed}/{n_runs}")

    set_seed(seed)

    # =====================================
    # MODELO
    # =====================================

    model = Sequential([

        Input(
            shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        ),

        LSTM(32),

        Dropout(0.2),

        Dense(16, activation="relu"),

        Dense(1, activation="sigmoid")
    ])

    # =====================================
    # COMPILAR
    # =====================================

    model.compile(

        optimizer=Adam(
            learning_rate=1e-3
        ),

        loss="binary_crossentropy",

        metrics=["accuracy"]
    )

    # =====================================
    # CALLBACKS
    # =====================================

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
    ]

    # =====================================
    # TRAIN
    # =====================================

    model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=100,

        batch_size=16,

        callbacks=callbacks,

        verbose=0
    )

    # =====================================
    # PREDICCIONES
    # =====================================

    y_prob = model.predict(
        X_test,
        verbose=0
    ).flatten()

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # =====================================
    # METRICAS
    # =====================================

    acc = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    # =====================================
    # BACKTEST
    # =====================================

    bt_metrics = run_backtest(
        test_returns,
        y_pred,
        transaction_cost=0.00
    )

    # =====================================
    # GUARDAR CURVAS
    # =====================================

    all_curves.append(
        bt_metrics["strategy_curve"]
    )

    if buy_hold_curve is None:

        buy_hold_curve = (
            bt_metrics["buy_hold_curve"]
        )

    # =====================================
    # RESULTADOS
    # =====================================

    results.append({

        "seed": seed,

        "accuracy": acc,

        "f1": f1,

        "auc": auc,

        **{
            k: v
            for k, v in bt_metrics.items()
            if not isinstance(v, np.ndarray)
        }
    })

# =========================================================
# 10) DATAFRAME RESULTADOS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "simulaciones_lstm.csv",
    index=False
)

# =========================================================
# 11) RESULTADOS
# =========================================================

print("\n=== RESULTADOS ===")

print(results_df)

print("\n=== ESTADISTICAS ===")

print(

    results_df.describe(

        percentiles=[
            0.05,
            0.25,
            0.5,
            0.75,
            0.95
        ]

    ).T
)

# =========================================================
# 12) GRAFICO MONTE CARLO
# =========================================================

plt.figure(figsize=(15, 8))

# =====================================
# TODAS LAS CURVAS
# =====================================

for curve in all_curves:

    plt.plot(

        test_dates,

        curve,

        color="royalblue",

        alpha=0.15,

        linewidth=1.5
    )

# =====================================
# MEDIA LSTM
# =====================================

mean_curve = np.mean(
    all_curves,
    axis=0
)

plt.plot(

    test_dates,

    mean_curve,

    color="blue",

    linewidth=2,

    label="Media LSTM"
)

# =====================================
# PERCENTILES
# =====================================

p5 = np.percentile(
    all_curves,
    5,
    axis=0
)

p95 = np.percentile(
    all_curves,
    95,
    axis=0
)

plt.fill_between(

    test_dates,

    p5,

    p95,

    color="blue",

    alpha=0.15,

    label="Percentil 5-95"
)

# =====================================
# BUY & HOLD
# =====================================

plt.plot(

    test_dates,

    buy_hold_curve,

    color="black",

    linewidth=2,

    linestyle="--",

    label="Buy & Hold"
)

# =========================================================
# FORMATO FECHAS
# =========================================================

ax = plt.gca()

ax.xaxis.set_major_locator(
    mdates.YearLocator()
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y')
)

plt.xticks(rotation=45)

# =========================================================
# ESTETICA
# =========================================================

plt.title(
    "Monte Carlo LSTM vs Buy & Hold",
    fontsize=20
)

plt.xlabel(
    "Fecha",
    fontsize=14
)

plt.ylabel(
    "Capital acumulado",
    fontsize=14
)

plt.legend(fontsize=12)

plt.grid(True)

plt.tight_layout()

plt.show()
