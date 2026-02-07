import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# =========================
# CARREGAMENTO DO DATASET
# =========================
df = pd.read_csv("dataset.csv", encoding='latin1', delimiter=';')
df.columns = ['Data', 'Valor']

df = df[df['Data'].str.match(r'^\d{2}/\d{4}$', na=False)]
df['Data'] = pd.to_datetime(df['Data'], format='%m/%Y')

df['Valor'] = (
    df['Valor']
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .astype(float)
)

df = df.set_index('Data').sort_index().asfreq('MS')

# =========================
# TRAÇAR A SÉRIE
# =========================
plt.figure(figsize=(12,5))
plt.plot(df['Valor'])
plt.title('Série Original')
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

# =========================
# DECOMPOSIÇÃO
# =========================
decomp = seasonal_decompose(df['Valor'], model='additive', period=12)
decomp.plot()
plt.suptitle("Decomposição da Série")
plt.tight_layout()
plt.show()

# =========================
# DIVISÃO TREINO/TESTE
# =========================
train = df.iloc[:-12]
test = df.iloc[-12:]

# =========================
# MODELO ARIMA
# =========================
model = auto_arima(train['Valor'], seasonal=True, m=12, stepwise=True)
pred = model.predict(n_periods=12)

# =========================
# PLOT PREVISÃO
# =========================
plt.figure(figsize=(12,5))
plt.plot(train['Valor'], label="Treino")
plt.plot(test['Valor'], label="Teste")
plt.plot(test.index, pred, label="Previsão ARIMA", linestyle="--")
plt.title("Previsão ARIMA")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# PREVISÃO FUTURA
# =========================
future_model = auto_arima(df['Valor'], seasonal=True, m=12)
future_pred = future_model.predict(n_periods=12)

future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1),
                             periods=12, freq='MS')

future_series = pd.Series(future_pred, index=future_dates)
print("\nPREVISÃO FUTURA (12 meses):")
print(future_series)

# =========================
# MODELO HOLT-WINTERS
# =========================
hw_model = ExponentialSmoothing(train['Valor'], trend='add', seasonal='add', seasonal_periods=12)
hw_fitted = hw_model.fit()
hw_pred = hw_fitted.forecast(12)

# =========================
# CÁLCULO DA PRECISÃO
# =========================
def calcular_precisao(real, previsto):
    erro_percentual = np.mean(np.abs((real - previsto) / real)) * 100
    return 100 - erro_percentual

precisao_arima = calcular_precisao(test['Valor'].values, pred)
precisao_hw = calcular_precisao(test['Valor'].values, hw_pred)

# =========================
# RESULTADOS
# =========================
print("\n=== COMPARAÇÃO DOS MODELOS ===")
print(f"ARIMA: {precisao_arima:.1f}% de precisão")
print(f"Holt-Winters: {precisao_hw:.1f}% de precisão")

if precisao_arima > precisao_hw:
    diferenca = precisao_arima - precisao_hw
    print(f"\n O modelo ARIMA é {diferenca:.1f}% melhor que o Holt-Winters")
else:
    diferenca = precisao_hw - precisao_arima
    print(f"\n O modelo Holt-Winters é {diferenca:.1f}% melhor que o ARIMA")