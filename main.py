import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# =============================================================================
# Carregando Dados
# =============================================================================

dataset_path = "dataset.csv"
df = pd.read_csv(dataset_path, encoding='latin1', delimiter=';', parse_dates=True)
df.columns = ['Data', 'Valor']
df = df[df['Data'].str.match(r'^\d{2}/\d{4}$', na=False)]
df['Data'] = pd.to_datetime(df['Data'], format='%m/%Y')
df['Valor'] = df['Valor'].str.replace('.', '', regex=False)
df['Valor'] = df['Valor'].str.replace(',', '.', regex=False).astype(float)
df = df.set_index('Data')
df = df.sort_index()
df = df.asfreq('MS')

print("=" * 60)
print("ANÁLISE COM MODELO HOLT-WINTERS")
print("=" * 60)

print("RESUMO DOS DADOS:")
print(f"Período: {df.index[0].strftime('%m/%Y')} a {df.index[-1].strftime('%m/%Y')}")
print(f"Total de meses: {len(df)}")
print(f"Valor médio: {df['Valor'].mean():.2f}")
print(f"Desvio padrão: {df['Valor'].std():.2f}")

# =============================================================================
# DECOMPOSIÇÃO DA SÉRIE TEMPORAL
# =============================================================================

print("\nDECOMPOSIÇÃO DA SÉRIE TEMPORAL")

# Gráfico da série original
plt.figure(figsize=(12,5))
plt.plot(df['Valor'])
plt.title('Transações Correntes (US$ milhões) - Série Original')
plt.xlabel('Ano')
plt.ylabel('Valor (US$ milhões)')
plt.grid(True)
plt.show()

# Decomposição sazonal
result = seasonal_decompose(df['Valor'], model='additive', period=12)
result.plot()
plt.suptitle('Decomposição da Série Temporal - Modelo Aditivo', y=1.02)
plt.tight_layout()
plt.show()

# =============================================================================
# ANÁLISE DA DECOMPOSIÇÃO
# =============================================================================

print("\nANÁLISE DA DECOMPOSIÇÃO:")
print("-" * 40)

# Verificar tendência
trend_present = not result.trend.dropna().empty
if trend_present:
    avg_trend = result.trend.mean()
    print(f"✓ TENDÊNCIA: Presente")
    print(f"  Direção: {'NEGATIVA' if avg_trend < 0 else 'POSITIVA'}")
    print(f"  Valor médio: {avg_trend:.2f}")

# Verificar sazonalidade
seasonal_present = not result.seasonal.dropna().empty
if seasonal_present:
    seasonal_strength = result.seasonal.std() / df['Valor'].std()
    print(f"✓ SAZONALIDADE: Presente")
    print(f"  Força: {seasonal_strength:.2%}")
    print(f"  Período: 12 meses (anual)")

# =============================================================================
# DIVISÃO TREINO/TESTE
# =============================================================================

meses_teste = 12
conjunto_treinamento = df.iloc[:-meses_teste]
conjunto_teste = df.iloc[-meses_teste:]

print(f"\nDIVISÃO TREINO/TESTE:")
print(f"Treino: {conjunto_treinamento.index[0].strftime('%m/%Y')} até {conjunto_treinamento.index[-1].strftime('%m/%Y')} ({len(conjunto_treinamento)} meses)")
print(f"Teste:  {conjunto_teste.index[0].strftime('%m/%Y')} até {conjunto_teste.index[-1].strftime('%m/%Y')} ({len(conjunto_teste)} meses)")

# =============================================================================
# MODELO HOLT-WINTERS
# =============================================================================

print("\nAJUSTANDO HOLT-WINTERS...")

model_hw = ExponentialSmoothing(
    endog=conjunto_treinamento['Valor'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

print("\nMODELO HOLT-WINTERS AJUSTADO:")
print(f"Parâmetros: Trend='add', Seasonal='add', Periods=12")

# Previsões para o período de teste
forecasting_hw = model_hw.forecast(steps=len(conjunto_teste))

# =============================================================================
# VISUALIZAÇÃO HOLT-WINTERS
# =============================================================================

plt.figure(figsize=(14, 8))

# Dados históricos
conjunto_treinamento['Valor']['2020-01-01':].plot(
    linewidth=2,
    label='Dados de Treinamento',
    color='blue'
)

# Dados de teste reais
conjunto_teste['Valor'].plot(
    linewidth=2,
    label='Dados Reais (Teste)',
    color='green'
)

# Previsões Holt-Winters
forecasting_hw.plot(
    linewidth=2,
    label='Previsão Holt-Winters',
    color='orange',
    linestyle='--',
    marker='o'
)

plt.title('Transações Correntes - Previsão com Holt-Winters', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('US$ milhões')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# MÉTRICAS HOLT-WINTERS
# =============================================================================

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100

mape_hw = MAPE(conjunto_teste['Valor'], forecasting_hw)

print(f"\nPRECISÃO DO MODELO HOLT-WINTERS:")
print(f"MAPE: {mape_hw:.2f}%")

# =============================================================================
# PREVISÃO FUTURA COM HOLT-WINTERS
# =============================================================================

print(f"\nPREVISÃO PARA OS PRÓXIMOS 12 MESES (HOLT-WINTERS):")

# Modelo final com todos os dados
model_final_hw = ExponentialSmoothing(
    endog=df['Valor'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

future_forecast_hw = model_final_hw.forecast(steps=12)
future_dates = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq='MS'
)

future_forecast_series_hw = pd.Series(future_forecast_hw, index=future_dates)

print("=" * 50)
for date, value in future_forecast_series_hw.items():
    trend = "CRESCENTE" if value > 0 else "DECRESCENTE"
    print(f"{trend} {date.strftime('%m/%Y')}: {value:>8.1f} milhões USD")

# =============================================================================
# VISUALIZAÇÃO PREVISÃO FUTURA HOLT-WINTERS
# =============================================================================

plt.figure(figsize=(14, 8))

# Dados históricos (últimos 3 anos)
df['Valor']['2022-01-01':].plot(
    linewidth=2,
    label='Dados Históricos',
    color='blue'
)

# Previsão futura
future_forecast_series_hw.plot(
    linewidth=2,
    label='Previsão Holt-Winters (Próximos 12 meses)',
    color='orange',
    linestyle='--',
    marker='o'
)

plt.axvline(x=df.index[-1], color='gray', linestyle=':', alpha=0.7, label='Fim Dados Reais')

plt.title('Transações Correntes - Previsão Futura com Holt-Winters', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('US$ milhões')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# RESUMO HOLT-WINTERS
# =============================================================================

print(f"\n" + "="*60)
print("RESUMO - MODELO HOLT-WINTERS")
print("="*60)

print(f"• PARÂMETROS: Trend='add', Seasonal='add', Periods=12")
print(f"• PRECISÃO (MAPE): {mape_hw:.1f}%")
print(f"• TENDÊNCIA: Negativa")
print(f"• SAZONALIDADE: Presente (anual)")
print(f"• PREVISÃO: Déficit persistente")

print(f"\nINSIGHTS HOLT-WINTERS:")
print("• Modelo explícito para tendência e sazonalidade")
print("• Adequado para séries com padrões sazonais estáveis")
print("• Simples de interpretar e implementar")
print("• Menos sensível a outliers que ARIMA")