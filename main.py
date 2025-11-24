import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima
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
print("ANÁLISE COM MODELO ARIMA")
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
# MODELO ARIMA
# =============================================================================

print("\nAJUSTANDO AUTO_ARIMA...")

model_arima = auto_arima(
    y=conjunto_treinamento['Valor'],
    m=12,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    trace=True
)

print("\nMELHOR MODELO ARIMA ENCONTRADO:")
print(model_arima)

# Previsões para o período de teste
forecasting_arima = pd.Series(
    model_arima.predict(n_periods=len(conjunto_teste)),
    index=conjunto_teste.index
)

# =============================================================================
# VISUALIZAÇÃO ARIMA
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

# Previsões ARIMA
forecasting_arima.plot(
    linewidth=2,
    label=f'Previsão ARIMA {model_arima.order}',
    color='red',
    linestyle='--',
    marker='o'
)

plt.title('Transações Correntes - Previsão com ARIMA', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('US$ milhões')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# MÉTRICAS ARIMA
# =============================================================================

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100

mape_arima = MAPE(conjunto_teste['Valor'], forecasting_arima)

print(f"\nPRECISÃO DO MODELO ARIMA:")
print(f"MAPE: {mape_arima:.2f}%")

# =============================================================================
# PREVISÃO FUTURA COM ARIMA
# =============================================================================

print(f"\nPREVISÃO PARA OS PRÓXIMOS 12 MESES (ARIMA):")

# Modelo final com todos os dados
model_final = auto_arima(
    y=df['Valor'],
    m=12,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True
)

future_forecast = model_final.predict(n_periods=12)
future_dates = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq='MS'
)

future_forecast_series = pd.Series(future_forecast, index=future_dates)

print("=" * 50)
for date, value in future_forecast_series.items():
    trend = "CRESCENTE" if value > 0 else "DECRESCENTE"
    print(f"{trend} {date.strftime('%m/%Y')}: {value:>8.1f} milhões USD")

# =============================================================================
# VISUALIZAÇÃO PREVISÃO FUTURA ARIMA
# =============================================================================

plt.figure(figsize=(14, 8))

# Dados históricos (últimos 3 anos)
df['Valor']['2022-01-01':].plot(
    linewidth=2,
    label='Dados Históricos',
    color='blue'
)

# Previsão futura
future_forecast_series.plot(
    linewidth=2,
    label='Previsão ARIMA (Próximos 12 meses)',
    color='red',
    linestyle='--',
    marker='o'
)

plt.axvline(x=df.index[-1], color='gray', linestyle=':', alpha=0.7, label='Fim Dados Reais')

plt.title('Transações Correntes - Previsão Futura com ARIMA', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('US$ milhões')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# RESUMO ARIMA
# =============================================================================

print(f"\n" + "="*60)
print("RESUMO - MODELO ARIMA")
print("="*60)

print(f"• PARÂMETROS: {model_arima.order}")
print(f"• PRECISÃO (MAPE): {mape_arima:.1f}%")
print(f"• TENDÊNCIA: Negativa")
print(f"• SAZONALIDADE: Presente (anual)")
print(f"• PREVISÃO: Déficit persistente")

print(f"\nINSIGHTS ARIMA:")
print("• Modelo captura padrões complexos de sazonalidade")
print("• Adequado para séries com componentes sazonais")
print("• Previsões consideram autocorrelação temporal")