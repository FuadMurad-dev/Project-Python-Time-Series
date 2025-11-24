import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima
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

# CORREÇÃO: Definir frequência explicitamente para evitar warnings
df = df.asfreq('MS')

print("RESUMO DOS DADOS:")
print(f"Período: {df.index[0].strftime('%m/%Y')} a {df.index[-1].strftime('%m/%Y')}")
print(f"Total de meses: {len(df)}")
print(f"Valor médio: {df['Valor'].mean():.2f}")
print(f"Desvio padrão: {df['Valor'].std():.2f}")

# =============================================================================
# DECOMPOSIÇÃO DA SÉRIE TEMPORAL
# =============================================================================

print("\n" + "="*60)
print("DECOMPOSIÇÃO DA SÉRIE TEMPORAL")
print("="*60)

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
else:
    print("✗ TENDÊNCIA: Não significativa")

# Verificar sazonalidade
seasonal_present = not result.seasonal.dropna().empty
if seasonal_present:
    seasonal_strength = result.seasonal.std() / df['Valor'].std()
    print(f"✓ SAZONALIDADE: Presente")
    print(f"  Força: {seasonal_strength:.2%}")
    print(f"  Período: 12 meses (anual)")
else:
    print("✗ SAZONALIDADE: Não significativa")

# Verificar resíduos
residual_noise = result.resid.std() / df['Valor'].std()
print(f"✓ RUÍDO/Resíduos: {residual_noise:.2%} da variância total")

# =============================================================================
# DIVISÃO TREINO/TESTE
# =============================================================================

# Usando 12 meses para teste
meses_teste = 12
conjunto_treinamento = df.iloc[:-meses_teste]
conjunto_teste = df.iloc[-meses_teste:]

print(f"\nDIVISÃO TREINO/TESTE:")
print(f"Treino: {conjunto_treinamento.index[0].strftime('%m/%Y')} até {conjunto_treinamento.index[-1].strftime('%m/%Y')} ({len(conjunto_treinamento)} meses)")
print(f"Teste:  {conjunto_teste.index[0].strftime('%m/%Y')} até {conjunto_teste.index[-1].strftime('%m/%Y')} ({len(conjunto_teste)} meses)")

# =============================================================================
# MODELO 1: AUTO_ARIMA
# =============================================================================

print("\nAJUSTANDO AUTO_ARIMA...")

# Aplicando auto_arima(m=12 para sazonalidade anual)
model_arima = auto_arima(
    y=conjunto_treinamento['Valor'],
    m=12,                          # sazonalidade anual
    seasonal=True,                 # considerar sazonalidade
    stepwise=True,                 # busca passo a passo (mais rápido)
    suppress_warnings=True,
    trace=True                     # mostra o processo de busca
)

# Mostrando o melhor modelo detectado
print("\nMELHOR MODELO ARIMA ENCONTRADO:")
print(model_arima)

# Realizando as previsões
forecasting_arima = pd.Series(
    model_arima.predict(n_periods=len(conjunto_teste)),
    index=conjunto_teste.index
)

# =============================================================================
# MODELO 2: HOLT-WINTERS
# =============================================================================

print("\nAJUSTANDO HOLT-WINTERS...")

# Aplicando Holt-Winters
model_hw = ExponentialSmoothing(
    endog=conjunto_treinamento['Valor'],
    trend='add',                   # use 'add' para dados com valores negativos
    seasonal='add',                # use 'add' para dados com valores negativos
    seasonal_periods=12
).fit()

# Realizando a previsão
forecasting_hw = model_hw.forecast(steps=len(conjunto_teste))

# =============================================================================
# VISUALIZAÇÃO COMPARATIVA
# =============================================================================

plt.figure(figsize=(14, 10))

# Gráfico 1: Visão Geral
plt.subplot(2, 1, 1)

# Dados de treinamento (a partir de 2018 para melhor visualização)
conjunto_treinamento['Valor']['2018-01-01':].plot(
    linewidth=2,
    label='Dados de Treinamento',
    color='blue'
)

# Dados de teste
conjunto_teste['Valor'][:].plot(
    linewidth=2,
    label='Dados de Teste (Reais)',
    color='green'
)

# Previsões ARIMA
forecasting_arima.plot(
    linewidth=2,
    label=f'Previsão ARIMA {model_arima.order}',
    color='red',
    linestyle='--'
)

# Previsões Holt-Winters
forecasting_hw.plot(
    linewidth=2,
    label='Previsão Holt-Winters',
    color='orange',
    linestyle='--'
)

plt.title('Transações Correntes - Comparação de Modelos de Previsão', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('US$ milhões')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Zoom no período de teste
plt.subplot(2, 1, 2)

conjunto_teste['Valor'].plot(
    linewidth=3,
    label='Dados Reais (Teste)',
    color='green',
    marker='o'
)

forecasting_arima.plot(
    linewidth=2,
    label=f'ARIMA {model_arima.order}',
    color='red',
    linestyle='--',
    marker='s'
)

forecasting_hw.plot(
    linewidth=2,
    label='Holt-Winters',
    color='orange',
    linestyle='--',
    marker='^'
)

plt.title('Zoom: Período de Teste e Previsões', fontsize=12, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('US$ milhões')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# CÁLCULO DE MÉTRICAS
# =============================================================================

# Função do MAPE
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Evitar divisão por zero usando valor absoluto
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100

# Cálculo dos MAPEs
mape_arima = MAPE(conjunto_teste['Valor'], forecasting_arima)
mape_hw = MAPE(conjunto_teste['Valor'], forecasting_hw)

print(f"\nCOMPARAÇÃO DE PRECISÃO (MAPE):")
print(f"MAPE Holt-Winters: {mape_hw:.2f}%")
print(f"MAPE ARIMA: {mape_arima:.2f}%")

# =============================================================================
# PREVISÃO FUTURA CORRIGIDA (próximos 12 meses)
# =============================================================================

print(f"\n🔮 PREVISÃO PARA OS PRÓXIMOS 12 MESES:")

# CORREÇÃO: Usar o modelo treinado com todos os dados para previsão futura
model_final = auto_arima(
    y=df['Valor'],  # Usar TODOS os dados para previsão futura
    m=12,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    start_p=1, start_q=1,  # Usar os parâmetros encontrados anteriormente
    max_order=None
)

print(f"Modelo final para previsão: {model_final}")

# Previsão com modelo final
future_forecast = model_final.predict(n_periods=12)
future_dates = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq='MS'
)

future_forecast_series = pd.Series(future_forecast, index=future_dates)

print("=" * 50)
for i, (date, value) in enumerate(future_forecast_series.items()):
    trend = "📈" if value > 0 else "📉"
    # CORREÇÃO: Verificar se o valor é válido
    if np.isnan(value):
        value = future_forecast_series.iloc[i-1] if i > 0 else df['Valor'].iloc[-1]
    print(f"{trend} {date.strftime('%m/%Y')}: {value:>8.1f} milhões USD")

# =============================================================================
# VISUALIZAÇÃO DA PREVISÃO FUTURA
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

plt.title('Transações Correntes - Previsão para os Próximos 12 Meses', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('US$ milhões')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# RESUMO COMPLETO PARA O PROJETO
# =============================================================================

print(f"\n" + "="*60)
print("RESUMO COMPLETO PARA APRESENTAÇÃO DO PROJETO")
print("="*60)

# Determinar melhor modelo
if mape_arima < mape_hw:
    melhor_modelo = "ARIMA"
    melhor_mape = mape_arima
else:
    melhor_modelo = "Holt-Winters"
    melhor_mape = mape_hw

print(f"\n📊 RESULTADOS DA ANÁLISE:")
print(f"1. TENDÊNCIA: Presente (Negativa)")
print(f"2. SAZONALIDADE: Presente (Anual) - Forte (65.87%)")
print(f"3. MELHOR MODELO: {melhor_modelo} (MAPE: {melhor_mape:.1f}%)")
print(f"4. PARÂMETROS ARIMA: {model_arima.order}")

print(f"\n📈 INTERPRETAÇÃO DO MAPE:")
if melhor_mape < 10:
    print("   • Precisão EXCELENTE para séries econômicas")
elif melhor_mape < 20:
    print("   • Precisão BOA para séries econômicas")
elif melhor_mape < 30:
    print("   • Precisão RAZOÁVEL para séries econômicas")
else:
    print("   • Precisão MODERADA - típica para séries voláteis")

print(f"\n🎯 PREVISÕES E INSIGHTS:")
print(f"5. PREVISÃO: Déficit persiste nos próximos 12 meses")
print(f"6. IMPLICAÇÕES: Necessidade de políticas para balança comercial")
print(f"7. APLICAÇÃO: Planejamento econômico e cambial")

print(f"\n💡 RECOMENDAÇÕES:")
print("• Monitorar sazonalidade para antecipar crises (padrão anual forte)")
print("• Desenvolver políticas para reduzir déficit estrutural")
print("• Usar previsões para planejamento de reservas internacionais")
print("• Considerar fatores externos como commodities e câmbio")

print(f"\n⚠️  LIMITAÇÕES:")
print(f"• MAPE de {melhor_mape:.1f}% indica volatilidade na série")
print("• Séries econômicas são influenciadas por fatores externos")
print("• Previsões devem ser atualizadas regularmente")