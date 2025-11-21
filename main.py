import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

dataset_path = "dataset.csv"

df = pd.read_csv(dataset_path,encoding='latin1',delimiter=';', parse_dates=True)

df.columns = ['Data', 'Valor'] # mudando as colunas para tratar mais facil

# tratando as linhas do dataset 
df = df[df['Data'].str.match(r'^\d{2}/\d{4}$', na=False)]

df['Data'] = pd.to_datetime(df['Data'], format='%m/%Y')


# Trocando as virgulas por . e mudando o tipo
df['Valor'] = df['Valor'].str.replace('.', '', regex=False)
df['Valor'] = df['Valor'].str.replace(',', '.', regex=False).astype(float)

df = df.set_index('Data')

plt.figure(figsize=(12,5))
plt.plot(df['Valor'])
plt.title('Transações Correntes (US$ milhões)')
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

result = seasonal_decompose(df['Valor'], model='additive', period=12)

result.plot()
plt.show()
