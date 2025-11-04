# Air Quality Analysis: Elasticidade do Ozônio

Este projeto realiza uma análise da relação entre os poluentes atmosféricos NO₂ (Dióxido de Nitrogênio) e O₃ (Ozônio) nas áreas urbanas e periféricas dos Estados Unidos, entre 2000 e 2016. A hipótese central, chamada **Elasticidade do Ozônio**, investiga como a redução do NO₂ afeta as concentrações de O₃ nas diferentes regiões (urbanas versus rurais/periféricas), considerando os efeitos não-lineares e as interações químicas que variam com a localização.

## Descrição do Projeto

O principal objetivo deste estudo foi investigar a dinâmica entre os poluentes primários e secundários em ambientes urbanos e periféricos, validando a hipótese de que a redução do NO₂ nas áreas urbanas pode, paradoxalmente, ter um efeito menor ou até contrário no aumento de O₃ devido ao desequilíbrio químico (efeito de titulação). Já nas áreas periféricas, o aumento de O₃ é governado principalmente pelo transporte de precursores e fatores climáticos, como a temperatura.

### Etapas do Projeto

1. **Importação e Limpeza dos Dados:**

   * O conjunto de dados contém informações sobre a qualidade do ar em várias localidades dos EUA de 2000 a 2016, incluindo medições de NO₂, O₃, SO₂ e CO.
   * A limpeza envolveu a remoção de colunas desnecessárias e o tratamento de valores ausentes, além da conversão das colunas de data para o formato apropriado.

2. **Exploração e Visualização de Dados:**

   * O uso de gráficos como boxplots e heatmaps foi fundamental para entender a distribuição temporal e geográfica das concentrações de NO₂ e O₃.
   * As visualizações também permitiram analisar a correlação entre NO₂ e O₃ nas diferentes zonas (urbana e periférica) ao longo dos anos.

3. **Criação de Zonas:**

   * A partir da média de NO₂ por condado, foram definidas zonas: **Urbana (Alta NO₂)**, **Rural/Periférica (Baixa NO₂)** e **Outra**.
   * A classificação permitiu comparar o comportamento das concentrações de O₃ entre essas regiões e entender como a elasticidade do ozônio se manifesta de maneira diferente em cada contexto.

4. **Cálculo de Correlação entre NO₂ e O₃:**

   * A correlação entre NO₂ e O₃ foi calculada para cada ano e zona, mostrando como a relação entre esses poluentes evoluiu ao longo do tempo.
   * A análise revelou uma transição significativa nas zonas periféricas, onde a correlação entre NO₂ e O₃ se tornou positiva após 2009, indicando uma mudança no regime químico.

5. **Análise e Conclusões:**

   * O comportamento observado nas áreas urbanas segue o **efeito de titulação**, onde o aumento de NO₂ (devido ao tráfego e emissões industriais) resulta em uma diminuição da concentração de O₃.
   * Nas áreas rurais/periféricas, a correlação entre NO₂ e O₃ se inverteu a partir de 2009, evidenciando que essas zonas se tornaram limitadas por NO₂, o que levou a um aumento no O₃ devido à maior formação atmosférica do poluente secundário.

6. **Exploração Adicional:**

   * Por fim, foi sugerido um estudo adicional para calcular a média de SO₂ por Estado, visando identificar regiões industriais e correlacionar com a poluição geral.

## Instalação e Uso

### Requisitos

* **Python 3.x**
* **Bibliotecas:**

  * pandas
  * matplotlib
  * seaborn
  * numpy

### Como rodar o código

1. **Instalar as dependências:**

   * Utilize o `pip` para instalar as bibliotecas necessárias:

   ```bash
   pip install pandas matplotlib seaborn numpy
   ```

2. **Carregar e Processar os Dados:**

   * O dataset pode ser carregado com o seguinte comando:

   ```python
   import pandas as pd
   df = pd.read_csv('caminho/para/seu/arquivo.csv')
   ```

3. **Gerar Gráficos de Poluição:**

   * O código pode gerar gráficos de boxplot e heatmaps para diferentes poluentes utilizando a função `pollution_graphs`:

   ```python
   pollution_graphs(df, metric='NO2 Mean', group_col=['State', 'Year'])
   ```

4. **Análise de Correlação:**

   * Para calcular e visualizar a correlação entre O₃ e NO₂ por zona e ano, utilize o seguinte comando:

   ```python
   sns.lineplot(data=df_correlacao, x='Year', y='Correlacao_O3_NO2', hue='Zona', marker='o')
   ```

### Exemplos de Resultados Esperados

* **Gráficos de Boxplot:**

  * Distribuição de NO₂, O₃, SO₂ e CO por Estado e Ano.
  * Heatmaps que mostram a evolução de cada poluente por Estado e Ano.

* **Análise de Correlação:**

  * Um gráfico de linha que ilustra a evolução da correlação entre O₃ e NO₂ ao longo do tempo para as zonas Urbana e Rural/Periférica.

## Conclusões e Interpretações

* **Zona Urbana (Alta NO₂):**

  * A correlação entre NO₂ e O₃ permaneceu negativamente correlacionada durante todo o período analisado, validando o efeito de titulação.
  * A alta concentração de NO₂ nas áreas urbanas tem um impacto direto na destruição do ozônio, resultando em uma relação inversa entre ambos os poluentes.

* **Zona Rural/Periférica (Baixa NO₂):**

  * Inicialmente, a correlação foi negativa, mas a partir de 2009 a correlação se tornou positiva, refletindo uma mudança no regime químico.
  * Em áreas com menores concentrações de NO₂, o aumento do precursor NO₂ passa a favorecer a formação de O₃, evidenciando o comportamento de elasticidade.

### Implicações Políticas

* **Políticas Urbanas:**

  * As políticas de controle de emissões de NO₂ nas áreas urbanas têm sido eficazes em reduzir a concentração de NO₂, mas o excesso de NOx ainda afeta negativamente a qualidade do ar, resultando em menor formação de O₃.
* **Políticas para Áreas Periféricas:**

  * Em regiões rurais, a sensibilidade à poluição de NO₂ está aumentando, o que exige políticas que abordem o controle de emissões de NOx, especialmente provenientes de áreas urbanas vizinhas, para mitigar a poluição de O₃.
