
# Análise de Sentimentos e Tópicos em Avaliações de Companhias Aéreas

Este projeto realiza uma análise detalhada de avaliações de passageiros sobre companhias aéreas, aplicando técnicas de processamento de linguagem natural (PLN), análise de sentimentos, modelagem de tópicos e visualização de dados. O objetivo principal é identificar padrões de sentimentos nas avaliações, identificar tópicos mais comuns e gerar insights sobre o desempenho de diferentes companhias aéreas.

## Tecnologias Utilizadas

* **Pandas**: Para manipulação e análise de dados.
* **NLTK**: Para processamento de texto e análise de sentimentos.
* **scikit-learn**: Para pré-processamento e modelagem de tópicos (LDA).
* **WordCloud**: Para geração de nuvens de palavras.
* **Matplotlib**: Para visualização de dados e gráficos.
* **Latent Dirichlet Allocation (LDA)**: Para modelagem de tópicos.

## Pré-requisitos

* Python 3.x
* Bibliotecas:

  * `nltk`
  * `pandas`
  * `sklearn`
  * `matplotlib`
  * `wordcloud`
  * `datetime`
  * `pathlib`

Instale as dependências com o seguinte comando:

```bash
pip install nltk pandas scikit-learn matplotlib wordcloud
```

## Como Usar

1. **Carregar os Dados**
   O script começa carregando um arquivo CSV com as avaliações das companhias aéreas. O arquivo CSV deve conter colunas como: `Review Date`, `Reviews`, `Verified`, `Recommended`, `Inflight Entertainment`, `Class`, `Type of Traveller`, `Airline`, entre outras.

2. **Limpeza e Pré-processamento**

   * A data de revisão (`Review Date`) é convertida para o formato `datetime`.
   * A coluna `Verified` é padronizada.
   * O campo `Month Flown` é transformado em um formato de data e o ano e mês são extraídos.
   * A coluna `Recommended` é convertida para valores binários (0 ou 1).

3. **Análise de Sentimentos**
   Usamos o **VADER** para realizar a análise de sentimentos nas avaliações. As avaliações são classificadas como `Positivo`, `Negativo` ou `Neutro` com base no escore de polaridade.

4. **Modelagem de Tópicos**
   Utilizamos o **CountVectorizer** para criar uma matriz de termos e depois aplicamos o **Latent Dirichlet Allocation (LDA)** para identificar os tópicos mais prevalentes nas avaliações.

5. **Análise de Sentimento por Tópico**
   A distribuição de sentimentos é calculada para cada tópico identificado pela modelagem LDA. Uma tabela de contingência é gerada para mostrar a porcentagem de sentimentos positivos, negativos e neutros por tópico.

6. **Análise Quantitativa por Segmento**
   A análise segmenta os dados por categorias como `Class`, `Type of Traveller`, e `Airline`, mostrando como o sentimento varia entre diferentes grupos.

7. **Geração de Nuvem de Palavras**
   Para tópicos negativos específicos, uma nuvem de palavras é gerada para identificar as palavras mais frequentemente mencionadas nas avaliações. Isso pode ser útil para descobrir os principais pontos de insatisfação dos passageiros.

8. **Score de Risco Operacional (SRO)**
   Calcula-se um **Score de Risco Operacional (SRO)**, baseado na presença de palavras-chave como "Atraso", "Problema", "Reclamação", entre outras. Esse score é usado para quantificar o risco associado a uma operação aérea, com base nas avaliações negativas.

9. **Salvamento de Resultados**
   Os resultados da análise são salvos em arquivos CSV e gráficos em formato PNG. As saídas incluem:

   * Tabela de análise de sentimentos por tópico.
   * Gráficos de distribuição de sentimentos por tópico.
   * Nuvem de palavras das reclamações negativas.

## Exemplo de Resultados

* **Tabela de Análise de Sentimentos por Tópico**:
  A tabela mostra a distribuição percentual de sentimentos (Positivo, Negativo, Neutro) para cada tópico identificado na análise de LDA.

* **Gráfico de Sentimentos por Tópico**:
  Um gráfico de barras que ilustra a distribuição de sentimentos (Positivo, Negativo, Neutro) para cada tópico.

* **Nuvem de Palavras**:
  A nuvem de palavras exibe as palavras mais comuns nas avaliações negativas de um tópico específico.

## Exemplos de Saída

Aqui está um exemplo de uma saída da análise de sentimentos por tópico:

```
| Tópico                                      | % Sentimento Negativo | Total Reviews |
|---------------------------------------------|-----------------------|---------------|
| Experiência em Voo (Assento/Comida)        | 38.5%                 | 1200          |
| Problemas em Solo (Atrasos/Aeroporto)      | 25.2%                 | 900           |
| Atendimento e Ticketing                    | 22.0%                 | 800           |
```

## Como Contribuir

1. Fork o repositório.
2. Crie uma nova branch (`git checkout -b feature-nome`).
3. Faça suas alterações e commit (`git commit -am 'Adiciona nova feature'`).
4. Push para a branch (`git push origin feature-nome`).
5. Crie um novo Pull Request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
