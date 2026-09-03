# Reprodutibilidade das análises da dissertação

## 1. Escopo

Este documento descreve a reprodução das análises adicionais da dissertação
**“Avaliação de Estratégias de Seleção de Clientes em Aprendizado Federado sob
a Perspectiva de Eficiência Energética”**.

A base experimental é exclusivamente a empregada no artigo aceito na LANC
2026:

- dataset: FashionMNIST;
- 100 clientes;
- particionamento *pathological non-IID* fixo entre as repetições;
- algoritmos: FedAvg, FedProx e SCAFFOLD;
- *join ratios*: 0,10; 0,25; 0,50; 0,75; e 1,00;
- 50 rodadas nominais;
- 5 épocas locais;
- 5 repetições por configuração;
- 15 configurações e 75 execuções primárias.

Os experimentos posteriores com TON-IoT e UNSW-NB15 não fazem parte desta
análise. Nenhum novo experimento de aprendizado federado é executado pelos
scripts descritos aqui.

## 2. Fontes dos dados

Os logs originais estão fora do repositório, no notebook utilizado para os
experimentos:

```text
~/research/experiments/fashionmnist_gpu
```

O repositório de análise está em:

```text
~/research/git/fl-join-ratio-paper
```

O arquivo compactado auditado dos logs apresentou o SHA-256:

```text
68d7036342b1b65764282c5eb58ec7efb62dc13daebd586154221d4772b1076d
```

Foram encontrados 76 diretórios de execução. Destes, 75 pertencem ao plano
experimental do artigo e um é uma execução adicional identificada como:

```text
SCAFFOLD/jr075/rep6_rerun_outlier
```

Essa execução adicional é preservada no registro de auditoria, mas não é
incluída nos resultados principais.

## 3. Ambiente de execução das análises

Os comandos abaixo pressupõem:

- Ubuntu no WSL;
- repositório como diretório atual;
- ambiente Python disponível em `~/fl_env`;
- dependências científicas já instaladas no ambiente.

Para entrar no repositório:

```bash
cd ~/research/git/fl-join-ratio-paper
```

Antes de executar um script, sua sintaxe pode ser verificada com:

```bash
~/fl_env/bin/python -m py_compile scripts/NOME_DO_SCRIPT.py
```

## 4. Ordem completa de reprodução

### 4.1 Extração, auditoria e validação

```bash
~/fl_env/bin/python scripts/build_dissertation_datasets.py \
  --logs-root ~/research/experiments/fashionmnist_gpu \
  --article-raw data/fashionmnist_raw_results.csv \
  --output-dir analysis_outputs
```

Principais produtos:

- `analysis_outputs/dissertation_runs.csv`: uma linha por execução primária;
- `analysis_outputs/dissertation_rounds.csv`: uma linha por avaliação e rodada;
- `analysis_outputs/validation_against_article.csv`: comparação com a base do artigo;
- `analysis_outputs/excluded_additional_runs.csv`: execução adicional excluída;
- `analysis_outputs/parsing_errors.csv`: erros de leitura, se houver;
- `analysis_outputs/audit_report.txt`: síntese da auditoria.

Resultado validado: 75 execuções primárias, 3.825 registros de rodada e nenhuma
diferença na melhor accuracy ou no tempo total em relação à base do artigo.

### 4.2 Efeito do join ratio dentro de cada algoritmo

```bash
~/fl_env/bin/python scripts/analyze_join_ratio_effect.py \
  --runs analysis_outputs/dissertation_runs.csv \
  --output-dir analysis_outputs/join_ratio_effect
```

Essa etapa calcula estatísticas descritivas, Kruskal-Wallis dentro de cada
algoritmo, comparações Mann-Whitney exatas com correção de Holm e tamanho de
efeito de Cliff. As comparações são tratadas como grupos independentes, pois
não se assume pareamento formal das repetições entre *join ratios*.

### 4.3 Retornos marginais

```bash
~/fl_env/bin/python scripts/analyze_marginal_returns.py \
  --runs analysis_outputs/dissertation_runs.csv \
  --output-dir analysis_outputs/marginal_returns
```

Essa etapa calcula, entre níveis consecutivos de participação:

- ganho de accuracy em pontos percentuais;
- acréscimo de tempo e energia;
- ganho marginal por 1.000 segundos;
- ganho marginal por kJ;
- candidatos exploratórios a joelhos das curvas.

### 4.4 Trade-off e fronteiras de Pareto

```bash
~/fl_env/bin/python scripts/analyze_pareto_tradeoff.py \
  --runs analysis_outputs/dissertation_runs.csv \
  --output-dir analysis_outputs/pareto_tradeoff
```

Essa etapa maximiza a accuracy e minimiza tempo e energia. As recomendações
balanceadas usam normalização e pesos iguais, devendo ser interpretadas como
decisões sob uma regra explicitamente escolhida, e não como ótimos universais.

### 4.5 Convergência ao longo das rodadas

```bash
~/fl_env/bin/python scripts/analyze_convergence.py \
  --rounds analysis_outputs/dissertation_rounds.csv \
  --output-dir analysis_outputs/convergence
```

Essa etapa calcula curvas médias, AULC normalizada, rodada do melhor resultado,
primeira rodada e rodada sustentada para limiares absolutos e relativos.

O limiar absoluto comum é accuracy 0,65. Os limiares relativos correspondem a
90%, 95% e 99% da melhor accuracy registrada em cada execução.

### 4.6 Caracterização da carga do sistema

```bash
~/fl_env/bin/python scripts/analyze_system_workload.py \
  --runs analysis_outputs/dissertation_runs.csv \
  --output-dir analysis_outputs/system_workload
```

Essa etapa combina os tempos e energias observados com contagens analíticas da
carga computacional e de comunicação. Ela não executa treinamento adicional.

### 4.7 Estabilidade das cinco repetições

```bash
~/fl_env/bin/python scripts/analyze_stability.py \
  --runs analysis_outputs/dissertation_runs.csv \
  --output-dir analysis_outputs/stability
```

Essa etapa calcula média, mediana, desvio-padrão, quartis, IQR, coeficiente de
variação e flags de possíveis outliers pela regra de Tukey. Nenhuma execução é
removida automaticamente.

### 4.8 Tabelas consolidadas

```bash
~/fl_env/bin/python scripts/build_dissertation_tables.py \
  --analysis-root analysis_outputs \
  --output-dir analysis_outputs/dissertation_tables
```

Cada tabela é gerada em CSV, para auditoria, e em LaTeX, para inclusão no texto.

## 5. Decisões metodológicas

### 5.1 Medidas centrais

- accuracy principal: média e desvio-padrão das cinco repetições;
- tempo de execução: mediana;
- energia estimada: mediana;
- estabilidade: média, mediana, desvio-padrão, IQR e coeficiente de variação.

A coluna `best_accuracy_reported` é usada nas análises da melhor accuracy por
preservar a precisão validada contra o artigo. A análise de convergência usa as
accuracies registradas por rodada, que possuem a precisão disponível nos logs.

### 5.2 Energia

A energia é uma estimativa baseada nos logs de utilização de CPU e GPU, e não
uma medição direta na tomada. O modelo empregado na análise original é:

```text
P = 15 × (CPU% / 800) + 10 × (GPU% / 100) + 4 + 2 watts
```

Os resultados consolidados são apresentados em quilojoules. A coluna de origem
`estimated_energy_j` é convertida uma única vez de joules para quilojoules.

### 5.3 Rodadas e avaliações

Os logs contêm avaliações numeradas de 0 a 50:

- rodada 0: avaliação inicial do modelo não treinado;
- rodadas 1 a 50: avaliações após 1 a 50 atualizações globais concluídas.

Na implementação inspecionada, o laço usa `range(global_rounds + 1)` e executa
treinamento depois de cada avaliação. Assim, o custo total medido inclui 51
ciclos de treinamento, embora existam 50 rodadas globais avaliadas após o
estado inicial. Essa particularidade deve ser declarada na dissertação.

### 5.4 Comunicação

O experimento é uma simulação em um único processo e copia parâmetros em
memória. Portanto, não houve medição de tráfego de rede.

Os volumes apresentados são estimativas de payload lógico em FP32:

- modelo: 582.026 parâmetros;
- vetor FP32: 2.328.104 bytes;
- FedAvg/FedProx: um vetor de modelo em cada direção no cenário selecionado;
- SCAFFOLD: dois vetores de tamanho equivalente em cada direção, considerando
  modelo e variáveis de controle.

O cenário distribuído com comunicação apenas com clientes selecionados é
hipotético e deve ser identificado dessa forma.

### 5.5 Outliers e multiplicidade

As cercas de Tukey são usadas apenas como diagnóstico. Com cinco repetições,
quartis e flags são sensíveis a pequenas diferenças. Nenhuma observação é
excluída sem justificativa externa verificável.

Nos testes par a par, a correção de Holm controla comparações múltiplas. A falta
de significância pós-correção não deve ser interpretada como prova de ausência
de efeito, especialmente com cinco observações por configuração. Os testes
globais, tamanhos de efeito, tendências e medidas de custo devem ser discutidos
em conjunto.

## 6. Organização recomendada na dissertação

No corpo principal:

1. visão geral de qualidade, tempo e energia;
2. efeito do *join ratio* dentro dos algoritmos;
3. convergência e carga computacional;
4. retornos marginais e joelhos;
5. Pareto e recomendações práticas;
6. estabilidade, ameaças à validade e limitações.

Resultados extensos, como todas as comparações par a par e todos os flags de
outliers, devem permanecer no repositório ou em apêndice. O texto principal
deve priorizar interpretações computacionais e operacionais.

## 7. Verificação do estado do repositório

Para conferir alterações pendentes:

```bash
git status --short
```

Para conferir o histórico das análises:

```bash
git log --oneline
```

Arquivos preexistentes em `figures/` e PDFs avulsos em `notebooks/` não fazem
parte automaticamente desta sequência e não devem ser adicionados sem revisão.
