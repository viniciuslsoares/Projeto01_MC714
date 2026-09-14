# Trabalho 1: Balanceador de Carga para um Sistema Distribuído

**Disciplina:** MC714 — Sistemas Distribuídos (2º Semestre de 2026)  
**Instituição:** Instituto de Computação — Universidade Estadual de Campinas (UNICAMP)  
**Docente:** Prof. Carlos Alberto Astudillo Trujillo  
**Monitoria (PED):** Diogo Maciel da Cunha  
**Data Limite de Entrega:** 22 de setembro de 2026  

---

## 📌 1. Visão Geral e Requisitos do Trabalho

| Item | Especificação |
| :--- | :--- |
| **Objetivo** | Implementar um simulador de eventos discretos, desenvolver a modelagem teórica analítica (Teoria de Filas) e avaliar experimentalmente o desempenho de um balanceador de carga para 3 servidores homogêneos e extensões. |
| **Formato de Trabalho** | **Em trio (3 integrantes)** — *Requisito:* a inclusão de **pelo menos um ponto extra** é obrigatória para grupos em trio. |
| **Linguagem / Plataforma** | **Python 3.10+** (Simulador de Eventos Discretos próprio de alto desempenho com `heapq`, `numpy`, `scipy`, `pandas` e `matplotlib`). |
| **Entregáveis** | `relatorio_projeto1_<nome_de_um_integrante>.pdf` (máx. 4 páginas, padrão IEEE coluna dupla) <br> `codigo_projeto1_<nome_de_um_integrante>.zip` (executável em Windows/Linux). |

---

## 🏗️ 2. Modelo do Sistema e Políticas de Balanceamento

```
                         ┌─────────────┐   μ
               ┌─────────┤ Servidor 1  ├───►
               │         └─────────────┘
Poisson(λ)     ▼
──────────►[Dispatcher]──►┌─────────────┐   μ
 (Chegadas)    ▲         │ Servidor 2  ├───►
               │         └─────────────┘
               │
               └─────────►┌─────────────┐   μ
                         │ Servidor 3  ├───►
                         └─────────────┘
```

### 2.1. Parâmetros do Sistema Base

- **Servidores:** 3 nós homogêneos, single-thread, fila FCFS (*First Come, First Served*) ilimitada.
- **Chegadas:** Processo de Poisson com taxa global $\lambda$ (intervalos entre chegadas exponenciais com média $1/\lambda$).
- **Tempo de Serviço:** Exponencial com taxa $\mu = 1.0$ (tempo médio de serviço $E[S] = 1/\mu = 1.0$ u.t.).
- **Balanceador (*Dispatcher*):** Instantâneo (overhead de decisão nulo).

### 2.2. Políticas de Despacho

1. **Escolha Aleatória (*Random*):** Roteia para o servidor $i \in \{1, 2, 3\}$ com probabilidade uniforme $1/3$, de forma estocasticamente independente para cada requisição.
2. **Round-Robin (*RR*):** Distribui as requisições em ordem cíclica determinística ($1 \to 2 \to 3 \to 1 \dots$).
3. **Fila Mais Curta (*Join the Shortest Queue - JSQ*):** Consulta o número de requisições em cada servidor (em fila + em atendimento) e encaminha para o de menor carga. Empates são resolvidos por sorteio uniforme.
4. **Proporcional (*Heterogêneo*):** Roteia com probabilidades proporcionais à capacidade de cada nó ($p_i = \mu_i / \sum \mu_j$).

---

## 📐 3. Modelagem Analítica (Checklist Teórico)

A modelagem analítica formal desenvolve a teoria para a **Política Aleatória** como referência analítica de comparação:

- [x] **(a) Teorema da Decomposição (Splitting de Poisson):**
  - Provar que sob a política aleatória ($p_i = 1/3$), o fluxo de chegadas a cada servidor $i$ é um processo de Poisson independente de taxa $\lambda_i = \lambda / 3$.
- [x] **(b) Dedução da Fila M/M/1 por Servidor:**
  - Cada servidor opera como uma fila $M/M/1$ com taxa de chegada $\lambda_i = \lambda/3$ e taxa de serviço $\mu = 1.0$.
  - Intensidade de tráfego: $\rho = \frac{\lambda_i}{\mu} = \frac{\lambda}{3}$.
  - Condição de estabilidade: $\rho < 1 \iff \lambda < 3.0$ req/u.t.
  - Utilização: $U_i = \rho = \frac{\lambda}{3}$.
  - Número médio de clientes no servidor: $E[N_i] = \frac{\rho}{1 - \rho} = \frac{\lambda}{3 - \lambda}$.
  - Número total no sistema: $E[N] = 3 \cdot E[N_i] = \frac{3\lambda}{3 - \lambda}$.
  - Tempo médio de resposta: $E[R] = \frac{1}{\mu - \lambda/3} = \frac{3}{3 - \lambda}$.
  - Tempo médio em fila: $E[T_Q] = E[R] - E[S] = \frac{\lambda/3}{1 - \lambda/3}$.
  - Vazão do sistema: $X = \lambda$.
- [x] **(c) Princípio da Conservação de Trabalho:**
  - Demonstrar por que $X = \lambda$ e $U_i = \lambda/3$ são idênticos em regime estacionário para as 3 políticas (sistema conservativo sem perdas e com servidores idênticos).
- [x] **(d) Comparação e Ordenação de Desempenho:**
  - Demonstrar: $E[R]_{\text{JSQ}} \le E[R]_{\text{RR}} \le E[R]_{\text{Aleatória}} \approx \frac{1}{\mu - \lambda/3}$.
  - Justificativa física: redução da variância dos tempos de espera, uso de informação de estado em tempo real (JSQ) vs balanceamento determinístico sem memória (RR) vs aleatoriedade pura.
  - Cálculo de ganho percentual: $\text{Ganho}_{\text{pol}}(\lambda) = \frac{E[R]_{\text{Aleat}} - E[R]_{\text{pol}}}{E[R]_{\text{Aleat}}} \times 100\%$.
- [x] **(e) Validação da Lei de Little:**
  - Verificação empírica: $E[N] = X \cdot E[R]$ com desvios inferiores a $0.2\%$.
- [x] **(f) Análise do Regime Instável ($\lambda = 3.3$):**
  - Para $\lambda > 3\mu = 3.0$, o sistema não atinge equilíbrio estacionário.
  - Modelo fluido: $N(t) \approx N(0) + (\lambda - 3\mu)t = 0.3 \cdot t$.

---

## 🧪 4. Planejamento Experimental

### 4.1. Parâmetros de Simulação

- **Cargas testadas ($\lambda$):** $\lambda \in \{0.6, 1.2, 1.8, 2.4, 2.7\}$ (15 configurações: 5 taxas $\times$ 3 políticas).
- **Duração da Simulação:** $5000$ unidades de tempo por réplica.
- **Período de Aquecimento (*Warm-up*):** Descarte obrigatório dos primeiros $500$ u.t.
- **Número de Réplicas:** 10 execuções com sementes independentes por configuração.
- **Estatística:** Média amostral e Intervalo de Confiança de 95% via distribuição $t$-Student ($n=10$, $t_{9, 0.025} \approx 2.262$).

### 4.2. Métricas Coletadas

1. **Vazão ($X$):** $\frac{\text{Requisições concluídas pós warm-up}}{\Delta T}$.
2. **Tempo Médio de Resposta ($E[R]$):** Média de $(t_{\text{saída}} - t_{\text{chegada}})$ das requisições geradas pós warm-up.
3. **Número Médio no Sistema ($E[N]$):** Integral temporal contínua da ocupação do sistema:
   $$E[N] = \frac{1}{T_{\text{fim}} - T_{\text{warmup}}} \int_{T_{\text{warmup}}}^{T_{\text{fim}}} N(t)\,dt$$
4. **Utilização ($U_i$):** Fração do tempo pós warm-up em que o servidor $i$ esteve ocupado.

---

## 🌟 5. Ponto Extra (Obrigatório para Trio)

### Opção A: Buffer Finito ($K \in \{5, 10, 20\}$)

- **Mecânica:** Fila com capacidade máxima $K$ (1 em atendimento + $K-1$ em fila). Requisições excedentes são descartadas (*dropped*).
- **Modelagem $M/M/1/K$:**
  $$p_k = \frac{(1-\rho)\rho^k}{1 - \rho^{K+1}}, \quad P_{\text{perda}} = p_K, \quad X_{\text{efetiva}} = \lambda(1 - P_{\text{perda}})$$
- **Análise:** Avaliação das taxas de descarte e do tempo médio de resposta $E[R]$ truncado.

### Opção B: Servidores Heterogêneos

- **Mecânica:** Servidores com taxas $\mu_1 = 1.5, \mu_2 = 1.0, \mu_3 = 0.5$ req/u.t.
- **Análise Teórica:** Com roteamento uniforme ($1/3$), o nó 3 satura para $\lambda \ge 1.5$. Com pesos ótimos $p_i = \mu_i / \sum \mu_j = [0.5, 0.333, 0.167]$, todos os nós operam sob a mesma intensidade de tráfego $\rho = \lambda / 3.0$.

---

## 📁 6. Estrutura do Repositório

```text
Projeto01_MC714/
├── README.md                  # Este documento (visão geral, teoria e instruções)
├── requirements.txt           # Dependências Python (numpy, scipy, matplotlib, pandas)
├── test_sanity.py             # Script de validação e teste de sanidade imediato
│
├── src/                       # Núcleo do Simulador de Eventos Discretos
│   ├── __init__.py
│   ├── models.py              # Entidades Request, Event e SimulationMetrics
│   ├── server.py              # Classe Server (FCFS, buffers K, taxas mu_i e tempos ocupados)
│   ├── load_balancer.py       # LoadBalancer e políticas (Random, Round-Robin, JSQ, Proporcional)
│   ├── simulator.py           # Motor DES (heapq, Poisson, warm-up e integral de E[N])
│   └── metrics.py             # Agregação estatística, IC 95% (t-Student) e Lei de Little
│
├── analysis/                  # Teoria e análise dos dados medidos
│   ├── __init__.py
│   ├── analytical_model.py    # MODELAGEM ANALÍTICA: fórmulas fechadas (M/M/1, E_3/M/1, M/M/1/K, heterogêneo)
│   └── analise.py             # ANÁLISE EMPÍRICA: relê os CSVs e gera tabelas, figuras e verificações
│
├── experiments/               # Automação de Experimentos
│   ├── __init__.py
│   ├── grid.py                # Executor da grade (compartilhado) + escrita atômica de CSV
│   ├── run_experiments.py     # Execução das 15 configurações base (10 sementes cada)
│   ├── run_instability.py     # Experimento de sobrecarga (lambda = 3.3)
│   └── run_all.py             # Porta da frente: gera todos os resultados em ordem
│
├── tests/                     # Bateria pytest dos dois seams do pipeline
│   ├── test_grid.py           # Execução da grade
│   └── test_analise.py        # Agregação das réplicas
│
├── results/                   # CSVs brutos e agregados, verificações
│   ├── figures/               # Figuras em PDF vetorial para o \includegraphics
│   └── tables/                # Fragmentos LaTeX para o \input
│
└── report/                    # Artigo em LaTeX (template IEEEtran de 4 páginas)
    ├── main.tex
    └── figures/
```

---

## 👥 7. Divisão de Tarefas no Trio

| Integrante | Foco Principal | Atribuições |
| :--- | :--- | :--- |
| **Integrante 1** | **Núcleo de Simulação & Políticas** | • Implementação do motor DES com `heapq` e entidades de dados.<br>• Implementação dos nós servidores e das 4 políticas de despacho.<br>• Lógica de descarte de warm-up e integração temporal contínua para $E[N]$. |
| **Integrante 2** | **Modelagem Analítica & Ponto Extra** | • Dedução formal e rigorosa dos itens *(a)* a *(f)*.<br>• Formulação matemática do ponto extra ($M/M/1/K$ ou Heterogêneo).<br>• Script de cálculo analítico e validação da Lei de Little. |
| **Integrante 3** | **Pipeline Experimental & Relatório IEEE** | • Automação das 15 configurações + 10 sementes + regime instável $\lambda=3.3$.<br>• Tratamento estatístico ($\text{IC}_{95\%}$) e geração de gráficos/tabelas.<br>• Redação e diagramação do artigo em LaTeX (máximo de 4 páginas IEEE). |

---

## 🛠️ 8. Instalação e Execução

Requer **Python 3.10 ou superior**. Todos os comandos são executados **na raiz do
repositório** e não dependem de nenhuma variável de ambiente.

### 1. Configurar Ambiente Virtual

```bash
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Gerar Todos os Resultados (~20 s)

```bash
python -m experiments.run_all
```

Produz em `results/`:

| Arquivo | Conteúdo |
| :--- | :--- |
| `metrics.csv` | 150 linhas — uma por (política, $\lambda$, réplica). Dado bruto, sem agregação. |
| `instability_summary.csv` | 9 linhas do regime instável $\lambda = 3.3$: vazão, $N$ final e utilizações. |
| `instability_trajectories.csv` | Trajetória $N(t)$ evento a evento (não versionada: regenerável). |
| `aggregated_metrics.csv` | Uma linha por configuração: média, semi-IC 95%, Lei de Little, teoria e ganho. |
| `verifications.txt` | As seis verificações (as mesmas impressas no terminal): réplicas, `n_arrived`, Little, $X$, $U_i$ e ordenação. |
| `tables/*.tex` | Tabela teórica (item *b*) e tabela medida (item *d*), prontas para `\input{}`. |
| `figures/*.pdf` | $E[R] \times \lambda$, painel de instabilidade e, quando houver dados, o painel do ponto extra. |
| `*.meta.json` | Proveniência: commit, data, versões e parâmetros de cada execução. |

### 4. Executar as Etapas Isoladamente

```bash
python -m experiments.run_experiments                 # grid do enunciado
python -m experiments.run_experiments --quick         # 2 réplicas, para depurar
python -m experiments.run_instability                 # regime instável lambda = 3.3
python -m analysis.analise                            # tabelas, figuras e verificações
```

A análise **nunca simula**: ela só relê os CSVs. Corrigir uma figura ou trocar o
nível de confiança custa reler um arquivo, não rodar a grade de novo. Ela aceita
vários CSVs de uma vez e agrupa por (política, $\lambda$, $\mu$, duração), então
concatenar rodadas funciona sem opção nova:

```bash
# Ponto extra: o painel uniforme x proporcional sai só de apontar para o CSV heterogêneo
python -m analysis.analise --input results/metrics.csv results/metrics_hetero.csv

# Convergência de horizonte finito: duração diferente vira outro grupo de linhas
python -m analysis.analise --input results/metrics.csv results/metrics_20k.csv
```

O grid padrão é exatamente o do enunciado; a linha de comando serve apenas para
sobrepor (`--lambdas`, `--policies`, `--replicas`, `--duration`, `--warmup`,
`--mu`, `--weights`, `--output`). Exemplos:

```bash
# Horizonte mais longo (converge o desvio residual em lambda = 2.7)
python -m experiments.run_experiments --duration 20000 --output results/metrics_20k.csv

# Ponto extra heterogêneo: mu = [1.5, 1.0, 0.5]
python -m experiments.run_experiments --policies proportional --mu 1.5 1.0 0.5
```

### 5. Executar os Testes

```bash
python test_sanity.py     # validação do motor contra a teoria
python -m pytest          # bateria dos dois seams: executor de grade e agregação
```

---

## 📝 9. Estrutura do Relatório IEEE (Máximo 4 Páginas)

1. **Resumo / Abstract:** Motivação, políticas comparadas, extensão implementada e principais resultados.
2. **Introdução e Arquitetura do Sistema:** Arquitetura do balanceador, modelagem FCFS dos servidores e processo de Poisson.
3. **Modelagem Analítica:** Deduções formais dos itens *(a)* a *(f)* + modelagem do Ponto Extra.
4. **Metodologia de Simulação:** Arquitetura orientada a eventos, sementes, descarte de warm-up e integral temporal de $N(t)$.
5. **Resultados e Discussão:**
   - Tabelas comparativas (Analítico vs Simulado com $\text{IC}_{95\%}$).
   - Curvas de $E[R] \times \lambda$ com barras de erro de 95%.
   - Ganhos percentuais relativos e validação da Lei de Little.
   - Análise do regime instável ($\lambda = 3.3$) vs aproximação fluida.
   - Resultados e análise do Ponto Extra.
6. **Divisão de Trabalho e Conclusão:** Tabela de contribuições individuais e conclusões do estudo.

---

## ⚠️ 10. Critérios de Avaliação

| Componente | Peso | Critério |
| :--- | :---: | :--- |
| **Relatório e Modelagem Analítica (a-f)** | **35%** | Rigor matemático, clareza nas deduções e inclusão do ponto extra. |
| **Implementação da Simulação** | **20%** | Correção do gerador Poisson, warm-up e coleta de métricas. |
| **Implementação do Balanceador (3 políticas)** | **20%** | Correção de empates no JSQ e independência no Random. |
| **Comparação Quantitativa Analítico × Simulação** | **20%** | Barras de erro de 95%, gráficos claros e tabelas detalhadas. |
| **Qualidade do Código e Documentação** | **5%** | Código modular, limpo, tipado, comentado e reprodutível. |
