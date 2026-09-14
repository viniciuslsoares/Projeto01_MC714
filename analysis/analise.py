"""
Análise empírica: relê os CSVs de réplicas e produz tabelas, figuras e verificações.

NOMES PARECIDOS, papéis opostos: analytical_model.py é a teoria (fórmula fechada);
este arquivo é o medido. Ele importa o outro porque o enunciado pede a comparação.

Este script NUNCA simula. Corrigir uma figura ou trocar o nível de confiança custa
reler um arquivo.

A entrada é uma LISTA de CSVs, agrupada por (política, lambda, mu, duração). Assim,
mu diferente faz sair o painel do ponto extra e duração diferente faz sair a tabela
de convergência, sem caso especial.

Uso:
    python -m analysis.analise
    python -m analysis.analise --input results/metrics.csv results/metrics_20k.csv
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # sem isto, gerar figura em terminal sem display falha
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.analytical_model import AnalyticalModel
from experiments.grid import setup, write_csv
from src.load_balancer import LoadBalancerPolicy
from src.metrics import aggregate_replications
from src.models import SimulationMetrics

Row = Dict[str, Any]              # uma linha do CSV do grid, já com números
Mu = Tuple[float, ...]
Key = Tuple[str, float, Mu, float]  # (política, lambda, mu, duração)
Qualifier = Callable[["ConfigResult"], str]

NUM_SERVERS = 3
LITTLE_TOLERANCE_PCT = 0.2   # orçamento do README para E[N] vs X*E[R]
THROUGHPUT_TOLERANCE_PCT = 1.0
MAX_TRAJECTORY_POINTS = 2000  # N(t) é escada; mais que isto só engorda o PDF
COLUMN_WIDTH_IN = 3.5         # largura de uma coluna do IEEE
CAPACITY = 3.0                # 3 servidores com mu = 1: o painel de instabilidade é só desse caso


@dataclass(frozen=True)
class PolicyStyle:
    """Como uma política se chama em português e como ela é desenhada."""
    label: str
    marker: str
    linestyle: str


# A ordem de inserção é a ordem em tabelas, legendas e na verificação de ordenação.
POLICIES = {
    LoadBalancerPolicy.RANDOM: PolicyStyle("aleatória", "o", "-"),
    LoadBalancerPolicy.ROUND_ROBIN: PolicyStyle("round-robin", "s", "--"),
    LoadBalancerPolicy.SHORTEST_QUEUE: PolicyStyle("fila mais curta", "^", ":"),
    LoadBalancerPolicy.PROPORTIONAL: PolicyStyle("proporcional", "D", "-."),
}

AGGREGATED_COLUMNS = [
    "policy", "lambda", "mu0", "mu1", "mu2", "duration", "replicas",
    "X_mean", "X_ci95", "E_R_mean", "E_R_ci95", "E_TQ_mean", "E_TQ_ci95",
    "E_N_mean", "E_N_ci95",
    "U0_mean", "U0_ci95", "U1_mean", "U1_ci95", "U2_mean", "U2_ci95",
    "little_product", "little_error_pct",
    "E_R_teorico_mm1", "E_R_teorico_e3m1", "desvio_teoria_pct",
    "ganho_vs_random_pct",
]


@dataclass
class ConfigResult:
    """Uma configuração agregada: média e semi-IC de 95% de cada métrica."""
    policy: str
    lambda_rate: float
    mu: Mu
    duration: float
    replicas: int

    X_mean: float
    X_ci95: float
    E_R_mean: float
    E_R_ci95: float
    E_TQ_mean: float
    E_TQ_ci95: float
    E_N_mean: float
    E_N_ci95: float
    U_means: List[float]
    U_ci95: List[float]

    # E[N] vem da integral no tempo e X das partidas: caminhos independentes.
    little_product: float
    little_error_pct: float

    E_R_teorico_mm1: Optional[float]
    E_R_teorico_e3m1: Optional[float]
    desvio_teoria_pct: Optional[float]
    ganho_vs_random_pct: Optional[float] = None

    @property
    def is_homogeneous(self) -> bool:
        return len(set(self.mu)) == 1


@dataclass
class Check:
    """Uma verificação: aviso legível, nunca exceção."""
    name: str
    ok: bool
    lines: List[str]


@dataclass
class AnalysisResults:
    configs: List[ConfigResult]
    checks: List[Check]


# --------------------------------------------------------------------------
# O seam: função pura, tudo o que é disco ou figura fica fora
# --------------------------------------------------------------------------

def aggregate(rows: List[Row]) -> AnalysisResults:
    """
    Agrega as linhas de réplica em uma linha por configuração.

    A teoria entra AQUI porque o item (d) compara medido com teórico: a comparação
    é resultado, não formatação.
    """
    if not rows:
        raise ValueError("Nenhuma linha para agregar.")

    groups: Dict[Key, List[Row]] = {}
    for row in rows:
        groups.setdefault(_key(row), []).append(row)

    replicas_iguais = _check_replica_counts(groups)

    configs = [_aggregate_one(key, group) for key, group in groups.items()]
    configs.sort(key=lambda c: (c.duration, c.mu, _policy_rank(c.policy), c.lambda_rate))
    _fill_gains(configs)

    checks = [replicas_iguais] + _run_checks(configs, rows)
    return AnalysisResults(configs=configs, checks=checks)


def _key(row: Row) -> Key:
    return (row["policy"], float(row["lambda"]), _mu_of(row), float(row["duration"]))


def _mu_of(row: Row) -> Mu:
    return tuple(float(row[f"mu{i}"]) for i in range(NUM_SERVERS))


def _policy_rank(policy: str) -> int:
    return list(POLICIES).index(policy) if policy in POLICIES else len(POLICIES)


def label_of(policy: str) -> str:
    """Rótulo em português; política desconhecida aparece com o nome cru."""
    return POLICIES[policy].label if policy in POLICIES else policy


def _check_replica_counts(groups: Dict[Key, List[Row]]) -> Check:
    """
    A única falha dura. Um CSV com uma réplica faltando parece completo e o IC
    sairia de 9 réplicas em silêncio — pior que uma exceção.

    A igualdade é exigida DENTRO de cada (mu, duração), que é o que uma rodada
    produz. Rodadas diferentes podem ter número de réplicas diferente de propósito
    (a de 200 000 u.t. é caríssima).
    """
    by_run: Dict[Tuple[Mu, float], Dict[Key, int]] = {}
    for key, group in groups.items():
        _, _, mu, duration = key
        by_run.setdefault((mu, duration), {})[key] = len(group)

    for (mu, duration), counts in by_run.items():
        if len(set(counts.values())) > 1:
            detalhe = ", ".join(
                f"{policy} λ={lam}: {n}" for (policy, lam, _, _), n in sorted(counts.items())
            )
            raise ValueError(
                f"Número de réplicas desigual em mu={list(mu)}, duração {duration}: {detalhe}. "
                "Rode a grade inteira de novo em vez de agregar um arquivo incompleto."
            )

    lines = [
        f"mu={list(mu)}, T={duration:.0f}: {len(counts)} config., "
        f"{next(iter(counts.values()))} réplicas cada"
        for (mu, duration), counts in sorted(by_run.items())
    ]
    return Check("réplicas por configuração todas iguais (falha dura)", True, lines)


def _aggregate_one(key: Key, group: List[Row]) -> ConfigResult:
    """Reusa a agregação já validada de src.metrics, via adaptação das linhas."""
    policy, lambda_rate, mu, duration = key
    agg = aggregate_replications([_row_to_metrics(row) for row in group])

    teorico_mm1, teorico_e3m1, referencia = _theory(lambda_rate, mu, policy)
    desvio = None
    if referencia is not None:
        desvio = (agg.response_time_mean - referencia) / referencia * 100.0

    return ConfigResult(
        policy=policy,
        lambda_rate=lambda_rate,
        mu=mu,
        duration=duration,
        replicas=agg.num_replicas,
        X_mean=agg.throughput_mean,
        X_ci95=agg.throughput_ci95,
        E_R_mean=agg.response_time_mean,
        E_R_ci95=agg.response_time_ci95,
        E_TQ_mean=agg.queue_time_mean,
        E_TQ_ci95=agg.queue_time_ci95,
        E_N_mean=agg.avg_n_mean,
        E_N_ci95=agg.avg_n_ci95,
        U_means=agg.utilization_means,
        U_ci95=agg.utilization_ci95,
        little_product=agg.little_law_product,
        little_error_pct=agg.little_law_error_percent,
        E_R_teorico_mm1=teorico_mm1,
        E_R_teorico_e3m1=teorico_e3m1,
        desvio_teoria_pct=desvio,
    )


def _row_to_metrics(row: Row) -> SimulationMetrics:
    """
    Adapta a linha do CSV à estrutura que a agregação espera. Dois campos não
    existem no CSV e recebem valor de "não registrado": a agregação não os lê.
    """
    return SimulationMetrics(
        policy_name=row["policy"],
        lambda_rate=float(row["lambda"]),
        seed=int(row["seed"]),
        duration=float(row["duration"]),
        warmup_time=float(row["warmup"]),
        throughput=float(row["X"]),
        avg_response_time=float(row["E_R"]),
        avg_queue_time=float(row["E_TQ"]),
        avg_requests_in_system=float(row["E_N"]),
        server_utilizations=[float(row[f"U{i}"]) for i in range(NUM_SERVERS)],
        server_request_counts=[],
        total_requests_arrived=int(row["n_arrived"]),
        requests_after_warmup=int(row["n_after_warmup"]),
        completed_requests=int(row["n_completed"]),
        dropped_requests=-1,
        drop_probability=float(row["drop_prob"]),
    )


def _theory(
    lambda_rate: float, mu: Mu, policy: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    E[R] teórico da aleatória (M/M/1) e do round-robin (E_3/M/1), mais a referência
    que vale para ESTA política.

    Com mu heterogêneo as duas fórmulas deixam de valer, e a fila mais curta não tem
    forma fechada: referência ausente é melhor que referência errada.
    """
    if len(set(mu)) != 1:
        return None, None, None

    mm1 = AnalyticalModel.mm1_random_policy(lambda_rate, num_servers=len(mu), mu=mu[0])
    e3m1 = AnalyticalModel.round_robin_e3m1(lambda_rate, num_servers=len(mu), mu=mu[0])
    teorico_mm1 = mm1["E_R"] if mm1["is_stable"] else None
    teorico_e3m1 = e3m1["E_R"] if e3m1["is_stable"] else None

    referencia = {
        LoadBalancerPolicy.RANDOM: teorico_mm1,
        LoadBalancerPolicy.ROUND_ROBIN: teorico_e3m1,
    }.get(policy)
    return teorico_mm1, teorico_e3m1, referencia


def _fill_gains(configs: List[ConfigResult]) -> None:
    """
    Ganho sempre contra a aleatória: é a política ingênua e a única com forma fechada
    exata, o que a torna o baseline natural. Ausente quando ela não está nos dados.
    """
    baseline = {
        (c.mu, c.duration, c.lambda_rate): c.E_R_mean
        for c in configs if c.policy == LoadBalancerPolicy.RANDOM
    }
    for config in configs:
        referencia = baseline.get((config.mu, config.duration, config.lambda_rate))
        if referencia:  # E[R] medido é sempre > 0, então isto é só o teste de presença
            config.ganho_vs_random_pct = (referencia - config.E_R_mean) / referencia * 100.0


# --------------------------------------------------------------------------
# Verificações: cinco avisos, e nenhum interrompe
# --------------------------------------------------------------------------

def _run_checks(configs: List[ConfigResult], rows: List[Row]) -> List[Check]:
    qualify = _qualifier(configs)
    return [
        _check_arrivals(rows),
        _check_little(configs, qualify),
        _check_throughput(configs, qualify),
        _check_utilization(configs, qualify),
        _check_ordering(configs, qualify),
    ]


def _by_lambda(configs: List[ConfigResult]) -> Dict[Tuple[Mu, float, float], List[ConfigResult]]:
    """Configurações que só diferem na política — as comparáveis entre si."""
    grouped: Dict[Tuple[Mu, float, float], List[ConfigResult]] = {}
    for config in configs:
        grouped.setdefault((config.mu, config.duration, config.lambda_rate), []).append(config)
    return grouped


def _qualifier(configs: List[ConfigResult]) -> Callable[[ConfigResult], str]:
    """
    Sufixo que diz de qual rodada a linha fala. Vazio no caso comum de uma rodada só;
    com vários arquivos, sem ele duas linhas "λ=0,6" ficariam indistinguíveis.
    """
    rodadas = {(c.mu, c.duration) for c in configs}
    if len(rodadas) == 1:
        return lambda config: ""
    return lambda config: f" (mu={list(config.mu)}, T={config.duration:.0f})"


def _check_arrivals(rows: List[Row]) -> Check:
    """n_arrived idêntico é a assinatura observável do pareamento de sementes."""
    grouped: Dict[Any, set] = {}
    for row in rows:
        chave = (float(row["lambda"]), _mu_of(row), float(row["duration"]), int(row["replica"]))
        grouped.setdefault(chave, set()).add(int(row["n_arrived"]))

    divergentes = {k: v for k, v in grouped.items() if len(v) > 1}
    lines = [
        f"λ={lam} (mu={list(mu)}, T={duration:.0f}), réplica {replica}: "
        f"chegadas divergiram entre políticas {sorted(v)}"
        for (lam, mu, duration, replica), v in sorted(divergentes.items())
    ]
    if not divergentes:
        lines = [f"{len(grouped)} pares (λ, réplica) com n_arrived idêntico entre as políticas."]
    return Check("n_arrived idêntico entre as políticas", not divergentes, lines)


def _check_little(configs: List[ConfigResult], qualify: Qualifier) -> Check:
    """|E[N] - X*E[R]| / E[N]. Não se aplica sem regime estacionário."""
    piores = sorted(configs, key=lambda c: -c.little_error_pct)
    fora = [c for c in piores if c.little_error_pct >= LITTLE_TOLERANCE_PCT]

    lines = [
        f"{_config_label(c, qualify)}: erro {c.little_error_pct:.3f}% (limiar {LITTLE_TOLERANCE_PCT}%)"
        for c in fora
    ]
    if not fora:
        pior = piores[0]
        lines = [
            f"maior erro {pior.little_error_pct:.3f}% em {_config_label(pior, qualify)}, "
            f"limiar {LITTLE_TOLERANCE_PCT}%."
        ]
    return Check("Lei de Little (E[N] contra X*E[R])", not fora, lines)


def _check_throughput(configs: List[ConfigResult], qualify: Qualifier) -> Check:
    """
    X é MEDIDO: coincide entre políticas dentro do IC, e nunca é idêntico. O que
    é exatamente idêntico é n_arrived.
    """
    lines, ok = [], True
    for _, grupo in sorted(_by_lambda(configs).items()):
        if len(grupo) < 2:
            continue
        medias = [c.X_mean for c in grupo]
        sobrepoe = max(c.X_mean - c.X_ci95 for c in grupo) <= min(c.X_mean + c.X_ci95 for c in grupo)
        delta = (max(medias) - min(medias)) / min(medias) * 100.0
        if not sobrepoe or delta >= THROUGHPUT_TOLERANCE_PCT:
            ok = False
        lines.append(
            f"λ={grupo[0].lambda_rate}{qualify(grupo[0])}: "
            f"Δ={delta:.3f}% (limiar {THROUGHPUT_TOLERANCE_PCT}%), "
            f"ICs {'sobrepostos' if sobrepoe else 'DISJUNTOS'}"
        )
    if not lines:
        lines = ["só uma política nos dados: nada a comparar."]
    return Check("X coincide entre as políticas dentro do IC", ok, lines)


def _check_utilization(configs: List[ConfigResult], qualify: Qualifier) -> Check:
    """Item (c) é uma igualdade a demonstrar: U_i = rho = λ/(3 mu)."""
    lines, ok = [], True
    for config in configs:
        if not config.is_homogeneous:
            continue
        rho = config.lambda_rate / (NUM_SERVERS * config.mu[0])
        desvios = [
            abs(media - rho) - semi
            for media, semi in zip(config.U_means, config.U_ci95)
        ]
        dentro = max(desvios) <= 0.0
        if not dentro:
            ok = False
            lines.append(
                f"{_config_label(config, qualify)}: rho={rho:.4f}, U={_fmt_list(config.U_means)} "
                f"± {_fmt_list(config.U_ci95)} — fora do semi-IC"
            )
    if ok:
        homogeneas = sum(1 for c in configs if c.is_homogeneous)
        lines = [f"{homogeneas} configurações com U_i dentro do semi-IC em torno de rho."]
        if homogeneas == 0:
            lines = ["mu heterogêneo: rho por servidor depende da política, verificação não aplicável."]
    return Check("U_i contra rho = λ/(3 mu)", ok, lines)


def _check_ordering(configs: List[ConfigResult], qualify: Qualifier) -> Check:
    """Ordenação esperada, e se os ICs se sobrepõem não há diferença a afirmar."""
    esperada = [
        LoadBalancerPolicy.SHORTEST_QUEUE,
        LoadBalancerPolicy.ROUND_ROBIN,
        LoadBalancerPolicy.RANDOM,
    ]
    lines, ok = [], True

    for _, grupo in sorted(_by_lambda(configs).items()):
        por_politica = {c.policy: c for c in grupo}
        presentes = [p for p in esperada if p in por_politica]
        if len(presentes) < 2:
            continue
        for melhor, pior in zip(presentes, presentes[1:]):
            a, b = por_politica[melhor], por_politica[pior]
            onde = f"λ={a.lambda_rate}{qualify(a)}"
            if a.E_R_mean > b.E_R_mean:
                ok = False
                lines.append(
                    f"{onde}: {label_of(melhor)} ({a.E_R_mean:.4f}) pior que "
                    f"{label_of(pior)} ({b.E_R_mean:.4f}) — ordenação invertida"
                )
            elif a.E_R_mean + a.E_R_ci95 >= b.E_R_mean - b.E_R_ci95:
                lines.append(
                    f"{onde}: {label_of(melhor)} < {label_of(pior)} nas médias, "
                    "mas os ICs se sobrepõem — não afirmar diferença"
                )
    if not lines:
        lines = ["ordenação fila mais curta <= round-robin <= aleatória, sem sobreposição de ICs."]
    return Check("ordenação das políticas em E[R]", ok, lines)


def _config_label(config: ConfigResult, qualify: Qualifier) -> str:
    return f"{label_of(config.policy)} λ={config.lambda_rate}{qualify(config)}"


# --------------------------------------------------------------------------
# Leitura, escrita e formatação
# --------------------------------------------------------------------------

def read_rows(paths: List[Path]) -> List[Row]:
    """
    Lê e concatena CSVs do grid. Serve tanto para as réplicas quanto para as
    trajetórias: em ambos, só a política não é número.
    """
    rows: List[Row] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                rows.append({k: v if k == "policy" else float(v) for k, v in raw.items()})
    return rows


def base_group(configs: List[ConfigResult]) -> List[ConfigResult]:
    """
    O grid do enunciado dentro do que foi carregado: mu homogêneo e a duração mais
    frequente. Os outros grupos aparecem no CSV agregado e nas figuras próprias.
    """
    homogeneas = [c for c in configs if c.is_homogeneous]
    if not homogeneas:
        return []
    duracao = Counter(c.duration for c in homogeneas).most_common(1)[0][0]
    return [c for c in homogeneas if c.duration == duracao]


def fmt_pt(value: Optional[float], decimals: int = 4) -> str:
    """Número no padrão do texto em português, com vírgula decimal."""
    if value is None:
        return "---"
    return f"{value:.{decimals}f}".replace(".", ",")


def _fmt_list(values: List[float]) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"


def format_checks(checks: List[Check]) -> str:
    blocos = []
    for check in checks:
        status = "OK   " if check.ok else "AVISO"
        blocos.append(f"[{status}] {check.name}")
        blocos.extend(f"         {linha}" for linha in check.lines)
    return "\n".join(blocos)


def aggregated_rows(configs: List[ConfigResult]) -> List[Row]:
    rows = []
    for c in configs:
        row: Row = {
            "policy": c.policy, "lambda": c.lambda_rate, "duration": c.duration,
            "replicas": c.replicas,
            "X_mean": c.X_mean, "X_ci95": c.X_ci95,
            "E_R_mean": c.E_R_mean, "E_R_ci95": c.E_R_ci95,
            "E_TQ_mean": c.E_TQ_mean, "E_TQ_ci95": c.E_TQ_ci95,
            "E_N_mean": c.E_N_mean, "E_N_ci95": c.E_N_ci95,
            "little_product": c.little_product, "little_error_pct": c.little_error_pct,
            "E_R_teorico_mm1": c.E_R_teorico_mm1, "E_R_teorico_e3m1": c.E_R_teorico_e3m1,
            "desvio_teoria_pct": c.desvio_teoria_pct,
            "ganho_vs_random_pct": c.ganho_vs_random_pct,
        }
        for i in range(NUM_SERVERS):
            row[f"mu{i}"] = c.mu[i]
            row[f"U{i}_mean"] = c.U_means[i]
            row[f"U{i}_ci95"] = c.U_ci95[i]
        rows.append(row)
    return rows


def _tabular(comments: List[str], colspec: str, header: List[str], body: List[str]) -> str:
    """O esqueleto booktabs que as duas tabelas compartilham."""
    return "\n".join([
        "% Gerado por python -m analysis.analise. Não editar à mão.",
        *comments,
        f"\\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
        *header,
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "\\end{tabular}",
    ]) + "\n"


def theoretical_table(configs: List[ConfigResult]) -> str:
    """Tabela do item (b): uma linha por λ, gerada pelo modelo e não digitada."""
    mu = configs[0].mu[0]

    body = []
    for lam in sorted({c.lambda_rate for c in configs}):
        mm1 = AnalyticalModel.mm1_random_policy(lam, num_servers=NUM_SERVERS, mu=mu)
        e3m1 = AnalyticalModel.round_robin_e3m1(lam, num_servers=NUM_SERVERS, mu=mu)
        body.append(
            f"    {fmt_pt(lam, 1)} & {fmt_pt(mm1['rho'])} & {fmt_pt(mm1['U_i'])} & "
            f"{fmt_pt(mm1['E_R'])} & {fmt_pt(mm1['E_TQ'])} & {fmt_pt(mm1['E_N'])} & "
            f"{fmt_pt(e3m1['E_R'])} \\\\"
        )

    return _tabular(
        [f"% Item (b): metricas analiticas exatas, mu = {fmt_pt(mu, 1)} por servidor."],
        "lcccccc",
        [
            "    $\\lambda$ & $\\rho$ & $U_i$ & $E[R]$ & $E[T_Q]$ & $E[N]$ & $E[R]$ RR \\\\",
            "    & & & (M/M/1) & (M/M/1) & (M/M/1) & (E$_3$/M/1) \\\\",
        ],
        body,
    )


def measured_table(configs: List[ConfigResult]) -> str:
    """
    Tabela do item (d): medido ± semi-IC, Lei de Little, desvio e ganho.

    Com mais de uma duração carregada ganha a coluna T e passa a ser também a
    tabela de convergência de horizonte finito, sem virar outro artefato.
    """
    duracoes = sorted({c.duration for c in configs})
    convergencia = len(duracoes) > 1

    body = []
    for c in configs:
        horizonte = f"{c.duration:.0f} & " if convergencia else ""
        body.append(
            f"    {label_of(c.policy)} & {horizonte}{fmt_pt(c.lambda_rate, 1)} & "
            f"{fmt_pt(c.X_mean)} & {fmt_pt(c.E_R_mean)} $\\pm$ {fmt_pt(c.E_R_ci95)} & "
            f"{fmt_pt(c.E_TQ_mean)} & {fmt_pt(c.E_N_mean)} & {fmt_pt(c.little_product)} & "
            f"{fmt_pt(c.little_error_pct, 3)} & {fmt_pt(c.desvio_teoria_pct, 2)} & "
            f"{fmt_pt(c.ganho_vs_random_pct, 1)} \\\\"
        )

    coluna_t = "r" if convergencia else ""
    titulo_t = "$T$ & " if convergencia else ""
    return _tabular(
        [
            f"% Item (d): medido em {configs[0].replicas} replicas, "
            f"duracao {'/'.join(f'{d:.0f}' for d in duracoes)} u.t.",
            "% Em coluna dupla do IEEE, envolver em \\resizebox se necessario.",
        ],
        f"l{coluna_t}rrrrrrrrr",
        [
            f"    Política & {titulo_t}$\\lambda$ & $X$ & $E[R]$ & $E[T_Q]$ & $E[N]$ & "
            "$X \\cdot E[R]$ & Little & Desvio & Ganho \\\\",
            f"    & {'& ' if convergencia else ''}& & & & & & (\\%) & (\\%) & (\\%) \\\\",
        ],
        body,
    )


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def _plot_policies(ax, configs: List[ConfigResult], rename: Dict[str, str] = {}) -> None:
    """Uma curva E[R] x λ por política, com barras de erro de 95%."""
    for policy, style in POLICIES.items():
        pontos = sorted((c for c in configs if c.policy == policy), key=lambda c: c.lambda_rate)
        if not pontos:
            continue
        ax.errorbar(
            [c.lambda_rate for c in pontos],
            [c.E_R_mean for c in pontos],
            yerr=[c.E_R_ci95 for c in pontos],
            marker=style.marker, markersize=4, capsize=3, linewidth=1,
            label=rename.get(policy, style.label),
        )
    ax.set_xlabel(r"$\lambda$ (req/u.t.)")
    ax.set_ylabel(r"$E[R]$ (u.t.)")
    ax.grid(alpha=0.3)


def figure_response_time(configs: List[ConfigResult], path: Path) -> None:
    """A curva obrigatória: E[R] x λ com barras de erro e as duas curvas teóricas."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.7))
    _plot_policies(ax, configs)

    mu = configs[0].mu[0]
    grade = np.linspace(min(c.lambda_rate for c in configs), max(c.lambda_rate for c in configs), 200)
    ax.plot(grade, [AnalyticalModel.mm1_random_policy(l, NUM_SERVERS, mu)["E_R"] for l in grade],
            color="0.4", linewidth=0.9, label="M/M/1 (aleatória)")
    ax.plot(grade, [AnalyticalModel.round_robin_e3m1(l, NUM_SERVERS, mu)["E_R"] for l in grade],
            color="0.4", linewidth=0.9, linestyle="--", label="E$_3$/M/1 (round-robin)")

    ax.legend(fontsize=6.5)
    _save(fig, path)


def figure_instability(trajectories: List[Row], path: Path) -> None:
    """
    Item (f). Cor por SEMENTE e estilo de linha por política: o leitor vê três
    feixes e o colapso das políticas é a coincidência dentro de cada feixe.
    Colorir por política entrelaçaria as cores e esconderia o colapso.
    """
    curvas: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    for p in trajectories:
        curvas.setdefault((p["policy"], int(p["seed"])), []).append((p["t"], p["N"]))

    sementes = sorted({semente for _, semente in curvas})
    politicas = sorted({politica for politica, _ in curvas}, key=_policy_rank)
    cores = {semente: f"C{i}" for i, semente in enumerate(sementes)}

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.7))
    for (politica, semente), pontos in curvas.items():
        passo = max(1, len(pontos) // MAX_TRAJECTORY_POINTS)
        # O último ponto entra sempre: é o N(T_fim) que o texto cita, e o passo
        # normalmente não cai exatamente nele.
        recorte = pontos[::passo] + [pontos[-1]]
        ax.plot(
            [t for t, _ in recorte], [n for _, n in recorte],
            color=cores[semente], linestyle=POLICIES[politica].linestyle, linewidth=0.8,
        )

    lam = trajectories[0]["lambda"]
    inclinacao = lam - CAPACITY
    t_max = max(p["t"] for p in trajectories)
    ax.plot([0, t_max], [0, inclinacao * t_max], color="black", linestyle=(0, (6, 3)),
            linewidth=1.2)

    legendas = (
        [Line2D([], [], color=cores[s], label=f"semente {s}") for s in sementes]
        + [Line2D([], [], color="0.3", linestyle=POLICIES[p].linestyle, label=label_of(p))
           for p in politicas]
        + [Line2D([], [], color="black", linestyle=(0, (6, 3)),
                  label=f"reta {fmt_pt(inclinacao, 1)}·t")]
    )
    ax.set_xlabel(r"$t$ (u.t.)")
    ax.set_ylabel(r"$N(t)$ (requisições)")
    ax.grid(alpha=0.3)
    ax.legend(handles=legendas, fontsize=6, ncol=2)
    _save(fig, path)


def figure_heterogeneous(configs: List[ConfigResult], path: Path) -> None:
    """
    Painel do ponto extra. Sem curva teórica: com mu desigual, a aleatória sobrecarrega
    o servidor lento e a fórmula fechada da M/M/1 deixa de valer.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.7))
    _plot_policies(ax, configs, rename={LoadBalancerPolicy.RANDOM: "uniforme (1/3)"})

    # Eixo log: quando a uniforme satura um servidor, o E[R] dela fica duas ordens
    # de grandeza acima da proporcional e em escala linear a proporcional viraria
    # uma reta em zero.
    ax.set_yscale("log")
    ax.set_title(rf"$\mu = {list(configs[0].mu)}$", fontsize=8)
    ax.legend(fontsize=6.5)
    _save(fig, path)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    # PDF vetorial: a extensão decide o formato. Sem data de criação, rodar de novo
    # com os mesmos dados dá um arquivo idêntico e o git não acusa mudança.
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


# --------------------------------------------------------------------------
# Linha de comando
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agrega os CSVs de réplica e gera tabelas, figuras e verificações."
    )
    p.add_argument("--input", type=Path, nargs="+", default=[Path("results/metrics.csv")],
                   help="Um ou mais CSVs de réplica. mu ou duração diferentes viram grupos.")
    p.add_argument("--trajectories", type=Path,
                   default=Path("results/instability_trajectories.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


def main() -> None:
    setup()
    args = parse_args()

    for path in args.input:
        if not path.exists():
            sys.exit(
                f"{path.as_posix()} não existe.\n"
                "Rode antes: python -m experiments.run_experiments"
            )
    if not args.trajectories.exists():
        sys.exit(
            f"{args.trajectories.as_posix()} não existe (as trajetórias não entram no git).\n"
            "Rode antes: python -m experiments.run_instability"
        )

    rows = read_rows(args.input)
    results = aggregate(rows)
    print(f"{len(rows)} linhas de réplica em {len(results.configs)} configurações\n")

    figuras = args.output_dir / "figures"
    tabelas = args.output_dir / "tables"
    gerados = []

    csv_path = args.output_dir / "aggregated_metrics.csv"
    write_csv(csv_path, AGGREGATED_COLUMNS, aggregated_rows(results.configs))
    gerados.append(csv_path)

    # A curva e a teoria usam uma duração só (misturar duplicaria os pontos); a
    # tabela medida leva todas, porque é ela que mostra a convergência.
    base = base_group(results.configs)
    homogeneas = [c for c in results.configs if c.is_homogeneous]
    if base:
        tabelas.mkdir(parents=True, exist_ok=True)
        for nome, conteudo in [
            ("theoretical_table.tex", theoretical_table(base)),
            ("measured_table.tex", measured_table(homogeneas)),
        ]:
            (tabelas / nome).write_text(conteudo, encoding="utf-8")
            gerados.append(tabelas / nome)

        figure_response_time(base, figuras / "response_time_vs_lambda.pdf")
        gerados.append(figuras / "response_time_vs_lambda.pdf")
    else:
        print("Nenhuma configuração com mu homogêneo: tabelas e curva E[R] x λ não geradas.\n")

    figure_instability(read_rows([args.trajectories]), figuras / "instability.pdf")
    gerados.append(figuras / "instability.pdf")

    heterogeneas = [c for c in results.configs if not c.is_homogeneous]
    if heterogeneas:
        figure_heterogeneous(heterogeneas, figuras / "heterogeneous.pdf")
        gerados.append(figuras / "heterogeneous.pdf")

    texto = format_checks(results.checks)
    print(texto)
    checks_path = args.output_dir / "verifications.txt"
    checks_path.write_text(texto + "\n", encoding="utf-8")
    gerados.append(checks_path)

    print("\nGerado:")
    for path in gerados:
        print(f"  {path.as_posix()}")


if __name__ == "__main__":
    main()
