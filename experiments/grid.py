"""
Executor de grade de experimentos — o ponto de execução compartilhado.

Os dois runners (run_experiments e run_instability) são casos do MESMO
experimento: rodar uma grade de configurações e coletar resultados. Só mudam os
parâmetros e o que se grava. A lógica fica aqui; os runners são invólucros de
linha de comando + escrita de arquivo.

SEMENTES: seed = seed_base + réplica, ou seja 1001..1010. A mesma semente é usada
nas três políticas e em todos os lambda.

- Entre POLÍTICAS é obrigatório: é o que faz cada réplica entregar as mesmas
  chegadas e as mesmas demandas de serviço às três (variáveis aleatórias comuns),
  de modo que a diferença medida em E[R] venha da política, não do sorteio.
- Entre LAMBDA é escolha: os pontos compartilham ruído e a curva E[R] x lambda
  sai lisa. Em troca, os cinco pontos não são independentes entre si. Os ICs por
  lambda não mudam, porque as réplicas dentro de cada lambda seguem independentes.
"""

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.models import SimulationMetrics
from src.simulator import Simulator

MIN_PYTHON = (3, 10)

Row = Dict[str, Any]

# Ordem das colunas nas saídas. A linha é a RÉPLICA, então as utilizações ficam em
# colunas (U0/U1/U2). As colunas de parâmetros são redundantes de propósito: é o
# que permite concatenar arquivos de durações ou mu diferentes sem ambiguidade.
METRIC_COLUMNS = [
    "policy", "lambda", "replica", "seed", "duration", "warmup", "mu0", "mu1", "mu2",
    "X", "E_R", "E_TQ", "E_N", "U0", "U1", "U2",
    "n_arrived", "n_after_warmup", "n_completed", "drop_prob",
]
TRAJECTORY_COLUMNS = ["policy", "lambda", "seed", "t", "N"]
SUMMARY_COLUMNS = ["policy", "lambda", "seed", "duration", "X", "N_final", "U0", "U1", "U2"]


@dataclass
class GridConfig:
    """Uma grade de configurações a executar."""
    lambdas: List[float]
    policies: List[str]
    replicas: int
    duration: float = 5000.0
    warmup: float = 500.0
    mu: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    weights: Optional[List[float]] = None
    track_trajectory: bool = False
    seed_base: int = 1000


@dataclass
class GridResults:
    """Dados brutos da grade. Agregar é trabalho da análise, não daqui."""
    metric_rows: List[Row]       # uma linha por (política, lambda, réplica)
    trajectory_rows: List[Row]   # vazio quando track_trajectory=False


def execute_grid(
    config: GridConfig,
    on_config_done: Optional[Callable[[str, float, List[Row]], None]] = None,
) -> GridResults:
    """
    Roda a grade inteira acumulando os resultados em memória.

    Nada é gravado aqui: quem grava é o runner, uma vez, no fim. Se um assert do
    simulador disparar, a exceção sobe e nenhum arquivo é escrito — um CSV com
    uma réplica faltando parece completo e falsearia o IC em silêncio.

    on_config_done é chamado ao terminar cada (política, lambda), para o runner
    imprimir progresso sem que a impressão entre aqui.
    """
    if config.duration <= config.warmup:
        raise ValueError(
            f"duration ({config.duration}) precisa ser maior que warmup ({config.warmup})."
        )
    if len(config.mu) != 3:
        raise ValueError(f"O trabalho usa 3 servidores; recebi mu={config.mu}.")

    metric_rows: List[Row] = []
    trajectory_rows: List[Row] = []

    for policy in config.policies:
        for lam in config.lambdas:
            config_rows: List[Row] = []
            for replica in range(1, config.replicas + 1):
                seed = config.seed_base + replica
                sim = Simulator(
                    lambda_rate=lam,
                    policy=policy,
                    mu_list=config.mu,
                    duration=config.duration,
                    warmup_time=config.warmup,
                    seed=seed,
                    track_trajectory=config.track_trajectory,
                    weights=config.weights,
                )
                metrics = sim.run()
                config_rows.append(_metric_row(config, policy, lam, replica, seed, metrics))

                if config.track_trajectory:
                    times, n_values = sim.get_trajectory()
                    trajectory_rows.extend(
                        {"policy": policy, "lambda": lam, "seed": seed, "t": t, "N": n}
                        for t, n in zip(times, n_values)
                    )

            metric_rows.extend(config_rows)
            if on_config_done is not None:
                on_config_done(policy, lam, config_rows)

    return GridResults(metric_rows=metric_rows, trajectory_rows=trajectory_rows)


def _metric_row(
    config: GridConfig,
    policy: str,
    lam: float,
    replica: int,
    seed: int,
    m: SimulationMetrics,
) -> Row:
    """Monta a linha do CSV de UMA réplica."""
    row: Row = {
        "policy": policy,
        "lambda": lam,
        "replica": replica,
        "seed": seed,
        "duration": config.duration,
        "warmup": config.warmup,
        "X": m.throughput,
        "E_R": m.avg_response_time,
        "E_TQ": m.avg_queue_time,
        "E_N": m.avg_requests_in_system,
        # Contadores para auditar a drenagem depois, sem re-simular.
        "n_arrived": m.total_requests_arrived,
        "n_after_warmup": m.requests_after_warmup,
        "n_completed": m.completed_requests,
        "drop_prob": m.drop_probability,
    }
    for i, mu_i in enumerate(config.mu):
        row[f"mu{i}"] = mu_i
    for i, u in enumerate(m.server_utilizations):
        row[f"U{i}"] = u
    return row


# --------------------------------------------------------------------------
# Utilidades compartilhadas pelos runners
# --------------------------------------------------------------------------

def setup() -> None:
    """Chamar no início de cada runner, antes do primeiro print."""
    # A reconfiguração vem ANTES da checagem de versão, porque a mensagem de erro
    # tem acento: o terminal padrão do Windows não usa UTF-8 e sem isto o aviso
    # de "Python 3.10 é necessário" morreria com UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} ou superior é necessário; "
            f"detectado {sys.version_info.major}.{sys.version_info.minor}."
        )


def mean_of(rows: List[Row], column: str) -> float:
    """Média de uma coluna das linhas de resultado."""
    return sum(r[column] for r in rows) / len(rows)


def write_csv(path: Path, columns: List[str], rows: List[Row]) -> None:
    """
    Grava em arquivo temporário e só então renomeia.

    Uma interrupção deixa o temporário para trás, nunca um CSV pela metade que
    parece completo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_metadata(path: Path, config: GridConfig) -> Path:
    """Grava ao lado da saída o que nenhuma coluna responde: qual código rodou."""
    meta = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        # Com "python -m", sys.argv[0] é o caminho do arquivo, que não serve para
        # repetir o comando.
        "comando": " ".join(
            [f"python -m experiments.{Path(sys.argv[0]).stem}"] + sys.argv[1:]
        ),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "config": asdict(config),
    }
    meta_path = path.with_name(path.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def _git_commit() -> str:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return done.stdout.strip() or "desconhecido"
    except Exception:
        return "desconhecido"
