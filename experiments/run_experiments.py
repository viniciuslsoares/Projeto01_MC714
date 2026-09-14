"""
Executa o grid do enunciado e grava o CSV bruto — uma linha por réplica.

Sem argumento nenhum, reproduz o enunciado: 5 lambda x 3 políticas x 10 réplicas.
A linha de comando existe só para sobrepor (ponto extra heterogêneo, horizonte
mais longo), para não virar um script copiado que divirja deste.

A agregação (média e IC 95%) não acontece aqui: quem tem o bruto reagrega
relendo o arquivo, sem re-simular.

Uso:
    python -m experiments.run_experiments
    python -m experiments.run_experiments --duration 20000 --output results/metrics_20k.csv
    python -m experiments.run_experiments --policies proportional --mu 1.5 1.0 0.5
"""

import argparse
from pathlib import Path
from typing import Callable, List, Optional

from analysis.analytical_model import AnalyticalModel
from experiments.grid import (
    METRIC_COLUMNS,
    GridConfig,
    Row,
    execute_grid,
    mean_of,
    setup,
    write_csv,
    write_metadata,
)
from src.load_balancer import LoadBalancerPolicy

# Padrão do enunciado. lambda = 3.3 fica no run_instability: sem regime
# estacionário, uma linha dele aqui seria lida como comparável às outras.
LAMBDAS = [0.6, 1.2, 1.8, 2.4, 2.7]
POLICIES = [
    LoadBalancerPolicy.RANDOM,
    LoadBalancerPolicy.ROUND_ROBIN,
    LoadBalancerPolicy.SHORTEST_QUEUE,
]
REPLICAS = 10
DURATION = 5000.0
WARMUP = 500.0
MU = [1.0, 1.0, 1.0]
OUTPUT = Path("results/metrics.csv")


def theoretical_e_r(lam: float, mu: List[float]) -> Optional[float]:
    """E[R] teórico da política aleatória. Só vale com servidores iguais e estável."""
    if len(set(mu)) != 1:
        return None
    res = AnalyticalModel.mm1_random_policy(lam, num_servers=len(mu), mu=mu[0])
    return res["E_R"] if res["is_stable"] else None


def make_progress_printer(mu: List[float]) -> Callable[[str, float, List[Row]], None]:
    """Imprime o medido ao lado do teórico: um absurdo aparece durante a execução."""
    def print_progress(policy: str, lam: float, rows: List[Row]) -> None:
        teorico = theoretical_e_r(lam, mu)
        referencia = "sem referência" if teorico is None else f"aleatória teórica {teorico:.4f}"
        print(f"  {policy:<15} λ={lam:<5.2f} E[R] = {mean_of(rows, 'E_R'):7.4f}  ({referencia})")

    return print_progress


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Executa o grid do enunciado e grava o CSV bruto (uma linha por réplica)."
    )
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    p.add_argument("--policies", nargs="+", default=POLICIES)
    p.add_argument("--replicas", type=int, default=REPLICAS)
    p.add_argument("--duration", type=float, default=DURATION)
    p.add_argument("--warmup", type=float, default=WARMUP)
    p.add_argument("--mu", type=float, nargs=3, default=MU)
    p.add_argument("--weights", type=float, nargs=3, default=None,
                   help="Pesos da política proporcional. Padrão: p_i = mu_i / soma(mu).")
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--quick", action="store_true",
                   help="2 réplicas e duração 1000, para depurar sem esperar o grid inteiro.")
    return p.parse_args()


def main() -> None:
    setup()
    args = parse_args()
    if args.quick:
        args.replicas = 2
        args.duration = 1000.0

    config = GridConfig(
        lambdas=args.lambdas,
        policies=args.policies,
        replicas=args.replicas,
        duration=args.duration,
        warmup=args.warmup,
        mu=args.mu,
        weights=args.weights,
    )

    total = len(config.policies) * len(config.lambdas) * config.replicas
    print(
        f"Grid: {len(config.policies)} políticas x {len(config.lambdas)} lambdas "
        f"x {config.replicas} réplicas = {total} execuções"
    )
    print(
        f"Duração {config.duration:.0f} | warm-up {config.warmup:.0f} | mu {config.mu} | "
        f"sementes {config.seed_base + 1}..{config.seed_base + config.replicas}\n"
    )

    results = execute_grid(config, on_config_done=make_progress_printer(config.mu))

    write_csv(args.output, METRIC_COLUMNS, results.metric_rows)
    meta_path = write_metadata(args.output, config)

    print(f"\n{len(results.metric_rows)} linhas gravadas em {args.output.as_posix()}")
    print(f"Proveniência em {meta_path.as_posix()}")


if __name__ == "__main__":
    main()
