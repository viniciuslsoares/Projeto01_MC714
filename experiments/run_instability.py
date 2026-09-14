"""
Executa o regime instável lambda = 3.3 (item f) e grava a trajetória N(t).

Aqui a métrica não é uma média — é uma reta. As chegadas (3,3) superam a
capacidade (3*mu = 3,0) e a fila cresce sem limite. Daí:

- warm-up = 0: o warm-up serve para chegar a um estado típico, e aqui não existe
  estado típico. Descartar os primeiros 500 u.t. jogaria fora o começo da reta.
- E[R] e E[N] não são gravados: sem regime estacionário eles só crescem com a
  duração escolhida. X é gravado porque satura em 3*mu, e isso é interpretável.
- Trajetória evento a evento: N(t) é função escada e só muda em evento, então o
  registro por evento é exato e a grade fixa sempre pode ser derivada dele.
- As três políticas, porque o colapso das trajetórias numa mesma reta é um
  resultado: nenhuma política salva um sistema sobrecarregado.

As trajetórias não entram no git — com semente fixa são regeneráveis em segundos.
O resumo curto é versionado.

Uso:
    python -m experiments.run_instability
    python -m experiments.run_instability --duration 20000
"""

import argparse
from pathlib import Path
from typing import Callable, List

from experiments.grid import (
    SUMMARY_COLUMNS,
    TRAJECTORY_COLUMNS,
    GridConfig,
    GridResults,
    Row,
    execute_grid,
    mean_of,
    setup,
    write_csv,
    write_metadata,
)
from src.load_balancer import LoadBalancerPolicy

LAMBDA_UNSTABLE = 3.3
POLICIES = [
    LoadBalancerPolicy.RANDOM,
    LoadBalancerPolicy.ROUND_ROBIN,
    LoadBalancerPolicy.SHORTEST_QUEUE,
]
REPLICAS = 3        # 3 sementes: a inclinação é do sistema, não de uma semente azarada
DURATION = 5000.0   # paridade com o grid principal
WARMUP = 0.0
MU = [1.0, 1.0, 1.0]
TRAJECTORY_OUTPUT = Path("results/instability_trajectories.csv")
SUMMARY_OUTPUT = Path("results/instability_summary.csv")


def summary_rows(results: GridResults) -> List[Row]:
    """Resumo curto: só o que é interpretável sem regime estacionário."""
    n_final = {}
    for p in results.trajectory_rows:
        n_final[(p["policy"], p["seed"])] = p["N"]  # o último ponto sobrescreve

    return [
        {
            "policy": r["policy"],
            "lambda": r["lambda"],
            "seed": r["seed"],
            "duration": r["duration"],
            "X": r["X"],
            "N_final": n_final[(r["policy"], r["seed"])],
            "U0": r["U0"],
            "U1": r["U1"],
            "U2": r["U2"],
        }
        for r in results.metric_rows
    ]


def make_progress_printer(mu: List[float]) -> Callable[[str, float, List[Row]], None]:
    """A vazão medida ao lado da capacidade: é nela que X deve travar."""
    def print_progress(policy: str, lam: float, rows: List[Row]) -> None:
        print(f"  {policy:<15} X = {mean_of(rows, 'X'):.4f}  (capacidade soma(mu) = {sum(mu):.4f})")

    return print_progress


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Executa o regime instável lambda = 3.3 e grava a trajetória N(t)."
    )
    p.add_argument("--lambda-rate", type=float, default=LAMBDA_UNSTABLE, dest="lambda_rate")
    p.add_argument("--policies", nargs="+", default=POLICIES)
    p.add_argument("--replicas", type=int, default=REPLICAS)
    p.add_argument("--duration", type=float, default=DURATION)
    p.add_argument("--mu", type=float, nargs=3, default=MU)
    p.add_argument("--trajectory-output", type=Path, default=TRAJECTORY_OUTPUT)
    p.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    return p.parse_args()


def main() -> None:
    setup()
    args = parse_args()

    config = GridConfig(
        lambdas=[args.lambda_rate],
        policies=args.policies,
        replicas=args.replicas,
        duration=args.duration,
        warmup=WARMUP,
        mu=args.mu,
        track_trajectory=True,
    )

    capacity = sum(config.mu)
    slope = args.lambda_rate - capacity
    print(
        f"Regime instável: λ={args.lambda_rate} contra capacidade {capacity:.1f} "
        f"| {len(config.policies)} políticas x {config.replicas} sementes"
    )
    print(
        f"Duração {config.duration:.0f} | warm-up 0 | "
        f"reta prevista N(t) = {slope:.1f}t → N({config.duration:.0f}) "
        f"= {slope * config.duration:.0f}\n"
    )

    results = execute_grid(config, on_config_done=make_progress_printer(config.mu))
    summary = summary_rows(results)

    write_csv(args.trajectory_output, TRAJECTORY_COLUMNS, results.trajectory_rows)
    write_csv(args.summary_output, SUMMARY_COLUMNS, summary)
    trajectory_meta = write_metadata(args.trajectory_output, config)
    summary_meta = write_metadata(args.summary_output, config)

    print(f"\nN final medido (média das sementes): {mean_of(summary, 'N_final'):.0f}")
    print(f"{len(results.trajectory_rows)} pontos de trajetória em {args.trajectory_output.as_posix()}")
    print(f"{len(summary)} linhas de resumo em {args.summary_output.as_posix()}")
    print(f"Proveniência em {trajectory_meta.as_posix()} e {summary_meta.as_posix()}")


if __name__ == "__main__":
    main()
