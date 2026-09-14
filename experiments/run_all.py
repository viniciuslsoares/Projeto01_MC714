"""
Porta da frente: gera todos os resultados na ordem certa, com um comando.

Os runners individuais continuam existindo para os integrantes trabalharem em
paralelo; este é para quem só quer avaliar o trabalho.

Uso:
    python -m experiments.run_all
"""

import subprocess
import sys

from experiments.grid import setup

HETERO_OUTPUT = "results/metrics_hetero.csv"

ETAPAS = [
    ("Grid do enunciado (5 lambdas x 3 políticas x 10 réplicas)",
     "experiments.run_experiments", []),
    ("Regime instável lambda = 3.3 (item f)",
     "experiments.run_instability", []),
    ("Ponto extra: servidores heterogêneos, uniforme x proporcional",
     "experiments.run_experiments",
     ["--policies", "random", "proportional", "--mu", "1.5", "1.0", "0.5",
      "--output", HETERO_OUTPUT]),
    # Último passo: só relê os CSVs, nunca simula.
    ("Análise: tabelas, figuras e verificações",
     "analysis.analise", ["--input", "results/metrics.csv", HETERO_OUTPUT]),
]


def main() -> None:
    setup()

    for titulo, modulo, argumentos in ETAPAS:
        # flush=True: sem isto, ao redirecionar a saída para um arquivo, os títulos
        # sairiam depois do texto dos subprocessos.
        print(f"\n{'=' * 72}\n {titulo}\n{'=' * 72}", flush=True)
        # sys.executable garante o mesmo Python (e o mesmo venv) desta execução.
        done = subprocess.run([sys.executable, "-m", modulo] + argumentos)
        if done.returncode != 0:
            sys.exit(f"\n{modulo} falhou. Nada mais será executado.")

    print(f"\n{'=' * 72}\n Tudo gerado em results/.\n{'=' * 72}")


if __name__ == "__main__":
    main()
