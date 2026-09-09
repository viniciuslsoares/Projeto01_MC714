"""
Script de validação e checagem de sanidade do Simulador de Balanceamento de Carga.
Executa testes rápidos e compara os resultados simulados com as fórmulas analíticas teóricas.
Grava a saída detalhada no diretório results/.
"""

import sys
import os
from pathlib import Path
from src.simulator import Simulator
from src.load_balancer import LoadBalancerPolicy
from src.metrics import aggregate_replications
from analysis.analytical_model import AnalyticalModel


class DualOutput:
    """Redireciona a saída simultaneamente para o terminal (stdout) e para um arquivo."""
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def run_sanity_tests():
    # Garante a existência do diretório results/
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "sanity_test_results.txt"

    original_stdout = sys.stdout
    dual_out = DualOutput(output_path)
    sys.stdout = dual_out

    try:
        print("=" * 80)
        print(" MC714 - TESTES DE SANIDADE DO SIMULADOR (DISCRETE EVENT SIMULATION)")
        print("=" * 80)

        # -------------------------------------------------------------
        # Teste 1: Carga Quase Nula (lambda = 0.1, mu = 1.0)
        # -------------------------------------------------------------
        print("\n[TESTE 1] Cheque de Sanidade com Carga Baixa (lambda = 0.1 req/u.t.)")
        print("Teoria: E[R] deve tender a E[S] = 1.0 u.t. e X approx 0.1 req/u.t.")
        
        sim_low = Simulator(
            lambda_rate=0.1,
            policy=LoadBalancerPolicy.RANDOM,
            duration=5000.0,
            warmup_time=500.0,
            seed=42
        )
        res_low = sim_low.run()
        theory_low = AnalyticalModel.mm1_random_policy(0.1)
        
        print(f" -> E[R] Teórico: {theory_low['E_R']:.4f} u.t. | E[R] Simulado: {res_low.avg_response_time:.4f} u.t.")
        print(f" -> E[N] Teórico: {theory_low['E_N']:.4f}      | E[N] Simulado: {res_low.avg_requests_in_system:.4f}")
        print(f" -> Vazão X Teórica: {theory_low['X']:.4f}      | Vazão X Simulada: {res_low.throughput:.4f}")
        
        # Compara com o valor teórico esperado (1.0345) dentro da margem estatística
        assert abs(res_low.avg_response_time - theory_low['E_R']) < 0.10, "Falha no teste de sanidade lambda -> 0!"
        print(" [OK] Teste 1 aprovado com sucesso!\n")

        # -------------------------------------------------------------
        # Teste 2: Carga Média (lambda = 1.8, mu = 1.0) - 10 Réplicas
        # -------------------------------------------------------------
        print("-" * 80)
        print("[TESTE 2] Comparação das 3 Políticas com lambda = 1.8 req/u.t. (10 réplicas)")
        theory_mid = AnalyticalModel.mm1_random_policy(1.8)
        print("Valores Analíticos Teóricos (M/M/1 - Aleatória):")
        print(f" -> E[R] Teórico  = {theory_mid['E_R']:.4f} u.t.")
        print(f" -> E[N] Teórico  = {theory_mid['E_N']:.4f} requisições")
        print(f" -> Utilização U_i = {theory_mid['U_i']:.4f}")
        print(f" -> Vazão X       = {theory_mid['X']:.4f} req/u.t.\n")

        policies = [
            ("Aleatória (Random)", LoadBalancerPolicy.RANDOM),
            ("Round-Robin", LoadBalancerPolicy.ROUND_ROBIN),
            ("Fila Mais Curta (JSQ)", LoadBalancerPolicy.SHORTEST_QUEUE)
        ]

        for label, pol in policies:
            rep_metrics = []
            for seed in range(1, 11):
                sim = Simulator(
                    lambda_rate=1.8,
                    policy=pol,
                    duration=5000.0,
                    warmup_time=500.0,
                    seed=seed
                )
                rep_metrics.append(sim.run())
            
            agg = aggregate_replications(rep_metrics)
            print(f"Política: {label}")
            print(f" -> E[R] = {agg.response_time_mean:.4f} ± {agg.response_time_ci95:.4f} u.t.")
            print(f" -> E[N] = {agg.avg_n_mean:.4f} ± {agg.avg_n_ci95:.4f} reqs")
            print(f" -> X    = {agg.throughput_mean:.4f} ± {agg.throughput_ci95:.4f} req/u.t.")
            print(f" -> U_i  = {[round(u, 4) for u in agg.utilization_means]}")
            print(f" -> Lei de Little (E[N] vs X*E[R]): {agg.avg_n_mean:.4f} vs {agg.little_law_product:.4f} (Erro: {agg.little_law_error_percent:.2f}%)")
            print()

        print("=" * 80)
        print(" TODOS OS TESTES DE SANIDADE FORAM CONCLUÍDOS COM SUCESSO!")
        print(f" [INFO] Resultados gravados em: {output_path.as_posix()}")
        print("=" * 80)

    finally:
        sys.stdout = original_stdout
        dual_out.close()


if __name__ == "__main__":
    run_sanity_tests()
