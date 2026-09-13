"""
Módulo para consolidação estatística, cálculo de intervalos de confiança (t-Student) e formatação de métricas.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats

from src.models import SimulationMetrics


@dataclass
class AggregatedMetrics:
    """
    Estrutura que armazena a média amostral e o intervalo de confiança de 95%
    para 10 réplicas independentes de uma dada configuração.
    """
    policy_name: str
    lambda_rate: float
    num_replicas: int

    # Throughput (X)
    throughput_mean: float
    throughput_ci95: float

    # Tempo Médio de Resposta E[R]
    response_time_mean: float
    response_time_ci95: float

    # Tempo Médio em Fila E[T_Q]
    queue_time_mean: float
    queue_time_ci95: float

    # Número Médio no Sistema E[N]
    avg_n_mean: float
    avg_n_ci95: float

    # Utilização média dos servidores
    utilization_means: List[float]
    utilization_ci95: List[float]

    # Probabilidade de perda (buffer finito)
    drop_prob_mean: float
    drop_prob_ci95: float

    # Verificação da Lei de Little: E[N] vs X * E[R]
    little_law_product: float
    little_law_error_percent: float


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcula a média e o semi-intervalo de confiança usando a distribuição t de Student.
    
    Returns:
        (media, margem_de_erro_ci95)
    """
    n = len(data)
    if n < 2:
        return float(np.mean(data)), 0.0

    mean = float(np.mean(data))
    std_err = float(stats.sem(data))  # s / sqrt(n)
    h = float(std_err * stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    return mean, h


def aggregate_replications(metrics_list: List[SimulationMetrics]) -> AggregatedMetrics:
    """
    Agrega os resultados de múltiplas réplicas (sementes) calculando médias e ICs de 95%.
    """
    if not metrics_list:
        raise ValueError("Lista de métricas vazia.")

    policy_name = metrics_list[0].policy_name
    lambda_rate = metrics_list[0].lambda_rate
    n = len(metrics_list)

    # Coleta de vetores
    throughputs = [m.throughput for m in metrics_list]
    resp_times = [m.avg_response_time for m in metrics_list]
    queue_times = [m.avg_queue_time for m in metrics_list]
    avg_ns = [m.avg_requests_in_system for m in metrics_list]
    drop_probs = [m.drop_probability for m in metrics_list]

    # Cálculos dos ICs 95%
    th_mean, th_ci = compute_confidence_interval(throughputs)
    resp_mean, resp_ci = compute_confidence_interval(resp_times)
    queue_mean, queue_ci = compute_confidence_interval(queue_times)
    n_mean, n_ci = compute_confidence_interval(avg_ns)
    drop_mean, drop_ci = compute_confidence_interval(drop_probs)

    # Utilização por servidor
    num_servers = len(metrics_list[0].server_utilizations)
    u_means = []
    u_cis = []
    for s_idx in range(num_servers):
        u_s = [m.server_utilizations[s_idx] for m in metrics_list]
        u_m, u_c = compute_confidence_interval(u_s)
        u_means.append(u_m)
        u_cis.append(u_c)

    # Validação da Lei de Little: E[N] = X * E[R]
    little_product = th_mean * resp_mean
    if n_mean > 0:
        little_error = abs(n_mean - little_product) / n_mean * 100.0
    else:
        little_error = 0.0

    return AggregatedMetrics(
        policy_name=policy_name,
        lambda_rate=lambda_rate,
        num_replicas=n,
        throughput_mean=th_mean,
        throughput_ci95=th_ci,
        response_time_mean=resp_mean,
        response_time_ci95=resp_ci,
        queue_time_mean=queue_mean,
        queue_time_ci95=queue_ci,
        avg_n_mean=n_mean,
        avg_n_ci95=n_ci,
        utilization_means=u_means,
        utilization_ci95=u_cis,
        drop_prob_mean=drop_mean,
        drop_prob_ci95=drop_ci,
        little_law_product=little_product,
        little_law_error_percent=little_error
    )
