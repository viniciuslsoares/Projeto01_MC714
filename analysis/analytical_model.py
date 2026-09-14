"""
Módulo com as fórmulas fechadas da Modelagem Analítica (Teoria de Filas).
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import brentq


class AnalyticalModel:
    """
    Implementação das fórmulas analíticas para filas M/M/1 e M/M/1/K.
    """

    @staticmethod
    def mm1_random_policy(
        lambda_rate: float,
        num_servers: int = 3,
        mu: float = 1.0
    ) -> Dict[str, float]:
        """
        Calcula as métricas analíticas exatas para o sistema base sob política aleatória.
        
        Cada servidor é modelado como uma fila M/M/1 independente com taxa lambda_i = lambda / num_servers.
        """
        lambda_i = lambda_rate / num_servers
        rho = lambda_i / mu

        if rho >= 1.0:
            return {
                "is_stable": False,
                "lambda": lambda_rate,
                "lambda_i": lambda_i,
                "mu": mu,
                "rho": rho,
                "U_i": 1.0,
                "E_R": float("inf"),
                "E_TQ": float("inf"),
                "E_Ni": float("inf"),
                "E_N": float("inf"),
                "X": num_servers * mu  # Vazão saturada máxima
            }

        U_i = rho
        E_Ni = rho / (1.0 - rho)
        E_N = num_servers * E_Ni
        E_R = 1.0 / (mu - lambda_i)
        E_TQ = E_R - (1.0 / mu)
        X = lambda_rate

        return {
            "is_stable": True,
            "lambda": lambda_rate,
            "lambda_i": lambda_i,
            "mu": mu,
            "rho": rho,
            "U_i": U_i,
            "E_R": E_R,
            "E_TQ": E_TQ,
            "E_Ni": E_Ni,
            "E_N": E_N,
            "X": X
        }

    @staticmethod
    def round_robin_e3m1(
        lambda_rate: float,
        num_servers: int = 3,
        mu: float = 1.0
    ) -> Dict[str, float]:
        """
        Calcula E[R] da política round-robin, que NÃO é uma M/M/1.

        Cada servidor recebe a cada num_servers-ésima chegada, então o intervalo
        entre chegadas nele é Erlang-k, não exponencial: o servidor é uma E_k/M/1,
        cuja solução exata é a raiz sigma em (0, 1) de

            sigma = A*(mu (1 - sigma)),   A*(s) = (lambda / (lambda + s))^k

        com E[T_Q] = sigma / (mu (1 - sigma)) e E[R] = E[T_Q] + 1/mu.

        ATENÇÃO: E_k/M/1 é teoria externa (GI/M/1, Harchol-Balter) e não está nos
        slides da Aula 5. Declarar a origem no relatório.
        """
        rho = lambda_rate / (num_servers * mu)

        if rho >= 1.0:
            return {
                "is_stable": False,
                "lambda": lambda_rate,
                "mu": mu,
                "rho": rho,
                "sigma": 1.0,
                "E_TQ": float("inf"),
                "E_R": float("inf")
            }

        def fixed_point(sigma: float) -> float:
            transform = (lambda_rate / (lambda_rate + mu * (1.0 - sigma))) ** num_servers
            return transform - sigma

        # sigma = 1 é sempre raiz e não é a que interessa. O intervalo isola a
        # outra: fixed_point(0) > 0 e fixed_point(1 - eps) < 0.
        sigma = float(brentq(fixed_point, 0.0, 1.0 - 1e-9))
        E_TQ = sigma / (mu * (1.0 - sigma))

        return {
            "is_stable": True,
            "lambda": lambda_rate,
            "mu": mu,
            "rho": rho,
            "sigma": sigma,
            "E_TQ": E_TQ,
            "E_R": E_TQ + 1.0 / mu
        }

    @staticmethod
    def mm1k_random_policy(
        lambda_rate: float,
        K: int,
        num_servers: int = 3,
        mu: float = 1.0
    ) -> Dict[str, float]:
        """
        Calcula as métricas analíticas exatas para servidor com Buffer Finito (M/M/1/K).
        
        K = capacidade total do nó (1 em serviço + fila de tamanho K-1).
        """
        lambda_i = lambda_rate / num_servers
        rho = lambda_i / mu

        if abs(rho - 1.0) < 1e-9:
            # Caso especial rho == 1
            p0 = 1.0 / (K + 1)
            p_K = p0
            E_Ni = K / 2.0
        else:
            p0 = (1.0 - rho) / (1.0 - (rho ** (K + 1)))
            p_K = p0 * (rho ** K)
            
            # E[N_i] = sum(k * p_k)
            k_indices = np.arange(0, K + 1)
            p_k = p0 * (rho ** k_indices)
            E_Ni = float(np.sum(k_indices * p_k))

        p_drop = p_K
        lambda_effective_i = lambda_i * (1.0 - p_drop)
        X_effective = num_servers * lambda_effective_i
        
        # Pela Lei de Little no fluxo admitido:
        E_R = E_Ni / lambda_effective_i if lambda_effective_i > 0 else (1.0 / mu)
        E_N = num_servers * E_Ni
        U_i = 1.0 - p0

        return {
            "lambda": lambda_rate,
            "K": K,
            "rho": rho,
            "P_perda": p_drop,
            "X_efetivo": X_effective,
            "E_R": E_R,
            "E_Ni": E_Ni,
            "E_N": E_N,
            "U_i": U_i
        }

    @staticmethod
    def heterogeneous_analysis(
        lambda_rate: float,
        mu_list: List[float] = [1.5, 1.0, 0.5]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analisa o impacto da heterogeneidade sob política Uniforme (1/3) vs Proporcional (p_i = mu_i / sum(mu)).
        """
        sum_mu = sum(mu_list)
        num_servers = len(mu_list)
        
        # 1. Política Uniforme (1/N)
        uniform_res = {}
        for i, mu in enumerate(mu_list):
            lam_i = lambda_rate / num_servers
            rho_i = lam_i / mu
            stable = rho_i < 1.0
            e_r = (1.0 / (mu - lam_i)) if stable else float("inf")
            uniform_res[f"server_{i+1}"] = {
                "mu": mu,
                "lambda_i": lam_i,
                "rho_i": rho_i,
                "is_stable": stable,
                "E_R_i": e_r
            }

        # 2. Política Proporcional (p_i = mu_i / sum_mu)
        proportional_res = {}
        rho_global = lambda_rate / sum_mu
        for i, mu in enumerate(mu_list):
            p_i = mu / sum_mu
            lam_i = lambda_rate * p_i
            rho_i = lam_i / mu  # Sempre igual a rho_global
            stable = rho_i < 1.0
            e_r = (1.0 / (mu - lam_i)) if stable else float("inf")
            proportional_res[f"server_{i+1}"] = {
                "mu": mu,
                "weight_p_i": p_i,
                "lambda_i": lam_i,
                "rho_i": rho_i,
                "is_stable": stable,
                "E_R_i": e_r
            }

        return {
            "uniform": uniform_res,
            "proportional": proportional_res,
            "global_stable_proportional": rho_global < 1.0
        }
