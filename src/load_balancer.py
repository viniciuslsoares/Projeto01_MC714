"""
Módulo contendo a implementação do balanceador de carga e suas políticas de despacho.
"""

from typing import List, Optional
import numpy as np

from src.server import Server


class LoadBalancerPolicy:
    """Constantes para os nomes das políticas de balanceamento."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    SHORTEST_QUEUE = "shortest_queue"
    PROPORTIONAL = "proportional"  # Para servidores heterogêneos


class LoadBalancer:
    """
    Balanceador de Carga (Dispatcher).
    
    Responsável por receber cada requisição e determinar para qual dos servidores
    ela deve ser encaminhada, com base na política configurada.
    """

    def __init__(
        self,
        policy: str = LoadBalancerPolicy.RANDOM,
        weights: Optional[List[float]] = None
    ):
        """
        Inicializa o balanceador.
        
        Args:
            policy: Nome da política ("random", "round_robin", "shortest_queue", "proportional").
            weights: Vetor de probabilidades/pesos para a política proporcional (opcional).
        """
        self.policy = policy.lower()
        self.rr_index = 0
        self.weights = weights

        valid_policies = [
            LoadBalancerPolicy.RANDOM,
            LoadBalancerPolicy.ROUND_ROBIN,
            LoadBalancerPolicy.SHORTEST_QUEUE,
            LoadBalancerPolicy.PROPORTIONAL,
        ]
        if self.policy not in valid_policies:
            raise ValueError(f"Política inválida: {self.policy}. Opções: {valid_policies}")

    def select_server(self, servers: List[Server], rng: np.random.Generator) -> Server:
        """
        Seleciona o servidor de destino para a requisição de acordo com a política.
        
        Args:
            servers: Lista de instâncias de Server disponíveis.
            rng: Gerador pseudoaleatório NumPy associado à réplica.
            
        Returns:
            A instância de Server selecionada.
        """
        num_servers = len(servers)
        if num_servers == 0:
            raise RuntimeError("Nenhum servidor disponível no balanceador.")

        # 1. Política de Escolha Aleatória (Probabilidade uniforme 1/N)
        if self.policy == LoadBalancerPolicy.RANDOM:
            idx = rng.integers(0, num_servers)
            return servers[idx]

        # 2. Política Round-Robin (Cíclica: 0, 1, 2, ..., N-1, 0, ...)
        elif self.policy == LoadBalancerPolicy.ROUND_ROBIN:
            selected = servers[self.rr_index]
            self.rr_index = (self.rr_index + 1) % num_servers
            return selected

        # 3. Política Fila Mais Curta (JSQ - Join the Shortest Queue)
        elif self.policy == LoadBalancerPolicy.SHORTEST_QUEUE:
            # Consulta a ocupação total (em fila + em atendimento) de cada servidor
            occupancies = [s.total_requests for s in servers]
            min_occ = min(occupancies)
            
            # Identifica todos os servidores que atingiram a ocupação mínima (possíveis empates)
            candidates = [i for i, occ in enumerate(occupancies) if occ == min_occ]
            
            # Se houver mais de um servidor com a menor fila, desempata aleatoriamente
            chosen_idx = rng.choice(candidates)
            return servers[chosen_idx]

        # 4. Política Proporcional (para Servidores Heterogêneos)
        elif self.policy == LoadBalancerPolicy.PROPORTIONAL:
            if self.weights is None:
                # Calcula pesos proporcionais às taxas mu_i se não fornecidos explicitamente
                mu_sum = sum(s.mu for s in servers)
                probs = [s.mu / mu_sum for s in servers]
            else:
                total_w = sum(self.weights)
                probs = [w / total_w for w in self.weights]
                
            chosen_idx = rng.choice(num_servers, p=probs)
            return servers[chosen_idx]

        raise NotImplementedError(f"Política {self.policy} não implementada.")

    def reset(self):
        """Reinicializa o estado do balanceador (ex: ponteiro de Round-Robin)."""
        self.rr_index = 0
