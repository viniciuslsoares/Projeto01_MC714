"""
Módulo que define a classe Server (Servidor com fila FCFS e suporte a buffer finito e taxas heterogêneas).
"""

from collections import deque
from typing import Optional
import numpy as np

from src.models import Request


class Server:
    """
    Representa um nó de processamento (servidor) com 1 thread e fila FCFS.
    
    Suporta:
    - Fila ilimitada (padrão do enunciado) ou limitada a K (extensão de buffer finito M/M/1/K).
    - Taxa de atendimento configurável mu (padrão mu = 1.0, ou heterogêneo).
    - Rastreamento preciso do tempo de ocupação pós warm-up para cálculo da utilização U_i.
    """
    
    def __init__(self, server_id: int, mu: float = 1.0, capacity: Optional[int] = None):
        """
        Inicializa o servidor.
        
        Args:
            server_id: Identificador único do servidor (ex: 0, 1, 2).
            mu: Taxa de atendimento (requisições por unidade de tempo).
            capacity: Capacidade máxima do servidor (fila + em execução). None para ilimitado.
        """
        self.id = server_id
        self.mu = mu
        self.capacity = capacity  # K (total no nó = 1 em serviço + fila)
        
        # Estado do servidor
        self.queue: deque[Request] = deque()
        self.current_request: Optional[Request] = None
        self.is_busy: bool = False
        
        # Variáveis para integração do tempo de ocupação (Utilização U_i)
        self.total_busy_time_post_warmup: float = 0.0
        self.last_busy_update_time: float = 0.0
        
        # Contadores estatísticos
        self.total_completed: int = 0
        self.total_dropped: int = 0

    @property
    def total_requests(self) -> int:
        """Retorna o número total de requisições no nó (em fila + em atendimento)."""
        return len(self.queue) + (1 if self.is_busy else 0)

    def can_accept(self) -> bool:
        """Verifica se o servidor pode aceitar mais uma requisição (considerando capacidade K)."""
        if self.capacity is None:
            return True
        return self.total_requests < self.capacity

    def update_busy_time(self, current_time: float, warmup_time: float):
        """
        Atualiza a integral do tempo ocupado pelo servidor, contabilizando apenas
        o período após o tempo de warm-up.
        """
        if current_time <= warmup_time:
            self.last_busy_update_time = max(current_time, self.last_busy_update_time)
            return

        effective_start = max(self.last_busy_update_time, warmup_time)
        if self.is_busy and current_time > effective_start:
            self.total_busy_time_post_warmup += (current_time - effective_start)
            
        self.last_busy_update_time = current_time

    def start_service(self, request: Request, current_time: float, rng: np.random.Generator) -> float:
        """
        Inicia o atendimento de uma requisição.
        
        Args:
            request: A requisição a ser atendida.
            current_time: Instante da simulação em que o serviço inicia.
            rng: Gerador de números aleatórios NumPy para sortear o tempo de serviço.
            
        Returns:
            Duração do tempo de serviço sorteado.
        """
        self.is_busy = True
        self.current_request = request
        request.server_id = self.id
        request.start_service_time = current_time
        
        # Sorteia o tempo de serviço exponencial: E[S] = 1/mu
        duration = rng.exponential(scale=1.0 / self.mu)
        request.service_time = duration
        return duration

    def finish_service(self, current_time: float) -> Optional[Request]:
        """
        Finaliza o atendimento da requisição corrente e libera o servidor.
        
        Args:
            current_time: Instante da conclusão do atendimento.
            
        Returns:
            A requisição finalizada.
        """
        finished = self.current_request
        if finished is not None:
            finished.departure_time = current_time
            self.total_completed += 1
            
        self.current_request = None
        self.is_busy = False
        return finished

    def enqueue(self, request: Request) -> bool:
        """
        Enfileira uma requisição na fila FCFS do servidor.
        
        Returns:
            True se foi aceita, False se foi descartada por estouro de buffer (K).
        """
        if not self.can_accept():
            request.is_dropped = True
            self.total_dropped += 1
            return False
            
        request.server_id = self.id
        self.queue.append(request)
        return True

    def reset(self):
        """Reseta o estado do servidor para uma nova execução/réplica."""
        self.queue.clear()
        self.current_request = None
        self.is_busy = False
        self.total_busy_time_post_warmup = 0.0
        self.last_busy_update_time = 0.0
        self.total_completed = 0
        self.total_dropped = 0
