"""
Modelos de dados e estruturas fundamentais para a simulação de eventos discretos.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict


class EventType(Enum):
    """Tipos de eventos processados pelo escalonador da simulação."""
    ARRIVAL = auto()      # Chegada de uma nova requisição no balanceador
    DEPARTURE = auto()    # Término de serviço de uma requisição em um servidor


@dataclass
class Request:
    """
    Representa uma requisição no sistema distribuído.
    
    Atributos:
        id: Identificador único sequencial da requisição.
        arrival_time: Instante exato de chegada da requisição no balanceador.
        service_time: Duração do atendimento gerada a partir da distribuição exponencial.
        server_id: ID do servidor para o qual a requisição foi alocada (None se descartada).
        start_service_time: Instante em que o servidor começou a processar a requisição.
        departure_time: Instante em que a requisição concluiu o serviço e saiu do sistema.
        is_dropped: Flag que indica se a requisição foi descartada por falta de buffer (ponto extra).
        arrived_after_warmup: Flag que indica se a requisição chegou após o período de warm-up.
    """
    id: int
    arrival_time: float
    service_time: float = 0.0
    server_id: Optional[int] = None
    start_service_time: Optional[float] = None
    departure_time: Optional[float] = None
    is_dropped: bool = False
    arrived_after_warmup: bool = False

    @property
    def response_time(self) -> Optional[float]:
        """Tempo total de resposta (E[R] = saída - chegada)."""
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        return None

    @property
    def queue_time(self) -> Optional[float]:
        """Tempo de espera na fila (E[T_Q] = início do serviço - chegada)."""
        if self.start_service_time is not None:
            return self.start_service_time - self.arrival_time
        return None


@dataclass(order=True)
class Event:
    """
    Representa um evento discreto na fila de prioridade do simulador.
    Ordenado primariamente pelo tempo de ocorrência (time) e secundariamente por priority_id.
    """
    time: float
    priority_id: int
    event_type: EventType = field(compare=False)
    request: Optional[Request] = field(default=None, compare=False)
    server_id: Optional[int] = field(default=None, compare=False)


@dataclass
class SimulationMetrics:
    """
    Estrutura que armazena os resultados e métricas consolidadas de uma execução de simulação.
    """
    policy_name: str
    lambda_rate: float
    seed: int
    duration: float
    warmup_time: float
    
    # Métricas globais pós warm-up
    throughput: float                  # Vazão do sistema X (req/u.t.)
    avg_response_time: float           # E[R] - Tempo médio de resposta
    avg_queue_time: float              # E[T_Q] - Tempo médio em fila
    avg_requests_in_system: float      # E[N] - Número médio de requisições no sistema (integral no tempo)
    
    # Métricas por servidor
    server_utilizations: List[float]   # Utilização U_i de cada servidor
    server_request_counts: List[int]   # Quantidade de requisições atendidas por servidor
    
    # Métricas de contagem e descarte
    total_requests_arrived: int
    requests_after_warmup: int
    completed_requests: int
    dropped_requests: int
    drop_probability: float            # P_perda (para extensões com buffer finito)
