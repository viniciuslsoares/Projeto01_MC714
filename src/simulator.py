"""
Módulo do Simulador de Eventos Discretos (DES) para o Balanceador de Carga.
"""

import heapq
from typing import List, Optional, Tuple
import numpy as np

from src.models import Event, EventType, Request, SimulationMetrics
from src.server import Server
from src.load_balancer import LoadBalancer, LoadBalancerPolicy


class Simulator:
    """
    Simulador de Eventos Discretos de Alta Performance.
    
    Gerencia a linha do tempo, a fila de eventos priorizada, a amostragem de variáveis aleatórias
    (Poisson e Exponencial) e o cálculo rigoroso das integrais de tempo e métricas de desempenho.
    """

    def __init__(
        self,
        lambda_rate: float,
        policy: str = LoadBalancerPolicy.RANDOM,
        num_servers: int = 3,
        mu_list: Optional[List[float]] = None,
        capacities: Optional[List[Optional[int]]] = None,
        duration: float = 5000.0,
        warmup_time: float = 500.0,
        seed: Optional[int] = 42,
        track_trajectory: bool = False
    ):
        """
        Inicializa os parâmetros do simulador.
        
        Args:
            lambda_rate: Taxa de chegada global do processo de Poisson.
            policy: Política de balanceamento ("random", "round_robin", "shortest_queue", "proportional").
            num_servers: Número de servidores (padrão: 3).
            mu_list: Taxas mu de cada servidor (padrão: [1.0, 1.0, 1.0]).
            capacities: Capacidades K de cada servidor (padrão: [None, None, None]).
            duration: Duração total da simulação em unidades de tempo (padrão: 5000.0).
            warmup_time: Período de aquecimento descartado da coleta de métricas (padrão: 500.0).
            seed: Semente pseudoaleatória para reprodutibilidade.
            track_trajectory: Se True, armazena a trajetória (t, N(t)) para gráficos temporais.
        """
        self.lambda_rate = lambda_rate
        self.policy_name = policy
        self.num_servers = num_servers
        self.duration = duration
        self.warmup_time = warmup_time
        self.seed = seed
        self.track_trajectory = track_trajectory

        # Configura taxas mu
        if mu_list is None:
            self.mu_list = [1.0] * num_servers
        else:
            self.mu_list = mu_list

        # Configura capacidades de buffer K
        if capacities is None:
            self.capacities = [None] * num_servers
        else:
            self.capacities = capacities

        # Instancia Servidores e Balanceador
        self.servers: List[Server] = [
            Server(server_id=i, mu=self.mu_list[i], capacity=self.capacities[i])
            for i in range(num_servers)
        ]
        self.load_balancer = LoadBalancer(policy=self.policy_name)

        # Gerador de números pseudoaleatórios NumPy
        self.rng = np.random.default_rng(seed)

        # Fila de eventos (Priority Queue com heapq) e contador de prioridade
        self.event_queue: List[Event] = []
        self.event_counter: int = 0

        # Relógio da simulação e rastreamento de estado
        self.current_time: float = 0.0
        self.last_event_time: float = 0.0
        self.request_counter: int = 0

        # Acumuladores para cálculo de E[N] via integral temporal
        self.integral_N: float = 0.0
        
        # Histórico de requisições e trajetória
        self.completed_requests: List[Request] = []
        self.dropped_requests: List[Request] = []
        self.trajectory_times: List[float] = []
        self.trajectory_n: List[int] = []

    def _schedule_event(
        self,
        event_time: float,
        event_type: EventType,
        request: Optional[Request] = None,
        server_id: Optional[int] = None
    ):
        """Insere um novo evento ordenado na fila de prioridade."""
        self.event_counter += 1
        event = Event(
            time=event_time,
            priority_id=self.event_counter,
            event_type=event_type,
            request=request,
            server_id=server_id
        )
        heapq.heappush(self.event_queue, event)

    @property
    def total_requests_in_system(self) -> int:
        """Soma o número de requisições em todos os servidores (filas + atendimentos)."""
        return sum(s.total_requests for s in self.servers)

    def _update_integrals(self, new_time: float):
        """
        Atualiza a integral no tempo para E[N] e o tempo de ocupação de cada servidor.
        Garante que apenas o intervalo após warmup_time contribua para as médias.
        """
        if new_time <= self.last_event_time:
            return

        # Atualiza a integral de ocupação de cada servidor individualmente
        for server in self.servers:
            server.update_busy_time(new_time, self.warmup_time)

        # Atualiza a integral do número total de requisições no sistema N(t)
        if new_time > self.warmup_time:
            effective_start = max(self.last_event_time, self.warmup_time)
            dt = new_time - effective_start
            if dt > 0:
                self.integral_N += self.total_requests_in_system * dt

        if self.track_trajectory:
            self.trajectory_times.append(new_time)
            self.trajectory_n.append(self.total_requests_in_system)

        self.last_event_time = new_time

    def _handle_arrival(self, event: Event):
        """Processa a chegada de uma nova requisição e agenda a próxima chegada."""
        req = event.request
        assert req is not None

        # 1. Seleciona o servidor de destino através do balanceador
        chosen_server = self.load_balancer.select_server(self.servers, self.rng)

        # 2. Se o servidor estiver ocioso, inicia o atendimento imediatamente
        if not chosen_server.is_busy:
            service_duration = chosen_server.start_service(req, self.current_time, self.rng)
            departure_time = self.current_time + service_duration
            self._schedule_event(
                event_time=departure_time,
                event_type=EventType.DEPARTURE,
                request=req,
                server_id=chosen_server.id
            )
        else:
            # Caso contrário, tenta enfileirar no servidor
            accepted = chosen_server.enqueue(req)
            if not accepted:
                self.dropped_requests.append(req)

        # 3. Agenda a próxima chegada do Processo de Poisson (Exp(lambda))
        next_inter_arrival = self.rng.exponential(scale=1.0 / self.lambda_rate)
        next_arrival_time = self.current_time + next_inter_arrival

        if next_arrival_time <= self.duration:
            self.request_counter += 1
            next_req = Request(
                id=self.request_counter,
                arrival_time=next_arrival_time,
                arrived_after_warmup=(next_arrival_time >= self.warmup_time)
            )
            self._schedule_event(
                event_time=next_arrival_time,
                event_type=EventType.ARRIVAL,
                request=next_req
            )

    def _handle_departure(self, event: Event):
        """Processa o término de serviço de uma requisição em um servidor."""
        server_id = event.server_id
        assert server_id is not None
        server = self.servers[server_id]

        # 1. Finaliza a requisição atual
        finished_req = server.finish_service(self.current_time)
        if finished_req is not None:
            self.completed_requests.append(finished_req)

        # 2. Se houver requisições aguardando na fila, inicia o atendimento da próxima (FCFS)
        if len(server.queue) > 0:
            next_req = server.queue.popleft()
            service_duration = server.start_service(next_req, self.current_time, self.rng)
            departure_time = self.current_time + service_duration
            self._schedule_event(
                event_time=departure_time,
                event_type=EventType.DEPARTURE,
                request=next_req,
                server_id=server.id
            )

    def run(self) -> SimulationMetrics:
        """
        Executa a simulação completa do instante 0 até self.duration.
        
        Returns:
            Instância de SimulationMetrics contendo os resultados consolidados.
        """
        # Agenda a primeira chegada de requisição em t ~ Exp(lambda)
        first_arrival_time = self.rng.exponential(scale=1.0 / self.lambda_rate)
        if first_arrival_time <= self.duration:
            self.request_counter += 1
            first_req = Request(
                id=self.request_counter,
                arrival_time=first_arrival_time,
                arrived_after_warmup=(first_arrival_time >= self.warmup_time)
            )
            self._schedule_event(
                event_time=first_arrival_time,
                event_type=EventType.ARRIVAL,
                request=first_req
            )

        # Loop principal de processamento de eventos discretos
        while self.event_queue:
            event = heapq.heappop(self.event_queue)

            if event.time > self.duration:
                # Atualiza integrais até o fim exato da simulação e encerra
                self._update_integrals(self.duration)
                break

            # Avança o relógio da simulação e atualiza integrais de tempo
            self.current_time = event.time
            self._update_integrals(self.current_time)

            # Despacha o evento para o handler apropriado
            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival(event)
            elif event.event_type == EventType.DEPARTURE:
                self._handle_departure(event)

        # Garante integração até self.duration caso a fila de eventos tenha esvaziado antes
        if self.current_time < self.duration:
            self._update_integrals(self.duration)

        return self._compute_metrics()

    def _compute_metrics(self) -> SimulationMetrics:
        """
        Consolida e calcula todas as métricas no regime estacionário pós warm-up.
        """
        effective_duration = self.duration - self.warmup_time
        if effective_duration <= 0:
            raise ValueError("Duração efetiva pós warm-up deve ser maior que zero.")

        # Filtra requisições que CHEGARAM após o warm-up e foram concluídas
        post_warmup_completed = [
            r for r in self.completed_requests
            if r.arrived_after_warmup and r.response_time is not None
        ]
        
        post_warmup_dropped = [
            r for r in self.dropped_requests
            if r.arrived_after_warmup
        ]

        # 1. Tempo Médio de Resposta E[R] e Tempo em Fila E[T_Q]
        if post_warmup_completed:
            avg_resp = float(np.mean([r.response_time for r in post_warmup_completed]))
            avg_queue = float(np.mean([r.queue_time for r in post_warmup_completed]))
        else:
            avg_resp = 0.0
            avg_queue = 0.0

        # 2. Número Médio de Requisições E[N] (Integral no tempo / tempo efetivo)
        avg_n = self.integral_N / effective_duration

        # 3. Vazão X (Requisições concluídas pós warm-up / tempo efetivo)
        throughput = len(post_warmup_completed) / effective_duration

        # 4. Utilização de cada servidor U_i
        server_utils = [
            min(1.0, s.total_busy_time_post_warmup / effective_duration)
            for s in self.servers
        ]
        server_counts = [s.total_completed for s in self.servers]

        # 5. Probabilidade de Perda (Ponto Extra Buffer Finito)
        total_post_warmup_arrivals = len(post_warmup_completed) + len(post_warmup_dropped)
        if total_post_warmup_arrivals > 0:
            drop_prob = len(post_warmup_dropped) / total_post_warmup_arrivals
        else:
            drop_prob = 0.0

        return SimulationMetrics(
            policy_name=self.policy_name,
            lambda_rate=self.lambda_rate,
            seed=self.seed if self.seed is not None else 0,
            duration=self.duration,
            warmup_time=self.warmup_time,
            throughput=throughput,
            avg_response_time=avg_resp,
            avg_queue_time=avg_queue,
            avg_requests_in_system=avg_n,
            server_utilizations=server_utils,
            server_request_counts=server_counts,
            total_requests_arrived=self.request_counter,
            requests_after_warmup=total_post_warmup_arrivals,
            completed_requests=len(post_warmup_completed),
            dropped_requests=len(post_warmup_dropped),
            drop_probability=drop_prob
        )

    def get_trajectory(self) -> Tuple[List[float], List[int]]:
        """Retorna as séries temporais (tempos, N(t)) gravadas durante a simulação."""
        return self.trajectory_times, self.trajectory_n
