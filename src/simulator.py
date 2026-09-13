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
        track_trajectory: bool = False,
        weights: Optional[List[float]] = None
    ):
        """
        Inicializa os parâmetros do simulador.

        Args:
            lambda_rate: Taxa de chegada global do processo de Poisson.
            policy: Política de balanceamento ("random", "round_robin", "shortest_queue", "proportional").
            num_servers: Número de servidores (padrão: 3).
            mu_list: Taxas mu de cada servidor (padrão: [1.0, 1.0, 1.0]).
            capacities: Capacidades K de cada servidor (padrão: [None, None, None]).
            duration: Duração da JANELA DE MEDIÇÃO em unidades de tempo (T_fim, padrão: 5000.0).
                Nenhuma requisição chega depois de T_fim, mas as que já estão no sistema são
                acompanhadas até terminar (fase de drenagem) — ver run().
            warmup_time: Período de aquecimento descartado da coleta de métricas (padrão: 500.0).
            seed: Semente pseudoaleatória para reprodutibilidade.
            track_trajectory: Se True, armazena a trajetória (t, N(t)) para gráficos temporais.
            weights: Pesos de roteamento explícitos para a política proporcional. Se None, o
                balanceador deduz p_i = mu_i / soma(mu_j). Necessário para comparar pesos
                arbitrários (ex.: 1/3 uniforme) contra os ótimos no ponto extra heterogêneo.
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
        self.weights = weights
        self.load_balancer = LoadBalancer(policy=self.policy_name, weights=weights)

        # Três geradores pseudoaleatórios INDEPENDENTES derivados da mesma semente.
        # Um gerador único compartilhado faria as três finalidades consumirem a mesma
        # sequência, de modo que a quantidade de sorteios gastos em roteamento (que varia
        # por política: 1 por requisição na aleatória, 0 no round-robin) deslocaria as
        # chegadas e os serviços. Com fluxos separados, "semente s" significa exatamente
        # as mesmas chegadas e as mesmas demandas de serviço em todas as políticas
        # (variáveis aleatórias comuns): a diferença medida em E[R] vem só da política.
        seed_sequence = np.random.SeedSequence(seed)
        self.rng_arrival, self.rng_service, self.rng_route = (
            np.random.default_rng(child) for child in seed_sequence.spawn(3)
        )

        # Fila de eventos (Priority Queue com heapq) e contador de prioridade
        self.event_queue: List[Event] = []
        self.event_counter: int = 0

        # Relógio da simulação e rastreamento de estado
        self.current_time: float = 0.0
        self.last_event_time: float = 0.0
        self.request_counter: int = 0

        # Acumuladores para cálculo de E[N] via integral temporal
        self.integral_N: float = 0.0

        # Contador explícito de requisições ACEITAS na janela de medição.
        # Não usar "concluídas" como sinônimo de "aceitas": são populações diferentes
        # (uma aceita pode ainda estar no sistema) e confundi-las falseia o P_perda.
        self.accepted_post_warmup: int = 0

        # Partidas ocorridas DENTRO da janela de medição — numerador da vazão X.
        # Contar "admitidas na janela, acompanhadas até concluir" daria X = lambda sempre,
        # inclusive em lambda=3.3, onde X reportado (3,30) excederia a capacidade 3*mu = 3,0.
        # Medir partidas no intervalo faz X saturar corretamente em 3*mu (item f).
        self.completed_in_window: int = 0

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

        As médias temporais (E[N] e U_i) valem sobre a janela de medição fechada
        [warmup_time, duration]. O intervalo é portanto recortado nas duas pontas: o
        warm-up é descartado no início e a fase de drenagem (t > duration, quando já não
        há chegadas e N(t) decai para zero) é descartada no fim. Sem o recorte superior a
        drenagem acrescentaria área com N pequeno e puxaria E[N] artificialmente para baixo.

        Já E[R] NÃO é média temporal e não sofre esse recorte: cada requisição que chegou
        na janela é acompanhada até terminar, ainda que termine depois de duration.
        """
        if new_time <= self.last_event_time:
            return

        # Atualiza a integral de ocupação de cada servidor individualmente
        for server in self.servers:
            server.update_busy_time(new_time, self.warmup_time, measurement_end=self.duration)

        # Atualiza a integral do número total de requisições no sistema N(t)
        if new_time > self.warmup_time:
            end = min(new_time, self.duration)
            effective_start = max(self.last_event_time, self.warmup_time)
            if end > effective_start:
                self.integral_N += self.total_requests_in_system * (end - effective_start)

        # A trajetória N(t) também se limita à janela de medição
        if self.track_trajectory and new_time <= self.duration:
            self.trajectory_times.append(new_time)
            self.trajectory_n.append(self.total_requests_in_system)

        self.last_event_time = new_time

    def _create_request(self, arrival_time: float) -> Request:
        """
        Cria uma requisição, já sorteando sua demanda de serviço normalizada ~ Exp(1).

        O sorteio acontece aqui — na chegada — e não no início do atendimento, porque as
        requisições são criadas em ordem de chegada, uma ordem que NÃO depende da política.
        Se a demanda fosse sorteada em start_service, a ordem de consumo do fluxo dependeria
        do roteamento (a mesma requisição pode ser a 1ª atendida sob JSQ e a 3ª sob RR), e o
        pareamento entre políticas se desfaria. A conversão em tempo de serviço acontece em
        Server.start_service, dividindo por mu.
        """
        self.request_counter += 1
        return Request(
            id=self.request_counter,
            arrival_time=arrival_time,
            service_base=float(self.rng_service.exponential(1.0)),
            arrived_after_warmup=(arrival_time >= self.warmup_time)
        )

    def _handle_arrival(self, event: Event):
        """Processa a chegada de uma nova requisição e agenda a próxima chegada."""
        req = event.request
        assert req is not None

        # 1. Seleciona o servidor de destino através do balanceador
        chosen_server = self.load_balancer.select_server(self.servers, self.rng_route)

        # 2. Se o servidor estiver ocioso, inicia o atendimento imediatamente
        if not chosen_server.is_busy:
            service_duration = chosen_server.start_service(req, self.current_time)
            departure_time = self.current_time + service_duration
            self._schedule_event(
                event_time=departure_time,
                event_type=EventType.DEPARTURE,
                request=req,
                server_id=chosen_server.id
            )
            accepted = True
        else:
            # Caso contrário, tenta enfileirar no servidor
            accepted = chosen_server.enqueue(req)
            if not accepted:
                self.dropped_requests.append(req)

        # Contabiliza a admissão na janela de medição no instante da DECISÃO, não da conclusão
        if accepted and req.arrived_after_warmup:
            self.accepted_post_warmup += 1

        # 3. Agenda a próxima chegada do Processo de Poisson (Exp(lambda))
        next_inter_arrival = self.rng_arrival.exponential(scale=1.0 / self.lambda_rate)
        next_arrival_time = self.current_time + next_inter_arrival

        # As chegadas cessam em T_fim: é isto que delimita a janela de medição.
        # Os eventos já agendados continuam a ser processados (ver run()).
        if next_arrival_time <= self.duration:
            self._schedule_event(
                event_time=next_arrival_time,
                event_type=EventType.ARRIVAL,
                request=self._create_request(next_arrival_time)
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
            # Vazão X conta partidas ocorridas na janela, independentemente de quando a
            # requisição chegou (as da fase de drenagem, t > duration, ficam de fora)
            if self.warmup_time < self.current_time <= self.duration:
                self.completed_in_window += 1

        # 2. Se houver requisições aguardando na fila, inicia o atendimento da próxima (FCFS)
        if len(server.queue) > 0:
            next_req = server.queue.popleft()
            service_duration = server.start_service(next_req, self.current_time)
            departure_time = self.current_time + service_duration
            self._schedule_event(
                event_time=departure_time,
                event_type=EventType.DEPARTURE,
                request=next_req,
                server_id=server.id
            )

    def run(self) -> SimulationMetrics:
        """
        Executa a simulação: chegadas no intervalo (0, duration] e depois drenagem.

        "Parar de gerar chegadas" e "parar de processar eventos" são coisas distintas, e
        somente a primeira delimita a janela de medição. As chegadas cessam em T_fim
        (_handle_arrival não agenda nada além disso), mas o laço de eventos continua até
        esvaziar, atendendo quem já estava no sistema. Interromper o laço em T_fim faria as
        requisições ainda em curso nunca receberem departure_time e, como só entram na média
        de E[R] as que têm response_time, elas seriam silenciosamente excluídas — justamente
        as mais lentas. Isso é viés, não ruído: em lambda=2,7 há E[N]=27 requisições dentro do
        sistema em T_fim, e o E[R] medido sairia sistematicamente abaixo do teórico 10,0, com
        erro crescente em lambda — deformando exatamente a curva E[R] x lambda.

        A drenagem é curta: em lambda=2,7 são ~27 requisições em 3 servidores a mu=1,
        cerca de 9 u.t. sobre 5000. As médias temporais continuam recortadas em T_fim
        (ver _update_integrals).

        Returns:
            Instância de SimulationMetrics contendo os resultados consolidados.
        """
        # Agenda a primeira chegada de requisição em t ~ Exp(lambda)
        first_arrival_time = self.rng_arrival.exponential(scale=1.0 / self.lambda_rate)
        if first_arrival_time <= self.duration:
            self._schedule_event(
                event_time=first_arrival_time,
                event_type=EventType.ARRIVAL,
                request=self._create_request(first_arrival_time)
            )

        # Loop principal de processamento de eventos discretos.
        # Roda até esvaziar: como não há novas chegadas após T_fim, o término é garantido.
        while self.event_queue:
            event = heapq.heappop(self.event_queue)

            # Avança o relógio da simulação e atualiza integrais de tempo
            self.current_time = event.time
            self._update_integrals(self.current_time)

            # Despacha o evento para o handler apropriado
            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival(event)
            elif event.event_type == EventType.DEPARTURE:
                self._handle_departure(event)

        # Fecha a integração até T_fim caso o último evento tenha ocorrido antes dele
        # (sistema esvaziou cedo, ou nenhuma chegada foi gerada)
        if self.last_event_time < self.duration:
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

        # Invariante da drenagem: toda requisição admitida na janela precisa ter concluído.
        # Se este assert falhar, o laço de eventos foi interrompido antes de esvaziar e o
        # E[R] estaria enviesado para baixo por excluir as requisições mais lentas.
        assert len(post_warmup_completed) == self.accepted_post_warmup, (
            f"{self.accepted_post_warmup - len(post_warmup_completed)} requisições admitidas "
            f"na janela de medição não concluíram: a simulação não drenou por completo."
        )

        # 1. Tempo Médio de Resposta E[R] e Tempo em Fila E[T_Q]
        if post_warmup_completed:
            avg_resp = float(np.mean([r.response_time for r in post_warmup_completed]))
            avg_queue = float(np.mean([r.queue_time for r in post_warmup_completed]))
        else:
            avg_resp = 0.0
            avg_queue = 0.0

        # 2. Número Médio de Requisições E[N] (Integral no tempo / tempo efetivo)
        avg_n = self.integral_N / effective_duration

        # 3. Vazão X: partidas ocorridas DENTRO da janela de medição / duração da janela.
        # É a definição do enunciado ("requisições concluídas pós warm-up / delta_T") e a
        # fisicamente correta: X é uma taxa de saída medida no intervalo, logo satura em
        # 3*mu quando lambda > 3*mu (item f). Note que a população aqui NÃO é a mesma de
        # E[R] (que segue as chegadas da janela até concluírem, mesmo além de T_fim) — em
        # regime estacionário as duas taxas coincidem, e é justamente por serem medidas
        # independentes que a Lei de Little (E[N] = X * E[R]) é uma checagem honesta.
        throughput = self.completed_in_window / effective_duration

        # 4. Utilização de cada servidor U_i.
        # Sem clamp deliberadamente: U_i > 1 é fisicamente impossível e denunciaria erro na
        # contabilidade do tempo ocupado. Um min(1.0, ...) transformaria um 1,03 revelador
        # num 1,00 plausível e esconderia o bug.
        server_utils = []
        for s in self.servers:
            u = s.total_busy_time_post_warmup / effective_duration
            assert u <= 1.0 + 1e-9, (
                f"Utilização do servidor {s.id} = {u:.6f} > 1: erro na contabilidade do "
                f"tempo ocupado (busy={s.total_busy_time_post_warmup}, janela={effective_duration})"
            )
            server_utils.append(u)
        server_counts = [s.total_completed for s in self.servers]

        # 5. Probabilidade de Perda (Ponto Extra Buffer Finito).
        # O denominador são as CHEGADAS na janela = aceitas + descartadas, com as aceitas
        # vindas de um contador explícito. Usar "concluídas" no lugar de "aceitas" subestima
        # o denominador (uma aceita pode ainda estar em serviço) e infla o P_perda.
        total_post_warmup_arrivals = self.accepted_post_warmup + len(post_warmup_dropped)
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
