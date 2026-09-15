"""
Testes das políticas de despacho — o único ponto de teste do balanceador.

Testa comportamento observável: para qual servidor cada política manda, dado o
estado dos servidores. Um teste por política, mais um para o empate do JSQ.
"""

import numpy as np
import pytest

from src.load_balancer import LoadBalancer, LoadBalancerPolicy
from src.models import Request
from src.server import Server

DECISOES = 3000


def servidores(ocupacoes=(0, 0, 0), mu=(1.0, 1.0, 1.0)):
    """Três servidores com a ocupação e as taxas pedidas."""
    lista = []
    for i, (n, mu_i) in enumerate(zip(ocupacoes, mu)):
        servidor = Server(server_id=i, mu=mu_i)
        for _ in range(n):
            servidor.enqueue(Request(id=0, arrival_time=0.0))
        lista.append(servidor)
    return lista


def contagem(policy, lista):
    """Quantas vezes cada servidor foi escolhido em DECISOES decisões."""
    balanceador = LoadBalancer(policy=policy)
    rng = np.random.default_rng(1001)
    contas = [0, 0, 0]
    for _ in range(DECISOES):
        contas[balanceador.select_server(lista, rng).id] += 1
    return contas


def test_round_robin_percorre_em_ciclo():
    """Cíclica e cega ao estado: o servidor 1 está livre e não é a vez dele."""
    balanceador = LoadBalancer(policy=LoadBalancerPolicy.ROUND_ROBIN)
    rng = np.random.default_rng(1001)
    lista = servidores(ocupacoes=(2, 0, 1))

    escolhidos = [balanceador.select_server(lista, rng).id for _ in range(4)]
    assert escolhidos == [0, 1, 2, 0]


def test_fila_mais_curta_escolhe_o_menos_ocupado():
    contas = contagem(LoadBalancerPolicy.SHORTEST_QUEUE, servidores(ocupacoes=(2, 0, 1)))

    assert contas == [0, DECISOES, 0]


def test_fila_mais_curta_sorteia_o_empate():
    """Sem sorteio, o empate cairia sempre no servidor 0."""
    contas = contagem(LoadBalancerPolicy.SHORTEST_QUEUE, servidores(ocupacoes=(1, 1, 1)))

    for conta in contas:
        assert conta == pytest.approx(DECISOES / 3, rel=0.1)


def test_aleatoria_usa_os_tres_com_a_mesma_frequencia():
    contas = contagem(LoadBalancerPolicy.RANDOM, servidores())

    for conta in contas:
        assert conta == pytest.approx(DECISOES / 3, rel=0.1)


def test_proporcional_segue_as_taxas_mu():
    """p_i = mu_i / soma(mu) = [0,5; 0,333; 0,167]."""
    contas = contagem(LoadBalancerPolicy.PROPORTIONAL, servidores(mu=(1.5, 1.0, 0.5)))

    assert contas[0] == pytest.approx(DECISOES / 2, rel=0.1)
    assert contas[1] == pytest.approx(DECISOES / 3, rel=0.1)
    assert contas[2] == pytest.approx(DECISOES / 6, rel=0.1)
