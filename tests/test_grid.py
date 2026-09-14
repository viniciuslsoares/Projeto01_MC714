"""
Testes do executor de grade — o único ponto de teste do pipeline experimental.

Testa comportamento observável: as linhas de resultado. Não testa a linha de
comando nem a escrita de arquivo (são finas e sem lógica).
"""

import pytest

from analysis.analytical_model import AnalyticalModel
from experiments.grid import GridConfig, execute_grid

POLICIES = ["random", "round_robin", "shortest_queue"]


def small_config(**overrides) -> GridConfig:
    """Grade minúscula: roda em segundos e exercita o executor inteiro."""
    params = dict(
        lambdas=[0.6, 1.8],
        policies=POLICIES,
        replicas=2,
        duration=600.0,
        warmup=100.0,
    )
    params.update(overrides)
    return GridConfig(**params)


def test_uma_linha_por_combinacao():
    results = execute_grid(small_config())

    assert len(results.metric_rows) == 2 * 3 * 2
    chaves = [(r["policy"], r["lambda"], r["replica"]) for r in results.metric_rows]
    assert len(set(chaves)) == len(chaves)


def test_politicas_recebem_as_mesmas_chegadas():
    """
    Assinatura observável das variáveis aleatórias comuns.

    As chegadas são sorteadas num fluxo próprio, independente do roteamento, então
    a MESMA semente tem de produzir exatamente o mesmo número de chegadas nas três
    políticas. Se alguém voltar a usar um gerador único compartilhado, isto falha.
    """
    results = execute_grid(small_config())

    por_replica: dict = {}
    for r in results.metric_rows:
        por_replica.setdefault((r["lambda"], r["replica"]), []).append(r)

    for linhas in por_replica.values():
        chegadas = {r["n_arrived"] for r in linhas}
        assert len(chegadas) == 1, f"chegadas divergiram entre políticas: {chegadas}"

        # X só difere pelo efeito de borda (quem ainda estava em serviço em T_fim),
        # que é de poucas requisições em milhares — daí a tolerância apertada.
        vazoes = [r["X"] for r in linhas]
        assert max(vazoes) == pytest.approx(min(vazoes), rel=0.01)


def test_mesma_configuracao_da_o_mesmo_resultado():
    assert execute_grid(small_config()).metric_rows == execute_grid(small_config()).metric_rows


def test_replicas_diferentes_dao_resultados_diferentes():
    """Guarda contra o erro oposto: uma semente acidentalmente constante."""
    results = execute_grid(small_config())

    ers = [
        r["E_R"] for r in results.metric_rows
        if r["policy"] == "random" and r["lambda"] == 1.8
    ]
    assert len(set(ers)) == len(ers)


def test_politica_aleatoria_bate_com_a_teoria():
    config = small_config(
        lambdas=[1.8], policies=["random"], replicas=5, duration=5000.0, warmup=500.0
    )
    results = execute_grid(config)

    medido = sum(r["E_R"] for r in results.metric_rows) / len(results.metric_rows)
    teorico = AnalyticalModel.mm1_random_policy(1.8)["E_R"]
    assert medido == pytest.approx(teorico, rel=0.10)


def test_trajetoria_e_uma_escada_dentro_da_janela():
    config = small_config(
        lambdas=[1.8], policies=["random"], replicas=1, track_trajectory=True
    )
    results = execute_grid(config)

    tempos = [p["t"] for p in results.trajectory_rows]
    ns = [p["N"] for p in results.trajectory_rows]

    assert tempos == sorted(tempos)
    assert max(tempos) <= config.duration  # a janela é recortada, mesmo com a drenagem
    assert ns[0] == 0  # o estado é gravado ANTES de aplicar o evento
    assert all(abs(b - a) == 1 for a, b in zip(ns, ns[1:]))


def test_grade_sem_trajetoria_nao_grava_trajetoria():
    assert execute_grid(small_config()).trajectory_rows == []


def test_lambda_instavel_satura_a_vazao():
    duration = 5000.0
    config = small_config(
        lambdas=[3.3], policies=["random"], replicas=3,
        duration=duration, warmup=0.0, track_trajectory=True,
    )
    results = execute_grid(config)

    # Chegam 3,3 mas só saem 3*mu = 3,0: a vazão satura na capacidade.
    for r in results.metric_rows:
        assert r["X"] == pytest.approx(3.0, rel=0.05)

    # N cresce como (lambda - 3*mu)*t = 0,3*t (aproximação de fluido).
    # Média das 3 sementes, não uma só: N(t) é um passeio aleatório com deriva e
    # uma semente isolada oscila umas 180 requisições em torno da reta.
    finais = {}
    for p in results.trajectory_rows:
        finais[p["seed"]] = p["N"]  # o último ponto de cada semente sobrescreve
    media_final = sum(finais.values()) / len(finais)
    assert media_final == pytest.approx(0.3 * duration, rel=0.20)


def test_configuracao_invalida_falha_alto():
    with pytest.raises(ValueError):
        execute_grid(small_config(duration=100.0, warmup=500.0))
