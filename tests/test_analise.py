"""
Testes da agregação — o único ponto de teste da etapa de análise.

Testa comportamento observável: o que a agregação devolve dado um conjunto de
linhas. Não testa leitura de CSV, escrita de arquivo nem figura (são finas, sem
lógica, e figura errada se vê olhando).

As linhas são construídas à mão com números cuja resposta certa é calculável no
papel, então quase todo limiar aqui é exato.
"""

import pytest

from analysis.analise import aggregate
from analysis.analytical_model import AnalyticalModel


def make_row(
    policy: str = "random",
    lam: float = 1.8,
    replica: int = 1,
    X: float = 1.8,
    E_R: float = 2.5,
    E_TQ: float = 1.5,
    E_N: float = 4.5,
    U: tuple = (0.6, 0.6, 0.6),
    mu: tuple = (1.0, 1.0, 1.0),
    duration: float = 5000.0,
    n_arrived: int = 9000,
) -> dict:
    """Uma linha de réplica, no mesmo formato que o CSV do grid."""
    return {
        "policy": policy, "lambda": lam, "replica": replica, "seed": 1000 + replica,
        "duration": duration, "warmup": 500.0,
        "mu0": mu[0], "mu1": mu[1], "mu2": mu[2],
        "X": X, "E_R": E_R, "E_TQ": E_TQ, "E_N": E_N,
        "U0": U[0], "U1": U[1], "U2": U[2],
        "n_arrived": n_arrived, "n_after_warmup": n_arrived, "n_completed": n_arrived,
        "drop_prob": 0.0,
    }


def rows_with_e_r(values, **overrides):
    """Uma réplica por valor de E[R], com o resto igual."""
    return [
        make_row(replica=i, E_R=v, **overrides)
        for i, v in enumerate(values, start=1)
    ]


def only(results, policy=None, lam=None):
    """A única configuração que casa com o filtro."""
    matches = [
        c for c in results.configs
        if (policy is None or c.policy == policy) and (lam is None or c.lambda_rate == lam)
    ]
    assert len(matches) == 1, f"esperava 1 configuração, achei {len(matches)}"
    return matches[0]


def test_media_e_semi_ic_com_t_de_student():
    """t(1) = 12,7062 para 2 réplicas e t(9) = 2,2622 para 10."""
    duas = aggregate(rows_with_e_r([2.0, 3.0]))
    config = only(duas)
    assert config.replicas == 2
    assert config.E_R_mean == pytest.approx(2.5)
    assert config.E_R_ci95 == pytest.approx(6.35310, rel=1e-4)  # 12,7062 * 0,5

    dez = aggregate(rows_with_e_r([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
    config = only(dez)
    assert config.replicas == 10
    assert config.E_R_mean == pytest.approx(5.5)
    assert config.E_R_ci95 == pytest.approx(2.16585, rel=1e-4)  # 2,2622 * 0,957427


def test_replica_unica_tem_semi_ic_zero():
    config = only(aggregate(rows_with_e_r([2.5])))

    assert config.replicas == 1
    assert config.E_R_mean == pytest.approx(2.5)
    assert config.E_R_ci95 == 0.0


def test_mu_diferente_nao_se_mistura():
    rows = (
        rows_with_e_r([2.0, 2.0])
        + rows_with_e_r([3.0, 3.0], mu=(1.5, 1.0, 0.5))
    )
    results = aggregate(rows)

    assert len(results.configs) == 2
    por_mu = {c.mu: c.E_R_mean for c in results.configs}
    assert por_mu[(1.0, 1.0, 1.0)] == pytest.approx(2.0)
    assert por_mu[(1.5, 1.0, 0.5)] == pytest.approx(3.0)


def test_duracao_diferente_nao_se_mistura():
    rows = (
        rows_with_e_r([2.0, 2.0])
        + rows_with_e_r([3.0, 3.0], duration=20000.0)
    )
    results = aggregate(rows)

    assert len(results.configs) == 2
    por_duracao = {c.duration: c.E_R_mean for c in results.configs}
    assert por_duracao[5000.0] == pytest.approx(2.0)
    assert por_duracao[20000.0] == pytest.approx(3.0)


def test_mesma_politica_em_lambdas_diferentes_da_duas_configuracoes():
    rows = rows_with_e_r([1.2, 1.3], lam=0.6) + rows_with_e_r([2.4, 2.6], lam=1.8)
    results = aggregate(rows)

    assert len(results.configs) == 2
    assert only(results, lam=0.6).E_R_mean == pytest.approx(1.25)
    assert only(results, lam=1.8).E_R_mean == pytest.approx(2.5)


def test_contagem_desigual_de_replicas_falha_alto():
    """
    Um IC saído de 9 réplicas em silêncio é pior que uma exceção: o arquivo
    incompleto parece completo.
    """
    rows = rows_with_e_r([2.0, 2.0], lam=0.6) + rows_with_e_r([3.0], lam=1.8)

    with pytest.raises(ValueError, match="réplicas"):
        aggregate(rows)


def test_ganho_percentual_contra_a_aleatoria():
    rows = (
        rows_with_e_r([2.5, 2.5], policy="random")
        + rows_with_e_r([2.0, 2.0], policy="round_robin")
    )
    results = aggregate(rows)

    # (2,5 - 2,0) / 2,5 = 20%: positivo quando a política é MELHOR que a aleatória.
    assert only(results, policy="round_robin").ganho_vs_random_pct == pytest.approx(20.0)
    assert only(results, policy="random").ganho_vs_random_pct == pytest.approx(0.0)


def test_ganho_ausente_sem_politica_aleatoria():
    results = aggregate(rows_with_e_r([2.0, 2.0], policy="proportional"))

    assert only(results).ganho_vs_random_pct is None


def test_erro_de_little_usa_colunas_independentes():
    """
    E[N] vem da integral no tempo e X das partidas: são caminhos independentes.
    Se alguém calcular E[N] = X * E[R], o erro passa a ser zero sempre e este
    teste falha.
    """
    config = only(aggregate([
        make_row(replica=1, X=2.0, E_R=2.0, E_N=5.0),
        make_row(replica=2, X=2.0, E_R=2.0, E_N=5.0),
    ]))

    assert config.little_product == pytest.approx(4.0)
    assert config.little_error_pct == pytest.approx(20.0)  # |5 - 4| / 5


def test_teoria_com_mu_homogeneo_e_a_formula_da_aleatoria():
    config = only(aggregate(rows_with_e_r([2.5, 2.5], lam=1.8)))

    assert config.E_R_teorico_mm1 == pytest.approx(1.0 / (1.0 - 1.8 / 3))  # 2,5
    assert config.E_R_teorico_e3m1 == pytest.approx(1.8134, rel=1e-4)
    assert config.desvio_teoria_pct == pytest.approx(0.0)


def test_teoria_ausente_com_mu_heterogeneo():
    """Referência ausente é melhor que referência errada."""
    config = only(aggregate(rows_with_e_r([2.5, 2.5], mu=(1.5, 1.0, 0.5))))

    assert config.E_R_teorico_mm1 is None
    assert config.E_R_teorico_e3m1 is None
    assert config.desvio_teoria_pct is None


def test_formula_do_round_robin_em_valor_conhecido():
    """Ponto fixo da E_3/M/1 em lambda = 1,8 e mu = 1."""
    res = AnalyticalModel.round_robin_e3m1(1.8)

    assert res["sigma"] == pytest.approx(0.448549, rel=1e-5)
    assert res["E_R"] == pytest.approx(1.8134, rel=1e-4)
    assert AnalyticalModel.round_robin_e3m1(3.3)["is_stable"] is False


def test_verificacoes_nao_interrompem():
    """Violar um limiar de aviso não pode custar os resultados agregados."""
    results = aggregate([
        make_row(replica=1, X=2.0, E_R=2.0, E_N=5.0),
        make_row(replica=2, X=2.0, E_R=2.0, E_N=5.0),
    ])

    little = [c for c in results.checks if "Little" in c.name]
    assert len(little) == 1
    assert little[0].ok is False
    assert len(results.configs) == 1
    assert results.configs[0].E_R_mean == pytest.approx(2.0)
