

import argparse
import csv
import random
import time
from pathlib import Path

from AgenteCacho import CATEGORIAS, normalizar, puntuar
from AgenteCachoIA import AgenteCachoIA
from AgenteIA.AgenteJugador import ElEstado


FIELDNAMES = [
    'game_id', 'seed', 'first_agent', 'base_score', 'improved_score',
    'winner', 'base_avg_decision_seconds', 'improved_avg_decision_seconds',
    'base_evaluations', 'improved_evaluations', 'base_decisions',
    'improved_decisions', 'evaluation_version', 'depth',
]


def simular_partida(game_id, seed, first_agent, depth=1):
    """Juega 10 anotaciones; solo el RNG del entorno conoce los dados futuros."""
    rng = random.Random(seed)
    agents = {
        'base': AgenteCachoIA(nombre='BASE', altura=depth, usar_mejorada=False),
        'improved': AgenteCachoIA(nombre='MEJORADA', altura=depth, usar_mejorada=True),
    }
    order = ['base', 'improved'] if first_agent == 'base' else ['improved', 'base']
    scores = {'base': 0, 'improved': 0}
    used = {'base': set(), 'improved': set()}
    statistics = {
        name: {'seconds': 0.0, 'evaluations': 0, 'decisions': 0}
        for name in agents
    }
    player_index = 0
    round_number = 0
    dice = normalizar([rng.randint(1, 6) for _ in range(5)])
    rolls_left = 3

    while round_number < 10:
        agent_name = order[player_index]
        agent = agents[agent_name]
        board = {
            'dados': dice,
            'tiradas_restantes': rolls_left,
            'puntajes': [scores[order[0]], scores[order[1]]],
            'usadas': [set(used[order[0]]), set(used[order[1]])],
            'ronda': round_number,
        }
        # Expectiminimax usa los asientos MAX=0/MIN=1. El torneo alterna el
        # orden de agentes y traduce las categorías usadas a ese orden.
        state = ElEstado(
            jugador=player_index,
            get_utilidad=0,
            tablero=board,
            movidas=None,
        )
        agent.estado = state
        start = time.perf_counter()
        action = agent.elegir_accion(state)
        statistics[agent_name]['seconds'] += time.perf_counter() - start
        statistics[agent_name]['evaluations'] += agent.evaluaciones
        statistics[agent_name]['decisions'] += 1

        if action is None:
            raise RuntimeError(f"{agent_name} no eligió acción en la partida {game_id}")
        kind, parameter = action
        if kind == 'tirar':
            if rolls_left <= 0:
                raise RuntimeError("El agente intentó tirar sin tiradas disponibles")
            rerolled = list(dice)
            for index in parameter:
                rerolled[index] = rng.randint(1, 6)
            dice = normalizar(rerolled)
            rolls_left -= 1
            continue

        if parameter in used[agent_name] or parameter not in CATEGORIAS:
            raise RuntimeError(f"Categoría inválida elegida: {parameter}")
        scores[agent_name] += puntuar(parameter, dice)
        used[agent_name].add(parameter)
        round_number += 1

        if round_number < 10:
            player_index = 1 - player_index
            dice = normalizar([rng.randint(1, 6) for _ in range(5)])
            rolls_left = 3

    winner = (
        'base' if scores['base'] > scores['improved']
        else 'mejorada' if scores['improved'] > scores['base']
        else 'empate'
    )
    return {
        'game_id': game_id,
        'seed': seed,
        'first_agent': first_agent,
        'base_score': scores['base'],
        'improved_score': scores['improved'],
        'winner': winner,
        'base_avg_decision_seconds': (
            statistics['base']['seconds'] / statistics['base']['decisions']
        ),
        'improved_avg_decision_seconds': (
            statistics['improved']['seconds'] / statistics['improved']['decisions']
        ),
        'base_evaluations': statistics['base']['evaluations'],
        'improved_evaluations': statistics['improved']['evaluations'],
        'base_decisions': statistics['base']['decisions'],
        'improved_decisions': statistics['improved']['decisions'],
        'evaluation_version': 'base_vs_mejorada',
        'depth': depth,
    }


def ejecutar_torneo(partidas=100, semilla=2025, profundidad=1, output=None):
    if partidas < 1:
        raise ValueError("El torneo requiere al menos una partida")
    output_path = Path(output) if output else Path(__file__).with_name('resultados_cacho.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for game_index in range(partidas):
        first_agent = 'base' if game_index % 2 == 0 else 'mejorada'
        rows.append(simular_partida(
            game_index + 1,
            semilla + game_index,
            first_agent,
            profundidad,
        ))

    with output_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} partidas registradas en {output_path}")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ejecutar_torneo(args.games, args.seed, args.depth, args.output)
