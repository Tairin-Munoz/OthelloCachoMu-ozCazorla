"""Torneo BASE contra POSICIONAL con colores alternados y métricas Alpha-Beta."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from AgenteIA.AgenteJugador import ElEstado
from AgentesEspecificos.AgenteOthello import AgenteOthello


FIELDNAMES = [
    'game_id', 'agente_a', 'evaluacion_a', 'profundidad_a',
    'agente_b', 'evaluacion_b', 'profundidad_b', 'black_agent',
    'resultado_a', 'fichas_a', 'fichas_b', 'diferencia_fichas_a',
    'tiempo_promedio_mov_a', 'tiempo_promedio_mov_b',
    'nodos_evaluados_a', 'nodos_evaluados_b',
    'nodos_por_segundo_a', 'nodos_por_segundo_b',
]


def inicializar_tablero():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = 2
    board[3, 4] = board[4, 3] = 1
    return board


def jugar_partida(game_id, profundidad, agente_a_color):
    agents = {
        'base': AgenteOthello(profundidad, 'base'),
        'posicional': AgenteOthello(profundidad, 'posicional'),
    }
    colors = {
        'base': agente_a_color,
        'posicional': 3 - agente_a_color,
    }
    board = inicializar_tablero()
    current_player = 1
    stats = {
        name: {'seconds': 0.0, 'nodes': 0, 'moves': 0}
        for name in agents
    }
    passes_in_a_row = 0

    while True:
        agent_name = next(name for name, color in colors.items() if color == current_player)
        agent = agents[agent_name]
        legal = agent._get_valid_moves(board, current_player)
        if not legal:
            passes_in_a_row += 1
            if passes_in_a_row == 2 or not np.any(board == 0):
                break
            current_player = 3 - current_player
            continue

        passes_in_a_row = 0
        state = ElEstado(
            jugador=current_player,
            tablero=board,
            movidas=legal,
            get_utilidad=0,
        )
        start = time.perf_counter()
        move = agent.podaAlphaBeta_eval(state)
        stats[agent_name]['seconds'] += time.perf_counter() - start
        stats[agent_name]['nodes'] += agent.nodos_evaluados
        stats[agent_name]['moves'] += 1
        if move not in legal:
            raise RuntimeError(f"{agent_name} produced illegal move {move}")
        board = agent.getResultado(state, move).tablero
        current_player = 3 - current_player

    base_color = colors['base']
    positional_color = colors['posicional']
    base_score = int(np.sum(board == base_color))
    positional_score = int(np.sum(board == positional_color))
    result = (
        'victoria' if base_score > positional_score
        else 'derrota' if base_score < positional_score
        else 'empate'
    )
    base_stats = stats['base']
    positional_stats = stats['posicional']
    base_avg = base_stats['seconds'] / base_stats['moves'] if base_stats['moves'] else 0.0
    positional_avg = (
        positional_stats['seconds'] / positional_stats['moves']
        if positional_stats['moves'] else 0.0
    )
    return {
        'game_id': game_id,
        'agente_a': 'base',
        'evaluacion_a': 'base',
        'profundidad_a': profundidad,
        'agente_b': 'posicional',
        'evaluacion_b': 'posicional',
        'profundidad_b': profundidad,
        'black_agent': 'base' if base_color == 1 else 'posicional',
        'resultado_a': result,
        'fichas_a': base_score,
        'fichas_b': positional_score,
        'diferencia_fichas_a': base_score - positional_score,
        'tiempo_promedio_mov_a': base_avg,
        'tiempo_promedio_mov_b': positional_avg,
        'nodos_evaluados_a': base_stats['nodes'],
        'nodos_evaluados_b': positional_stats['nodes'],
        'nodos_por_segundo_a': (
            base_stats['nodes'] / base_stats['seconds']
            if base_stats['seconds'] else 0.0
        ),
        'nodos_por_segundo_b': (
            positional_stats['nodes'] / positional_stats['seconds']
            if positional_stats['seconds'] else 0.0
        ),
    }


def ejecutar_torneo(profundidades=(4, 5, 6, 7), output=None):
    depths = tuple(dict.fromkeys(profundidades))
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("Especifique profundidades positivas")
    output_path = Path(output) if output else Path(__file__).with_name('resultados_othello.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    game_id = 0
    with output_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        csv_file.flush()
        for depth in depths:
            for base_color in (1, 2):
                game_id += 1
                row = jugar_partida(game_id, depth, base_color)
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
    print(f"{len(rows)} partidas registradas en {output_path}")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--depths',
        type=int,
        nargs='+',
        default=[4, 5, 6, 7],
        help='Profundidades a comparar (default: 4 5 6 7)',
    )
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ejecutar_torneo(args.depths, args.output)
