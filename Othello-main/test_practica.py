import unittest
import tempfile
from pathlib import Path

import numpy as np

from AgenteIA.AgenteJugador import ElEstado
from AgentesEspecificos.AgenteOthello import AgenteOthello
from torneo_othello import ejecutar_torneo


def estado_inicial():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = 2
    board[3, 4] = board[4, 3] = 1
    return board


class TestOthello(unittest.TestCase):

    def test_movimientos_legales_iniciales(self):
        agent = AgenteOthello()
        self.assertEqual(
            set(agent._get_valid_moves(estado_inicial(), 1)),
            {(2, 3), (3, 2), (4, 5), (5, 4)},
        )

    def test_movimiento_voltea_ficha(self):
        agent = AgenteOthello()
        board = estado_inicial()
        state = ElEstado(1, 0, board, agent._get_valid_moves(board, 1))
        next_state = agent.getResultado(state, (2, 3))
        self.assertEqual(next_state.tablero[2, 3], 1)
        self.assertEqual(next_state.tablero[3, 3], 1)
        self.assertEqual(next_state.jugador, 2)

    def test_pase_y_terminalidad(self):
        agent = AgenteOthello()
        board = np.ones((8, 8), dtype=int)
        board[0, 0] = 0
        board[0, 2] = 2
        state = ElEstado(1, 0, board, [])
        self.assertFalse(agent.testTerminal(state))
        passed = agent.getResultado(state, None)
        self.assertEqual(passed.jugador, 2)
        self.assertIn((0, 0), passed.movidas)

        board[0, 2] = 1
        terminal = ElEstado(2, 0, board, [])
        self.assertTrue(agent.testTerminal(terminal))
        self.assertTrue(agent.testTerminal(
            ElEstado(1, 0, np.ones((8, 8), dtype=int), [])
        ))

    def test_evaluaciones_son_consistentes_con_perspectiva_raiz(self):
        agent = AgenteOthello(estrategia='base')
        board = estado_inicial()
        first = ElEstado(1, 0, board, agent._get_valid_moves(board, 1))
        second = ElEstado(2, 0, board, agent._get_valid_moves(board, 2))
        agent.jugador_raiz = 1
        self.assertEqual(agent.funcion_evaluacion_base(first),
                         agent.funcion_evaluacion_base(second))
        positional = AgenteOthello(estrategia='posicional')
        positional.jugador_raiz = 1
        self.assertIsInstance(positional.funcion_evaluacion_posicional(first), float)

    def test_busqueda_alpha_beta_retira_movimiento_legal_y_cuenta_nodos(self):
        agent = AgenteOthello(altura=4, estrategia='base')
        board = estado_inicial()
        state = ElEstado(1, 0, board, agent._get_valid_moves(board, 1))
        move = agent.podaAlphaBeta_eval(state)
        self.assertIn(move, state.movidas)
        self.assertGreater(agent.nodos_evaluados, 0)
        self.assertGreater(agent.evaluaciones, 0)

    def test_torneo_mide_ambos_colores_a_la_misma_profundidad(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = ejecutar_torneo([1], Path(directory) / 'results.csv')
            self.assertEqual(len(rows), 2)
            self.assertEqual({row['profundidad_a'] for row in rows}, {1})
            self.assertEqual({row['profundidad_b'] for row in rows}, {1})
            self.assertEqual({row['black_agent'] for row in rows}, {'base', 'posicional'})
            for row in rows:
                self.assertIn(row['resultado_a'], ('victoria', 'derrota', 'empate'))
                self.assertGreater(row['nodos_evaluados_a'], 0)
                self.assertGreater(row['nodos_evaluados_b'], 0)


if __name__ == '__main__':
    unittest.main()
