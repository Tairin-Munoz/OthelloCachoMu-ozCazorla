import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from AgenteCacho import (
    CATEGORIAS,
    DISTRIBUCION_CINCO_DADOS,
    AgenteCacho,
    normalizar,
    puntuar,
    probabilidad_tirada,
)
from AgenteCachoIA import AgenteCachoIA
from EntornoCacho import EntornoCacho
from AgenteIA.AgenteJugador import ElEstado
from torneo_cacho import ejecutar_torneo


def estado(jugador=0, dados=(1, 2, 3, 4, 5), restantes=3, ronda=0):
    return ElEstado(
        jugador=jugador,
        get_utilidad=0,
        tablero={
            'dados': dados,
            'tiradas_restantes': restantes,
            'puntajes': [0, 0],
            'usadas': [set(), set()],
            'ronda': ronda,
        },
        movidas=None,
    )


class TestCacho(unittest.TestCase):

    def test_probabilidades_de_tirada_suman_uno_y_ponderan_repetidos(self):
        outcomes = probabilidad_tirada((1, 1, 1, 1, 1), frozenset(range(5)))
        self.assertAlmostEqual(sum(outcomes.values()), 1.0)
        self.assertAlmostEqual(outcomes[(1, 1, 1, 1, 1)], 1 / 7776)
        self.assertAlmostEqual(outcomes[(1, 1, 1, 1, 2)], 5 / 7776)
        self.assertEqual(len(DISTRIBUCION_CINCO_DADOS), 252)

    def test_puntuaciones_reales(self):
        self.assertEqual(puntuar('6', (6, 6, 1, 2, 3)), 12)
        self.assertEqual(puntuar('escalera', (1, 2, 3, 4, 5)), 25)
        self.assertEqual(puntuar('full', (2, 2, 2, 5, 5)), 30)
        self.assertEqual(puntuar('poker', (4, 4, 4, 4, 2)), 40)
        self.assertEqual(puntuar('grande', (3, 3, 3, 3, 3)), 50)

    def test_acciones_disponibles_y_actualizacion_de_estado(self):
        agent = AgenteCacho()
        start = estado(dados=(2, 2, 3, 4, 5), restantes=2)
        actions = agent.jugadas(start)
        self.assertEqual(
            {parameter for kind, parameter in actions if kind == 'anotar'},
            set(CATEGORIAS),
        )
        reroll = agent.getResultado(start, ('tirar', frozenset({0})))
        self.assertEqual(reroll.jugador, start.jugador)
        self.assertEqual(reroll.tablero['tiradas_restantes'], 1)
        self.assertEqual(reroll.tablero['dados'], start.tablero['dados'])

        scored = agent.getResultado(start, ('anotar', 'full'))
        self.assertEqual(scored.jugador, 1)
        self.assertEqual(scored.tablero['puntajes'][0], 0)
        self.assertIn('full', scored.tablero['usadas'][0])
        self.assertEqual(scored.tablero['ronda'], 1)
        self.assertEqual(scored.tablero['tiradas_restantes'], 3)

    def test_condicion_terminal_es_diez_anotaciones(self):
        agent = AgenteCacho()
        self.assertFalse(agent.testTerminal(estado(ronda=9)))
        self.assertTrue(agent.testTerminal(estado(ronda=10)))

    def test_entorno_respeta_turno_puntuacion_terminal_y_semilla(self):
        environments = [EntornoCacho(seed=73), EntornoCacho(seed=73)]
        for environment in environments:
            environment.insertar(AgenteCachoIA())
            environment.insertar(AgenteCachoIA())
        with redirect_stdout(io.StringIO()):
            for environment in environments:
                environment._aplicar_accion(('anotar', '1'))
        states = [environment.get_estado() for environment in environments]
        self.assertEqual(states[0].jugador, 1)
        self.assertEqual(states[0].tablero['puntajes'][0], 5)
        self.assertIn('1', states[0].tablero['usadas'][0])
        self.assertEqual(states[0].tablero['ronda'], 1)
        self.assertEqual(states[0].tablero['dados'], states[1].tablero['dados'])

        roll_environment = EntornoCacho(seed=73)
        roll_environment.insertar(AgenteCachoIA())
        roll_environment.insertar(AgenteCachoIA())
        with redirect_stdout(io.StringIO()):
            roll_environment._aplicar_accion(('tirar', frozenset({0})))
        rolled_state = roll_environment.get_estado()
        self.assertEqual(rolled_state.jugador, 0)
        self.assertEqual(rolled_state.tablero['tiradas_restantes'], 2)

        final_environment = EntornoCacho(seed=73)
        final_environment.insertar(AgenteCachoIA())
        final_environment.insertar(AgenteCachoIA())
        final_environment.juegoActual.tablero['ronda'] = 9
        with redirect_stdout(io.StringIO()):
            final_environment._aplicar_accion(('anotar', '1'))
        self.assertTrue(final_environment.es_fin())
        self.assertTrue(final_environment.finalizar())

    def test_evaluaciones_son_de_perspectiva_max_y_esperanzas_calculadas(self):
        agent = AgenteCacho()
        start = estado(dados=(1, 1, 1, 1, 1))
        expected = sum(
            probability * puntuar('grande', dice)
            for dice, probability in DISTRIBUCION_CINCO_DADOS.items()
        )
        self.assertAlmostEqual(agent._valor_esperado_cat('grande'), expected)

        start.tablero['usadas'][0].update(c for c in CATEGORIAS if c != 'grande')
        start.tablero['puntajes'][:] = [20, 7]
        base = agent.funcion_evaluacion_base(start)
        self.assertAlmostEqual(base, 20 + 0.6 * expected - 7)
        other_to_move = ElEstado(
            jugador=1,
            get_utilidad=0,
            tablero=start.tablero,
            movidas=None,
        )
        self.assertAlmostEqual(agent.funcion_evaluacion_base(other_to_move), base)

    def test_expectiminimax_pesa_las_tiradas_por_su_probabilidad(self):
        agent = AgenteCachoIA(altura=1, usar_mejorada=False)
        start = estado()
        expected = agent.funcion_evaluacion_base(start)
        actual = agent._expectativa_nueva_ronda(start, 1, True)
        self.assertAlmostEqual(actual, expected)
        self.assertEqual(agent.nodos_evaluados, len(DISTRIBUCION_CINCO_DADOS))
        roll_value = agent._valor_accion(start, ('tirar', frozenset({0})), 0)
        self.assertAlmostEqual(roll_value, expected)

    def test_memo_clave_distingue_datos_estado_version_y_profundidad(self):
        agent = AgenteCachoIA()
        first = estado()
        second = estado(dados=(1, 2, 3, 4, 6))
        self.assertNotEqual(agent._clave_estado(first, 0, True, 'base'),
                            agent._clave_estado(second, 0, True, 'base'))
        self.assertNotEqual(agent._clave_estado(first, 0, True, 'base'),
                            agent._clave_estado(first, 1, True, 'base'))
        self.assertNotEqual(agent._clave_estado(first, 0, True, 'base'),
                            agent._clave_estado(first, 0, True, 'mejorada'))

    def test_torneo_es_reproducible_completo_y_alterna_inicio(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / 'results.csv'
            rows = ejecutar_torneo(2, 100, 1, output)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row['first_agent'] for row in rows], ['base', 'mejorada'])
            self.assertTrue(output.exists())
            self.assertEqual(rows[0]['seed'], 100)
            for row in rows:
                self.assertIn(row['winner'], ('base', 'mejorada', 'empate'))
                self.assertGreaterEqual(row['base_decisions'], 5)
                self.assertGreaterEqual(row['improved_decisions'], 5)
                self.assertGreater(row['base_evaluations'], 0)
                self.assertGreater(row['improved_evaluations'], 0)


if __name__ == '__main__':
    unittest.main()
