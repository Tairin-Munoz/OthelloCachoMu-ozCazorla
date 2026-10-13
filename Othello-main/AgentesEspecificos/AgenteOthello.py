import numpy as np

from AgenteIA.AgenteJugador import AgenteJugador, ElEstado


PESOS_ESTRATEGICOS = np.array([
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120],
], dtype=float)

ESQUINAS = ((0, 0), (0, 7), (7, 0), (7, 7))
DIRECCIONES = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)


class AgenteOthello(AgenteJugador):

    def __init__(self, altura=3, estrategia="base"):
        super().__init__(altura)
        if estrategia not in ("base", "posicional"):
            raise ValueError("La estrategia debe ser 'base' o 'posicional'")
        self.BOARD_SIZE = 8
        self.estrategia = estrategia
        self.jugador_raiz = None
        self.nodos_evaluados = 0
        self.evaluaciones = 0

    def jugadas(self, estado):
        return self._get_valid_moves(estado.tablero, estado.jugador)

    def testTerminal(self, estado):
        tablero = estado.tablero
        if not np.any(tablero == 0):
            return True
        return (
            not self._get_valid_moves(tablero, 1)
            and not self._get_valid_moves(tablero, 2)
        )

    def get_utilidad(self, estado, jugador):
        if not self.testTerminal(estado):
            return 0
        diferencia = int(np.sum(estado.tablero == jugador) - np.sum(estado.tablero == 3 - jugador))
        if diferencia > 0:
            return 100
        if diferencia < 0:
            return -100
        return 0

    def getResultado(self, estado, jugada):
        jugador = estado.jugador
        legal = self._get_valid_moves(estado.tablero, jugador)
        if jugada is None:
            if legal:
                raise ValueError("No se puede pasar si existen movimientos legales")
            oponente = 3 - jugador
            op_moves = self._get_valid_moves(estado.tablero, oponente)
            return ElEstado(
                jugador=oponente,
                tablero=np.copy(estado.tablero),
                movidas=op_moves,
                get_utilidad=0,
            )
        if jugada not in legal:
            raise ValueError(f"Movimiento ilegal: {jugada}")

        return self._resultado_legal(estado, jugada)

    def _resultado_legal(self, estado, jugada):
        jugador = estado.jugador
        nuevo_tablero = np.copy(estado.tablero)
        nuevo_tablero[jugada] = jugador
        oponente = 3 - jugador

        for dr, dc in DIRECCIONES:
            r, c = jugada[0] + dr, jugada[1] + dc
            fichas_a_voltear = []
            while (
                0 <= r < self.BOARD_SIZE
                and 0 <= c < self.BOARD_SIZE
                and nuevo_tablero[r, c] == oponente
            ):
                fichas_a_voltear.append((r, c))
                r += dr
                c += dc
            if (
                fichas_a_voltear
                and 0 <= r < self.BOARD_SIZE
                and 0 <= c < self.BOARD_SIZE
                and nuevo_tablero[r, c] == jugador
            ):
                for fr, fc in fichas_a_voltear:
                    nuevo_tablero[fr, fc] = jugador

        siguiente = oponente
        siguientes_movidas = self._get_valid_moves(nuevo_tablero, siguiente)
        if not siguientes_movidas:
            propias = self._get_valid_moves(nuevo_tablero, jugador)
            if propias:
                siguiente = jugador
                siguientes_movidas = propias

        return ElEstado(
            jugador=siguiente,
            tablero=nuevo_tablero,
            movidas=siguientes_movidas,
            get_utilidad=0,
        )

    def _perspectiva(self, estado):
        return self.jugador_raiz if self.jugador_raiz is not None else estado.jugador

    def funcion_evaluacion(self, estado):
        if self.estrategia == "posicional":
            return self.funcion_evaluacion_posicional(estado)
        return self.funcion_evaluacion_base(estado)

    def funcion_evaluacion_base(self, estado):
        """
        BASE: E = Δfichas + 200·Δesquinas + 2·Δmovilidad.
        Todas las diferencias son raíz menos oponente; esquina (25) se
        pondera por 8, y la movilidad legal por 2.
        """
        jugador = self._perspectiva(estado)
        oponente = 3 - jugador
        tablero = estado.tablero
        delta_fichas = int(np.sum(tablero == jugador) - np.sum(tablero == oponente))
        delta_esquinas = sum(
            1 if tablero[r, c] == jugador else -1 if tablero[r, c] == oponente else 0
            for r, c in ESQUINAS
        )
        delta_movilidad = (
            len(self._get_valid_moves(tablero, jugador))
            - len(self._get_valid_moves(tablero, oponente))
        )
        return delta_fichas + 200 * delta_esquinas + 2 * delta_movilidad

    def funcion_evaluacion_posicional(self, estado):
        """
        POSICIONAL: E = 1.2·Δfichas + 420·Δesquinas
        + 3·Δmovilidad + 0.8·Δposicional.
        La matriz asigna valores positivos a esquinas/bordes estables y
        negativos a casillas peligrosas cercanas a esquinas vacías.
        """
        jugador = self._perspectiva(estado)
        oponente = 3 - jugador
        tablero = estado.tablero
        delta_fichas = int(np.sum(tablero == jugador) - np.sum(tablero == oponente))
        delta_esquinas = sum(
            1 if tablero[r, c] == jugador else -1 if tablero[r, c] == oponente else 0
            for r, c in ESQUINAS
        )
        delta_movilidad = (
            len(self._get_valid_moves(tablero, jugador))
            - len(self._get_valid_moves(tablero, oponente))
        )
        delta_posicional = float(
            np.sum(PESOS_ESTRATEGICOS[tablero == jugador])
            - np.sum(PESOS_ESTRATEGICOS[tablero == oponente])
        )
        return (
            1.2 * delta_fichas
            + 420 * delta_esquinas
            + 3 * delta_movilidad
            + 0.8 * delta_posicional
        )

    def podaAlphaBeta_eval(self, estado):
        """Alpha-Beta con evaluación coherente desde el jugador raíz."""
        self.jugador_raiz = estado.jugador
        self.nodos_evaluados = 0
        self.evaluaciones = 0
        if not self._get_valid_moves(estado.tablero, estado.jugador):
            return None

        def buscar(actual, profundidad, alpha, beta):
            self.nodos_evaluados += 1
            movidas = self._get_valid_moves(actual.tablero, actual.jugador)
            movidas_oponente = self._get_valid_moves(actual.tablero, 3 - actual.jugador)
            if not movidas and not movidas_oponente:
                diferencia = int(
                    np.sum(actual.tablero == self.jugador_raiz)
                    - np.sum(actual.tablero == 3 - self.jugador_raiz)
                )
                if diferencia > 0:
                    return 100000 + diferencia
                if diferencia < 0:
                    return -100000 + diferencia
                return 0
            if profundidad >= self.altura:
                self.evaluaciones += 1
                return self.funcion_evaluacion(actual)

            if not movidas:
                pasado = ElEstado(
                    jugador=3 - actual.jugador,
                    tablero=actual.tablero,
                    movidas=movidas_oponente,
                    get_utilidad=0,
                )
                return buscar(pasado, profundidad + 1, alpha, beta)

            es_max = actual.jugador == self.jugador_raiz
            mejor = -float('inf') if es_max else float('inf')
            movidas.sort(
                key=lambda movimiento: PESOS_ESTRATEGICOS[movimiento],
                reverse=es_max,
            )
            for movimiento in movidas:
                siguiente = self._resultado_legal(actual, movimiento)
                valor = buscar(siguiente, profundidad + 1, alpha, beta)
                if es_max:
                    mejor = max(mejor, valor)
                    alpha = max(alpha, mejor)
                else:
                    mejor = min(mejor, valor)
                    beta = min(beta, mejor)
                if beta <= alpha:
                    break
            return mejor

        mejor_accion = None
        mejor_valor = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for movimiento in self._get_valid_moves(estado.tablero, estado.jugador):
            siguiente = self._resultado_legal(estado, movimiento)
            valor = buscar(siguiente, 1, alpha, beta)
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = movimiento
            alpha = max(alpha, mejor_valor)
        return mejor_accion

    def _is_valid_move(self, board, player, row, col):
        if board[row, col] != 0:
            return False
        opponent = 3 - player
        for dr, dc in DIRECCIONES:
            r, c = row + dr, col + dc
            found_opponent = False
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and board[r, c] == opponent:
                found_opponent = True
                r += dr
                c += dc
            if found_opponent and 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and board[r, c] == player:
                return True
        return False

    def _get_valid_moves(self, board, player):
        return [
            (r, c)
            for r in range(self.BOARD_SIZE)
            for c in range(self.BOARD_SIZE)
            if self._is_valid_move(board, player, r, c)
        ]

    def programa(self):
        if not self._get_valid_moves(self.estado.tablero, self.estado.jugador):
            self.set_acciones(None)
            return
        self.set_acciones(self.podaAlphaBeta_eval(self.estado))
