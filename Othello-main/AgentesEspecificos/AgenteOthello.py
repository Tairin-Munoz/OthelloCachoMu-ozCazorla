import numpy as np
from AgenteIA.AgenteJugador import AgenteJugador, ElEstado

class AgenteOthello(AgenteJugador):

    def __init__(self, altura=3):
        super().__init__(altura)
        self.BOARD_SIZE = 8

    def jugadas(self, estado):
        """Devuelve los movimientos válidos desde el estado actual."""
        return estado.movidas

    def testTerminal(self, estado):
        """Un estado es terminal si no hay movimientos válidos para ningún jugador."""
        return not estado.movidas

    def get_utilidad(self, estado, jugador):
        """Calcula la utilidad del estado final (diferencia de fichas)."""
        if not self.testTerminal(estado):
            return 0
        
        score_jugador = np.sum(estado.tablero == jugador)
        score_oponente = np.sum(estado.tablero == (3 - jugador))
        
        if score_jugador > score_oponente:
            return 100  # Victoria
        elif score_oponente > score_jugador:
            return -100 # Derrota
        else:
            return 0    # Empate

    def getResultado(self, estado, jugada):
        """
        Devuelve el nuevo estado del juego después de realizar una jugada.
        La lógica está adaptada de la clase OthelloGame.
        """
        if jugada not in self.jugadas(estado):
            return estado # Jugada inválida, no cambia el estado

        nuevo_tablero = np.copy(estado.tablero)
        jugador_actual = estado.jugador
        
        # Realizar la jugada
        nuevo_tablero[jugada] = jugador_actual

        # Voltear fichas
        oponente = 3 - jugador_actual
        direcciones = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in direcciones:
            r, c = jugada[0] + dr, jugada[1] + dc
            fichas_a_voltear = []
            
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and nuevo_tablero[r, c] == oponente:
                fichas_a_voltear.append((r, c))
                r += dr
                c += dc
            
            if 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and nuevo_tablero[r, c] == jugador_actual:
                for fr, fc in fichas_a_voltear:
                    nuevo_tablero[fr, fc] = jugador_actual

        # Calcular siguiente jugador y sus movimientos
        siguiente_jugador = 3 - jugador_actual
        movidas_siguiente = self._get_valid_moves(nuevo_tablero, siguiente_jugador)

        # Si el siguiente jugador no tiene movimientos, el turno vuelve al jugador actual
        if not movidas_siguiente:
            siguiente_jugador = jugador_actual
            movidas_siguiente = self._get_valid_moves(nuevo_tablero, siguiente_jugador)

        return ElEstado(
            jugador=siguiente_jugador,
            tablero=nuevo_tablero,
            movidas=movidas_siguiente,
            get_utilidad=0 # La utilidad solo se calcula en estados terminales
        )

    def funcion_evaluacion(self, estado):
        """
        Función heurística para evaluar un estado no terminal.
        Considera la diferencia de fichas, el control de las esquinas y la movilidad.
        """
        jugador = self.estado.jugador # El jugador para el que estamos evaluando
        oponente = 3 - jugador

        # 1. Diferencia de fichas
        score_fichas = np.sum(estado.tablero == jugador) - np.sum(estado.tablero == oponente)

        # 2. Control de esquinas (muy valiosas)
        esquinas = [(0, 0), (0, 7), (7, 0), (7, 7)]
        score_esquinas = 0
        for r, c in esquinas:
            if estado.tablero[r, c] == jugador:
                score_esquinas += 25
            elif estado.tablero[r, c] == oponente:
                score_esquinas -= 25

        # 3. Movilidad (número de movimientos posibles)
        movidas_jugador = len(self._get_valid_moves(estado.tablero, jugador))
        movidas_oponente = len(self._get_valid_moves(estado.tablero, oponente))
        score_movilidad = movidas_jugador - movidas_oponente
        
        # Puntuación final ponderada
        puntuacion_final = (1 * score_fichas) + (8 * score_esquinas) + (2 * score_movilidad)
        return puntuacion_final

    # --- Métodos de ayuda internos ---
    def _is_valid_move(self, board, player, row, col):
        if board[row, col] != 0:
            return False
        
        opponent = 3 - player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
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
        return [(r, c) for r in range(self.BOARD_SIZE) for c in range(self.BOARD_SIZE) if self._is_valid_move(board, player, r, c)]