from AgenteCacho import (
    AgenteCacho,
    DISTRIBUCION_CINCO_DADOS,
    probabilidad_tirada,
)
from AgenteIA.AgenteJugador import ElEstado


class AgenteCachoIA(AgenteCacho):

    def __init__(self, nombre="IA", altura=1, usar_mejorada=True):
        super().__init__(nombre=nombre, altura=altura)
        self.usar_mejorada = usar_mejorada
        self._memo = {}
        self.nodos_evaluados = 0
        self.evaluaciones = 0

    def _evaluacion_hoja(self, estado):
        self.evaluaciones += 1
        if self.usar_mejorada:
            return self.funcion_evaluacion_mejorada(estado)
        return self.funcion_evaluacion_base(estado)

    @staticmethod
    def _clave_estado(estado, profundidad, es_max, version):
        tablero = estado.tablero
        return (
            estado.jugador,
            tuple(tablero['dados']),
            tablero['tiradas_restantes'],
            tuple(tablero['puntajes']),
            tuple(tuple(sorted(usadas)) for usadas in tablero['usadas']),
            tablero.get('ronda', 0),
            profundidad,
            es_max,
            version,
        )

    @staticmethod
    def _estado_con_dados(estado, dados):
        tablero = {
            'dados': tuple(dados),
            'tiradas_restantes': estado.tablero['tiradas_restantes'],
            'puntajes': list(estado.tablero['puntajes']),
            'usadas': [set(estado.tablero['usadas'][0]), set(estado.tablero['usadas'][1])],
            'ronda': estado.tablero.get('ronda', 0),
        }
        return ElEstado(
            jugador=estado.jugador,
            get_utilidad=0,
            tablero=tablero,
            movidas=None,
        )

    def _expectativa_nueva_ronda(self, estado, profundidad, es_max):
        """La tirada inicial del siguiente turno también es un nodo de azar."""
        return sum(
            probabilidad * self.expectiminimax(
                self._estado_con_dados(estado, dados), profundidad, es_max
            )
            for dados, probabilidad in DISTRIBUCION_CINCO_DADOS.items()
        )

    def _valor_accion(self, estado, accion, profundidad):
        tipo, parametro = accion
        es_max = estado.jugador == 0

        if tipo == 'anotar':
            siguiente = self.getResultado(estado, accion)
            if self.testTerminal(siguiente):
                return self._evaluacion_hoja(siguiente)
            return self._expectativa_nueva_ronda(siguiente, profundidad + 1, not es_max)

        if tipo == 'tirar':
            probabilidades = probabilidad_tirada(estado.tablero['dados'], parametro)
            valor = 0.0
            for dados, probabilidad in probabilidades.items():
                siguiente = self._estado_con_dados(estado, dados)
                siguiente.tablero['tiradas_restantes'] -= 1
                valor += probabilidad * self.expectiminimax(
                    siguiente, profundidad + 1, es_max
                )
            return valor

        raise ValueError(f"Acción desconocida: {tipo}")

    def expectiminimax(self, estado, profundidad, es_max):
        """Expectiminimax: decisiones MAX/MIN y esperanza exacta en nodos de azar."""
        self.nodos_evaluados += 1
        version = 'mejorada' if self.usar_mejorada else 'base'
        clave = self._clave_estado(estado, profundidad, es_max, version)
        if clave in self._memo:
            return self._memo[clave]

        if self.testTerminal(estado) or profundidad >= self.max_profundidad:
            valor = self._evaluacion_hoja(estado)
            self._memo[clave] = valor
            return valor

        acciones = self.jugadas(estado)
        if not acciones:
            valor = self._evaluacion_hoja(estado)
        else:
            valores = [
                self._valor_accion(estado, accion, profundidad)
                for accion in acciones
            ]
            valor = max(valores) if estado.jugador == 0 else min(valores)

        self._memo[clave] = valor
        return valor

    def elegir_accion(self, estado):
        """Selecciona la mejor acción; la búsqueda parte de un estado limpio."""
        acciones = self.jugadas(estado)
        if not acciones:
            return None

        self._memo.clear()
        self.nodos_evaluados = 0
        self.evaluaciones = 0
        valores = [
            (self._valor_accion(estado, accion, 0), accion)
            for accion in acciones
        ]
        if estado.jugador == 0:
            return max(valores, key=lambda item: item[0])[1]
        return min(valores, key=lambda item: item[0])[1]

    def programa(self):
        self.set_acciones([])
        accion = self.elegir_accion(self.estado)
        if accion is not None:
            self.set_acciones(accion)
