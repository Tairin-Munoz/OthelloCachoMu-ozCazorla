from AgenteIA.AgenteJugador import AgenteJugador, ElEstado
from collections import Counter
from itertools import combinations_with_replacement
from functools import lru_cache
from math import factorial

# ============================================================
# CONSTANTES DEL JUEGO
# ============================================================

CATEGORIAS = ['1','2','3','4','5','6','escalera','full','poker','grande']

NOMBRES_CAT = {
    '1':'Balas (1)', '2':'Doses', '3':'Treses', '4':'Cuatros',
    '5':'Cincos', '6':'Seises', 'escalera':'Escalera',
    'full':'Full', 'poker':'Póker', 'grande':'Grande'
}

TODOS_LOS_ESTADOS = list(combinations_with_replacement(range(1, 7), 5))


def distribucion_dados(n_dados):
    """Distribución exacta de multiconjuntos de n dados justos."""
    total = 6 ** n_dados
    distribucion = {}
    for tirada in combinations_with_replacement(range(1, 7), n_dados):
        cuenta = Counter(tirada)
        formas = factorial(n_dados)
        for frecuencia in cuenta.values():
            formas //= factorial(frecuencia)
        distribucion[tirada] = formas / total
    return distribucion


DISTRIBUCION_CINCO_DADOS = distribucion_dados(5)

# ============================================================
# UTILIDADES
# ============================================================

def normalizar(dados):
    return tuple(sorted(dados))

def puntuar(categoria, dados):
    c = Counter(dados)
    valores = sorted(c.values(), reverse=True)
    if categoria in '123456':
        return dados.count(int(categoria)) * int(categoria)
    elif categoria == 'escalera':
        s = set(dados)
        return 25 if s == {1,2,3,4,5} or s == {2,3,4,5,6} else 0
    elif categoria == 'full':
        return 30 if valores == [3, 2] else 0
    elif categoria == 'poker':
        return 40 if valores[0] >= 4 else 0
    elif categoria == 'grande':
        return 50 if valores[0] == 5 else 0
    return 0

def probabilidad_tirada(dados_actuales, dados_a_tirar):
    indices_mantener = [i for i in range(5) if i not in dados_a_tirar]
    dados_fijos = [dados_actuales[i] for i in indices_mantener]
    n_tirar = len(dados_a_tirar)
    resultados = combinations_with_replacement(range(1, 7), n_tirar)
    total = 6 ** n_tirar
    probas = {}
    for combo in resultados:
        cuenta = Counter(combo)
        formas = factorial(n_tirar)
        for v in cuenta.values():
            formas //= factorial(v)
        estado = normalizar(list(dados_fijos) + list(combo))
        probas[estado] = probas.get(estado, 0) + formas / total
    return probas


@lru_cache(maxsize=None)
def valor_esperado_categoria(categoria, dados, tiradas_restantes):
    """Máximo puntaje esperado para una sola categoría con hasta n relanzamientos."""
    valor_actual = puntuar(categoria, dados)
    if tiradas_restantes <= 0:
        return float(valor_actual)

    # Retener cada submulticonjunto distinto evita recalcular máscaras equivalentes
    # cuando los dados repetidos tienen el mismo valor.
    opciones_retener = {()}
    for dado in dados:
        opciones_retener |= {opcion + (dado,) for opcion in tuple(opciones_retener)}

    mejor_valor = float(valor_actual)
    for retenidos in opciones_retener:
        frecuencia_retenida = Counter(retenidos)
        indices_retenidos = set()
        usados = Counter()
        for indice, dado in enumerate(dados):
            if usados[dado] < frecuencia_retenida[dado]:
                indices_retenidos.add(indice)
                usados[dado] += 1
        indices_tirar = frozenset(set(range(5)) - indices_retenidos)
        if not indices_tirar:
            continue
        esperado = sum(
            probabilidad * valor_esperado_categoria(
                categoria, nueva_tirada, tiradas_restantes - 1
            )
            for nueva_tirada, probabilidad in probabilidad_tirada(dados, indices_tirar).items()
        )
        mejor_valor = max(mejor_valor, esperado)
    return mejor_valor


@lru_cache(maxsize=None)
def valor_esperado_categoria_desde_inicio(categoria):
    """Esperanza de la estrategia óptima para la categoría desde cinco dados nuevos."""
    return sum(
        probabilidad * valor_esperado_categoria(categoria, dados, 3)
        for dados, probabilidad in DISTRIBUCION_CINCO_DADOS.items()
    )

def acciones_reservar(dados):
    acciones = set()
    for mascara in range(32):
        mantener = tuple(dados[i] for i in range(5) if mascara & (1 << i))
        tirar = frozenset(i for i in range(5) if not (mascara & (1 << i)))
        acciones.add((normalizar(mantener), tirar))
    return list(acciones)

# ============================================================
# CLASE BASE PARA AGENTES DEL CACHO
# ============================================================

class AgenteCacho(AgenteJugador):
    """Clase base con lógica común a IA y humano."""

    def __init__(self, nombre="Agente", altura=1):
        AgenteJugador.__init__(self, altura=altura)
        self.nombre = nombre
        self.max_profundidad = altura

    # --------- API requerida por AgenteJugador ---------

    def jugadas(self, estado):
        """
        Genera todas las acciones legales:
        - ('tirar', frozenset_indices) si quedan tiradas
        - ('anotar', categoria) siempre
        """
        acciones = []
        turno = estado.jugador
        disponibles = [c for c in CATEGORIAS if c not in estado.tablero['usadas'][turno]]

        # Anotar siempre es posible
        for cat in disponibles:
            acciones.append(('anotar', cat))

        # Tirar solo si quedan tiradas
        if estado.tablero['tiradas_restantes'] > 0:
            for _, tirar in acciones_reservar(estado.tablero['dados']):
                if tirar:
                    acciones.append(('tirar', tirar))

        return acciones

    def getResultado(self, estado, m):
        """Aplica una acción y devuelve el nuevo estado."""
        tablero = self._copiar_tablero(estado.tablero)

        tipo, param = m
        turno = estado.jugador

        if tipo == 'anotar':
            cat = param
            if cat not in CATEGORIAS or cat in tablero['usadas'][turno]:
                raise ValueError(f"Categoría no disponible: {cat}")
            pts = puntuar(cat, tablero['dados'])
            tablero['puntajes'][turno] += pts
            tablero['usadas'][turno].add(cat)
            tablero['dados'] = (1, 1, 1, 1, 1)
            tablero['tiradas_restantes'] = 3
            tablero['ronda'] = estado.tablero.get('ronda', 0) + 1
            nuevo_jugador = 1 - turno
        elif tipo == 'tirar':
            if not param or tablero['tiradas_restantes'] <= 0:
                raise ValueError("No quedan tiradas o no se seleccionaron dados")
            tablero['tiradas_restantes'] -= 1
            nuevo_jugador = turno
        else:
            raise ValueError(f"Acción desconocida: {tipo}")

        return ElEstado(
            jugador=nuevo_jugador,
            get_utilidad=0,
            tablero=tablero,
            movidas=self.jugadas(ElEstado(nuevo_jugador, 0, tablero, None))
        )

    def get_utilidad(self, estado, jugador):
        """Utilidad: diferencia de puntaje desde la perspectiva del jugador."""
        p = estado.tablero['puntajes']
        return (p[jugador] - p[1 - jugador])

    def testTerminal(self, estado):
        return estado.tablero.get('ronda', 0) >= 10

    # --------- Función de evaluación para hojas ---------

    def funcion_evaluacion(self, estado):
        """Mantiene la heurística base del docente para comparación."""
        return self.funcion_evaluacion_base(estado)

    def funcion_evaluacion_base(self, estado):
        """Evaluación docente: E=(pMAX+0.6*Vfuturo,MAX)-pMIN."""
        p = estado.tablero['puntajes']
        restantes_max = [c for c in CATEGORIAS if c not in estado.tablero['usadas'][0]]
        futuro_max = sum(self._valor_esperado_cat(c) for c in restantes_max)
        if self.testTerminal(estado):
            return p[0] - p[1]
        return (p[0] + futuro_max * 0.6) - p[1]

    def funcion_evaluacion_mejorada(self, estado):
        """
        Expectativa dinámica exacta por categoría, con distribución real de dados,
        puntuación y relanzamientos restantes. Cada categoría se estima por separado,
        como en la heurística base; su suma es potencial y no una garantía conjunta.
        """
        p = estado.tablero['puntajes']
        if self.testTerminal(estado):
            return p[0] - p[1]

        futuros = [0.0, 0.0]
        for jugador in (0, 1):
            restantes = [c for c in CATEGORIAS if c not in estado.tablero['usadas'][jugador]]
            if jugador == estado.jugador:
                futuros[jugador] = sum(
                    valor_esperado_categoria(
                        categoria,
                        tuple(sorted(estado.tablero['dados'])),
                        estado.tablero['tiradas_restantes'],
                    )
                    for categoria in restantes
                )
            else:
                futuros[jugador] = sum(
                    valor_esperado_categoria_desde_inicio(categoria)
                    for categoria in restantes
                )
        return (p[0] + 0.6 * futuros[0]) - (p[1] + 0.6 * futuros[1])

    def _valor_esperado_cat(self, categoria):
        if not hasattr(self, '_cache_ve'):
            self._cache_ve = {}
        if categoria not in self._cache_ve:
            self._cache_ve[categoria] = sum(
                probability * puntuar(categoria, dados)
                for dados, probability in DISTRIBUCION_CINCO_DADOS.items()
            )
        return self._cache_ve[categoria]

    def _copiar_tablero(self, tablero):
        return {
            'dados': tuple(tablero['dados']),
            'tiradas_restantes': tablero['tiradas_restantes'],
            'puntajes': list(tablero['puntajes']),
            'usadas': [set(tablero['usadas'][0]), set(tablero['usadas'][1])],
            'ronda': tablero['ronda'],
        }