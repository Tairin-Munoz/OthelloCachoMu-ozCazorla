# Práctica de IA: Othello y Cacho

El repositorio conserva dos subproyectos independientes del material original:
`Othello-main` y `Cacho/Cacho`. Cada uno mantiene sus reglas y agente; los
torneos se ejecutan de forma separada.

## Requisitos

- Python 3.
- NumPy para Othello (`python -m pip install numpy` si falta).
- No se requieren librerías estadísticas o gráficas: el análisis produce SVG
  con la biblioteca estándar.

## Othello

`AgentesEspecificos/AgenteOthello.py` ofrece dos evaluaciones desde la
perspectiva fija de la raíz de cada búsqueda:

- **BASE:** `Δfichas + 200·Δesquinas + 2·Δmovilidad`.
- **POSICIONAL:** `1.2·Δfichas + 420·Δesquinas + 3·Δmovilidad + 0.8·Δposicional`,
  con una matriz que favorece esquinas y penaliza casillas vulnerables próximas.

La búsqueda conserva Alpha-Beta, cuenta nodos y trata pases como turnos sin
movimiento. La partida termina cuando ambos jugadores carecen de movimientos o
el tablero se llena.

Desde `Othello-main`:

```powershell
python -m unittest -v test_practica.py
python torneo_othello.py --depths 4 5
```

Cada profundidad ejecuta dos partidas, alternando quién juega negras. La
medición ejecutada y entregada contiene profundidades 4 y 5. Una partida a
profundidad 6 tomó más de diez minutos por juego en este equipo; por eso no se
incluyeron partidas incompletas de 6 ni resultados inventados de profundidad
7. Para intentar niveles mayores, ejecutar por separado
`python torneo_othello.py --depths 6` o `--depths 7`. Para una prueba rápida,
usar `python torneo_othello.py --depths 4`. El CSV
`resultados_othello.csv` registra evaluaciones, profundidad, color, fichas,
resultado, segundos por movimiento, nodos y nodos por segundo.

## Cacho

`AgenteCacho.py` conserva la evaluación del docente:

`E_base = (p_MAX + 0.6·V_futuro,MAX) - p_MIN`.

Las expectativas base se calculan con la probabilidad real de cada multiconjunto
de cinco dados, no dando el mismo peso a todas las combinaciones únicas.

La evaluación **MEJORADA** calcula para cada categoría disponible el máximo
puntaje esperado condicionado por los dados actuales y las tiradas restantes.
Enumera tiradas posibles con sus probabilidades y puntúa cada resultado con las
reglas implementadas por `puntuar`. Las expectativas de categorías se suman
como potenciales independientes, al igual que en la heurística base; no
significa que todas puedan maximizarse con una misma tirada. No se añade una
penalización de riesgo separada, para no contabilizar dos veces la dispersión
ya representada por la distribución.

Expectiminimax mantiene decisiones MAX/MIN y nodos de azar. El nuevo lanzamiento
de cinco dados al cambiar de turno también se modela como una esperanza. La
condición de fin existente se conserva: diez anotaciones totales.

Desde `Cacho/Cacho`:

```powershell
python -m unittest -v test_practica.py
python torneo_cacho.py --games 100 --seed 2025 --depth 1
```

BASE y MEJORADA alternan quién comienza; cada partida usa un generador aleatorio
local con su propia semilla. Las tiradas futuras solo las obtiene el simulador
cuando el agente elige relanzar dados. El CSV `resultados_cacho.csv` registra
semilla, agente inicial, puntajes, resultado, tiempos promedio, decisiones,
evaluaciones y profundidad.

## Estadística y gráficos

Después de ejecutar ambos torneos, desde la raíz del repositorio:

```powershell
python -m unittest -v test_analizar_resultados.py
python analizar_resultados.py
```

El script exige al menos 100 partidas completas de Cacho y genera
`resultados/resumen_estadistico.txt` y gráficos SVG. Informa puntajes y tiempos
promedio, diferencias, victorias/derrotas/empates, intervalo Wilson del 95% y
prueba binomial exacta bilateral para `H0: p=0.5` frente a `H1: p≠0.5`
(`α=0.05`). En Othello los enfrentamientos por profundidad tienen dos partidas
cada uno; esos cortes se describen, pero no se presentan como evidencia
suficiente de superioridad estadística.
