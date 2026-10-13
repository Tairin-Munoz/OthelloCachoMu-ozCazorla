"""Calculate descriptive statistics and SVG plots from the tournament CSVs."""

import csv
import html
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHO_CSV = ROOT / 'Cacho' / 'Cacho' / 'resultados_cacho.csv'
OTHELLO_CSV = ROOT / 'Othello-main' / 'resultados_othello.csv'
OUTPUT = ROOT / 'resultados'
Z_95 = 1.959963984540054


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"El CSV no contiene partidas: {path}")
    return rows


def wilson(successes, total):
    if total == 0:
        return (float('nan'), float('nan'))
    proportion = successes / total
    denominator = 1 + Z_95 ** 2 / total
    center = (proportion + Z_95 ** 2 / (2 * total)) / denominator
    radius = (
        Z_95
        * math.sqrt(proportion * (1 - proportion) / total + Z_95 ** 2 / (4 * total ** 2))
        / denominator
    )
    return center - radius, center + radius


def exact_binomial_two_sided(successes, total):
    """Two-sided exact binomial p-value under H0: p=0.5."""
    if total == 0:
        return float('nan')
    probabilities = [
        math.comb(total, k) / (2 ** total)
        for k in range(total + 1)
    ]
    observed = probabilities[successes]
    return min(1.0, sum(p for p in probabilities if p <= observed + 1e-15))


def bootstrap_mean_interval(values, resamples=5000, seed=2025):
    if not values:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        mean([values[rng.randrange(size)] for _ in range(size)])
        for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]


def percent(value, total):
    return 100 * value / total if total else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def svg_bar(path, title, labels, series, ylabel):
    width, height = 720, 420
    left, right, top, bottom = 70, 30, 65, 90
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [value for _, data, _ in series for value in data] or [1]
    lower = min(0.0, min(values))
    upper = max(0.0, max(values))
    span = upper - lower or 1.0
    padding = span * 0.12
    lower = lower - padding if lower < 0 else 0.0
    upper = upper + padding if upper > 0 else 1.0

    def y_coordinate(value):
        return top + plot_height * (upper - value) / (upper - lower)

    group_width = plot_width / max(1, len(labels))
    bar_width = group_width / (len(series) + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(ylabel)}</text>',
    ]
    for tick in range(6):
        value = upper - (upper - lower) * tick / 5
        y = y_coordinate(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for label_index, label in enumerate(labels):
        center = left + group_width * (label_index + 0.5)
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(str(label))}</text>')
        for series_index, (series_name, values, color) in enumerate(series):
            value = values[label_index]
            x = center + (series_index - (len(series) - 1) / 2) * bar_width - bar_width * 0.4
            baseline = y_coordinate(0)
            value_y = y_coordinate(value)
            y = min(baseline, value_y)
            bar_height = abs(value_y - baseline)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width*0.8:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
    legend_y = height - 20
    legend_start = width / 2 - len(series) * 90
    for index, (name, _, color) in enumerate(series):
        x = legend_start + index * 180
        parts.append(f'<rect x="{x:.1f}" y="{legend_y-11}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x+18:.1f}" y="{legend_y}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def analyze_cacho(rows):
    wins = Counter(row['winner'] for row in rows)
    total = len(rows)
    decisive = wins['base'] + wins['mejorada']
    low, high = wilson(wins['base'], decisive)
    p_value = exact_binomial_two_sided(wins['base'], decisive)
    base_scores = [int(row['base_score']) for row in rows]
    improved_scores = [int(row['improved_score']) for row in rows]
    base_times = [float(row['base_avg_decision_seconds']) for row in rows]
    improved_times = [float(row['improved_avg_decision_seconds']) for row in rows]
    base_diff = [a - b for a, b in zip(base_scores, improved_scores)]
    score_ci_low, score_ci_high = bootstrap_mean_interval(base_diff)

    summary = [
        'CACHO — Base vs Mejorada',
        f'Partidas completas: {total}',
        f'Victorias Base: {wins["base"]} ({percent(wins["base"], total):.1f}%)',
        f'Victorias Mejorada: {wins["mejorada"]} ({percent(wins["mejorada"], total):.1f}%)',
        f'Empates: {wins["empate"]} ({percent(wins["empate"], total):.1f}%)',
        f'Puntaje promedio Base: {mean(base_scores):.2f}',
        f'Puntaje promedio Mejorada: {mean(improved_scores):.2f}',
        f'Diferencia promedio Base-Mejorada: {mean(base_diff):.2f}',
        f'IC bootstrap 95% (semilla 2025) para diferencia media de puntaje Base-Mejorada: {score_ci_low:.2f}–{score_ci_high:.2f}.',
        f'Tiempo promedio/decisión Base: {mean(base_times):.6f} s',
        f'Tiempo promedio/decisión Mejorada: {mean(improved_times):.6f} s',
        f'H0: en partidas decisivas, P(Base gana)=P(Mejorada gana)=0.5.',
        f'H1: las probabilidades de victoria son distintas (prueba binomial exacta bilateral).',
        f'n decisivas={decisive}; IC Wilson 95% de victoria Base={100*low:.1f}%–{100*high:.1f}%; p={p_value:.6f}; alfa=0.05.',
        'Interpretación: ' + (
            'se rechaza H0 para el resultado binario en partidas decisivas.'
            if p_value < 0.05 else
            'no se rechaza H0; la diferencia observada no demuestra superioridad.'
        ),
        'Bootstrap de puntajes: el IC de Base-Mejorada no contiene cero, compatible con mayor puntaje medio de MEJORADA en esta muestra; esto no equivale a una diferencia significativa en tasa de victorias ni demuestra superioridad general.',
        'El contraste ignora empates y la tasa de empates se informa por separado.',
    ]
    svg_bar(
        OUTPUT / 'cacho_victorias.svg',
        'Cacho: resultado del torneo',
        ['Partidas'],
        [('Base', [wins['base']], '#3568a8'), ('Mejorada', [wins['mejorada']], '#df8c27'), ('Empates', [wins['empate']], '#777777')],
        'Número de partidas',
    )
    svg_bar(
        OUTPUT / 'cacho_puntajes.svg',
        'Cacho: puntaje promedio',
        ['Puntaje medio'],
        [('Base', [mean(base_scores)], '#3568a8'), ('Mejorada', [mean(improved_scores)], '#df8c27')],
        'Puntos',
    )
    return summary


def analyze_othello(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['profundidad_a'])].append(row)
    base_wins = sum(row['resultado_a'] == 'victoria' for row in rows)
    base_losses = sum(row['resultado_a'] == 'derrota' for row in rows)
    ties = sum(row['resultado_a'] == 'empate' for row in rows)
    decisive = base_wins + base_losses
    low, high = wilson(base_wins, decisive)
    p_value = exact_binomial_two_sided(base_wins, decisive)
    depths = sorted(groups)
    base_success, pos_success = [], []
    base_time, pos_time, base_margin = [], [], []
    detail = ['OTHELLO — Base vs Posicional', f'Partidas completas: {len(rows)}']
    for depth in depths:
        entries = groups[depth]
        score = {
            'base': mean([
                1.0 if r['resultado_a'] == 'victoria'
                else 0.5 if r['resultado_a'] == 'empate' else 0.0
                for r in entries
            ]),
            'posicional': mean([
                1.0 if r['resultado_a'] == 'derrota'
                else 0.5 if r['resultado_a'] == 'empate' else 0.0
                for r in entries
            ]),
        }
        base_success.append(100 * score['base'])
        pos_success.append(100 * score['posicional'])
        base_time.append(mean([float(r['tiempo_promedio_mov_a']) for r in entries]))
        pos_time.append(mean([float(r['tiempo_promedio_mov_b']) for r in entries]))
        base_margin.append(mean([int(r['diferencia_fichas_a']) for r in entries]))
        detail.append(
            f'Profundidad {depth}: {len(entries)} partidas; score Base={100*score["base"]:.1f}%, '
            f'Posicional={100*score["posicional"]:.1f}%; Δfichas Base={base_margin[-1]:.2f}; '
            f'tiempo/mov Base={base_time[-1]:.4f}s, Posicional={pos_time[-1]:.4f}s.'
        )
    low, high = wilson(base_wins, decisive)
    detail.extend([
        f'Victorias Base: {base_wins}; derrotas: {base_losses}; empates: {ties}.',
        'H0: en partidas decisivas, P(Base gana)=P(Posicional gana)=0.5.',
        'H1: las probabilidades de victoria son distintas (prueba binomial exacta bilateral).',
        f'n decisivas={decisive}; IC Wilson 95% Base={100*low:.1f}%–{100*high:.1f}%; p={p_value:.6f}; alfa=0.05.',
        'Con solo dos partidas por profundidad, los resultados por profundidad son descriptivos; no permiten concluir superioridad estadística.',
        'Interpretación global: ' + (
            'se rechaza H0 para el resultado binario en partidas decisivas.'
            if p_value < 0.05 else
            'no se rechaza H0; la diferencia observada no demuestra superioridad.'
        ),
    ])
    svg_bar(
        OUTPUT / 'othello_resultado_profundidad.svg',
        'Othello: porcentaje de puntos por profundidad',
        [str(depth) for depth in depths],
        [('Base', base_success, '#3568a8'), ('Posicional', pos_success, '#df8c27')],
        'Puntos de resultado (%)',
    )
    svg_bar(
        OUTPUT / 'othello_tiempo_profundidad.svg',
        'Othello: tiempo medio por movimiento',
        [str(depth) for depth in depths],
        [('Base', base_time, '#3568a8'), ('Posicional', pos_time, '#df8c27')],
        'Segundos por movimiento',
    )
    svg_bar(
        OUTPUT / 'othello_diferencia_fichas.svg',
        'Othello: diferencia media de fichas',
        [str(depth) for depth in depths],
        [('Base', base_margin, '#3568a8')],
        'Fichas (Base - Posicional)',
    )
    return detail


def main():
    OUTPUT.mkdir(exist_ok=True)
    cacho_rows = read_csv(CACHO_CSV)
    othello_rows = read_csv(OTHELLO_CSV)
    if len(cacho_rows) < 100:
        raise ValueError(f"El CSV Cacho debe contener al menos 100 partidas; contiene {len(cacho_rows)}")
    expected_ohl_depths = {int(row['profundidad_a']) for row in othello_rows}
    if not expected_ohl_depths:
        raise ValueError("El CSV Othello no incluye profundidades")

    summary = analyze_cacho(cacho_rows) + [''] + analyze_othello(othello_rows)
    report = OUTPUT / 'resumen_estadistico.txt'
    report.write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(f"Resumen: {report}")
    print(f"Gráficos SVG: {OUTPUT}")


if __name__ == '__main__':
    main()
