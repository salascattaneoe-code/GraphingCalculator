import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fundamental para Vercel (evita errores de GUI)
import matplotlib.pyplot as plt
import sympy as sp
import warnings
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")

app = Flask(__name__)

def generar_grafico_base64(func_str):
    x_sym = sp.Symbol('x')
    expr = sp.sympify(func_str)
    
    dict_raices = sp.roots(expr, x_sym)
    raices_info = []

    for r, m in dict_raices.items():
        if r.is_real or r.is_real is None:
            try:
                r_float = float(r.evalf())
                if abs(sp.im(r.evalf())) < 1e-10:
                    raices_info.append((r_float, r, m))
            except:
                pass

    raices_info.sort(key=lambda item: item[0])

    intervalos = []
    puntos_prueba = []

    if not raices_info:
        intervalos.append("(-∞; +∞)")
        puntos_prueba.append(0)
    else:
        val_exacto_primero = sp.latex(raices_info[0][1])
        intervalos.append(f"(-∞; ${val_exacto_primero}$)")
        puntos_prueba.append(raices_info[0][0] - 1)

        for i in range(len(raices_info)-1):
            val_exacto_i = sp.latex(raices_info[i][1])
            val_exacto_i_mas_1 = sp.latex(raices_info[i+1][1])
            intervalos.append(f"(${val_exacto_i}$; ${val_exacto_i_mas_1}$)")
            puntos_prueba.append((raices_info[i][0] + raices_info[i+1][0])/2)

        val_exacto_ultimo = sp.latex(raices_info[-1][1])
        intervalos.append(f"(${val_exacto_ultimo}$; +∞)")
        puntos_prueba.append(raices_info[-1][0] + 1)

    f = sp.lambdify(x_sym, expr, 'numpy')
    c_mas, c_menos, signos = [], [], []

    for intervalo, pt in zip(intervalos, puntos_prueba):
        valor = f(pt)
        if valor > 0:
            signos.append('(+) Positivo')
            c_mas.append(intervalo)
        else:
            signos.append('(-) Negativo')
            c_menos.append(intervalo)

    c_mas_str = " U ".join(c_mas) if c_mas else "∅"
    c_menos_str = " U ".join(c_menos) if c_menos else "∅"

    f_0_exact = expr.subs(x_sym, 0)
    f_0_float = float(f_0_exact.evalf())

    if not raices_info:
        x_min, x_max = -5, 5
    elif len(raices_info) == 1:
        x_min, x_max = raices_info[0][0] - 3, raices_info[0][0] + 3
    else:
        r_min, r_max = raices_info[0][0], raices_info[-1][0]
        margen = max((r_max - r_min) * 0.20, 1)
        x_min, x_max = r_min - margen, r_max + margen

    x_vals = np.linspace(x_min, x_max, 2000)
    y_vals = f(x_vals)

    fig = plt.figure(figsize=(11, 11.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[5.5, 1.3, 1.5], hspace=0.4)

    # --- SUBPLOT 1: GRÁFICO ---
    ax_plot = fig.add_subplot(gs[0])
    ax_plot.plot(x_vals, y_vals, 'k-', linewidth=2, label='f(x)', zorder=3)
    ax_plot.fill_between(x_vals, y_vals, 0, where=(y_vals >= 0), facecolor='green', alpha=0.3, label='f(x) > 0', zorder=1)
    ax_plot.fill_between(x_vals, y_vals, 0, where=(y_vals <= 0), facecolor='none', hatch='///', edgecolor='red', label='f(x) < 0', zorder=1)
    ax_plot.axhline(0, color='black', linewidth=1.2, zorder=2)
    ax_plot.axvline(0, color='black', linewidth=1.2, zorder=2)

    if len(raices_info) >= 2:
        x_inner = np.linspace(raices_info[0][0], raices_info[-1][0], 1000)
        y_inner_max = np.max(np.abs(f(x_inner)))
        if y_inner_max > 0:
            ax_plot.set_ylim(-y_inner_max * 1.5, y_inner_max * 1.5)

    ax_plot.grid(True, which='major', linestyle='-', alpha=0.6, zorder=0)

    for r_float, r_exact, _ in raices_info:
        ax_plot.plot(r_float, 0, 'ro', markersize=8, zorder=5)
        ax_plot.annotate(f"$({sp.latex(r_exact)}, 0)$", xy=(r_float, 0), xytext=(0, -10),
                         textcoords='offset points', ha='center', va='top', fontsize=11, 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax_plot.plot(0, f_0_float, 'ro', markersize=8, zorder=5)
    ax_plot.annotate(f"$(0, {sp.latex(f_0_exact)})$", xy=(0, f_0_float), xytext=(10, 5),
                     textcoords='offset points', ha='left', va='bottom', fontsize=11, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax_plot.set_title(f"Gráfico de $f(x) = {sp.latex(expr)}$", fontsize=15)
    ax_plot.text(0.02, 0.05, f"C+ = {c_mas_str}\nC- = {c_menos_str}", transform=ax_plot.transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'), verticalalignment='bottom')

    # --- SUBPLOT 2: BOLZANO ---
    ax_bolz = fig.add_subplot(gs[1])
    ax_bolz.axis('off')
    ax_bolz.set_title("Tabla de Bolzano", loc='center', fontsize=13, fontweight='bold')
    tb1 = ax_bolz.table(cellText=[["Intervalos:"] + intervalos, ["Signo:"] + signos], loc='center', cellLoc='center', bbox=[0.02, 0.0, 0.96, 1.0])
    
    for (row, col), cell in tb1.get_celld().items():
        cell.set_edgecolor('black')
        cell.get_text().set_usetex(False)
        if col == 0: cell.set_text_props(weight='bold', fontsize=12)
        elif row == 0: cell.set_text_props(weight='bold', fontsize=13)
        elif row == 1:
            color = 'green' if '(+)' in signos[col-1] else 'red'
            cell.set_text_props(weight='bold', fontsize=13, color=color)

    # --- SUBPLOT 3: RAÍCES ---
    ax_raices = fig.add_subplot(gs[2])
    ax_raices.axis('off')
    ax_raices.set_title("Raíces:", loc='center', fontsize=13, fontweight='bold')
    datos_raices = [[f"${sp.latex(r_exact)}$", str(m)] for _, r_exact, m in raices_info] if raices_info else [["No hay raíces", "-"]]
    tb2 = ax_raices.table(cellText=datos_raices, colLabels=["Raíz", "Multiplicidad"], loc='center', cellLoc='center', bbox=[0.20, 0.0, 0.60, 1.0])
    
    for (row, col), cell in tb2.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0: cell.set_text_props(weight='bold', fontsize=14)
        else:
            cell.get_text().set_fontsize(13)
            if col == 0: cell.get_text().set_weight('bold')

    plt.tight_layout()
    
    # Exportar a Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'function' not in data:
        return jsonify({'error': 'No se proporcionó ninguna función'}), 400
    
    try:
        img_base64 = generar_grafico_base64(data['function'])
        return jsonify({'success': True, 'image': f"data:image/png;base64,{img_base64}"})
    except Exception as e:
        return jsonify({'success': False, 'error': f"Error al procesar la función: Verifica la sintaxis. (Ej: x**3 - 2*x**2 + 4)"})
