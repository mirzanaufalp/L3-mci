from flask import Flask, render_template_string, request, send_file, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import io
import math
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>MCI Assay-Constrained Processor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f4f7f6; color: #333; }
        .container { max-width: 900px; margin: auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #1F4E78; text-align: center; }
        .form-group { margin-bottom: 20px; }
        label { font-weight: bold; display: block; margin-bottom: 5px; }
        input[type="file"], input[type="submit"] { padding: 10px; border-radius: 5px; border: 1px solid #ccc; width: 100%; box-sizing: border-box; }
        input[type="submit"] { background-color: #1F4E78; color: white; border: none; font-size: 16px; cursor: pointer; margin-top: 10px; }
        input[type="submit"]:hover { background-color: #153754; }
        .result-img { text-align: center; margin-top: 20px; }
        .result-img img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
        .btn-download { display: inline-block; padding: 10px 20px; background-color: #27ae60; color: white; text-decoration: none; border-radius: 5px; margin-top: 15px; }
    </style>
</head>
<body>
<div class="container">
    <h1>⚡ Resistivity -> MCI Processor</h1>
    <p>Upload file Excel (<code>L3.xlsx</code>) dengan sheet <strong>res_L3</strong> dan <strong>Assay L3</strong>.</p>
    <form method="POST" enctype="multipart/form-data">
        <div class="form-group">
            <label for="file">Pilih File Excel (.xlsx):</label>
            <input type="file" id="file" name="file" accept=".xlsx" required>
        </div>
        <input type="submit" value="Proses Data MCI & Generate Penampang">
    </form>

    {% if img_data %}
    <hr style="margin-top: 30px;">
    <h2>🖼️ Penampang Profil MCI</h2>
    <div class="result-img">
        <img src="data:image/png;base64,{{ img_data }}" alt="Penampang MCI">
    </div>
    {% endif %}
</div>
</body>
</html>
'''

ZONE_MCI = {"tp": 10.0, "soft": 30.0, "hard cluster": 65.0, "hard": 90.0}
H_SCALE_M, V_SCALE_M = 15.0, 0.40
MAX_ASSAY_WEIGHT, MIN_EFFECTIVE_WEIGHT = 0.95, 0.01

def normalize_name(value):
    if pd.isna(value): return ""
    return " ".join(str(value).strip().lower().split())

def compute_profile_axis(x, y):
    xy = np.column_stack([x, y]).astype(float)
    center = np.nanmean(xy, axis=0)
    good = np.isfinite(xy).all(axis=1)
    xy0 = xy[good] - center
    _, _, vh = np.linalg.svd(xy0, full_matrices=False)
    axis = vh[0].astype(float)
    if abs(axis[0]) >= abs(axis[1]):
        if axis[0] < 0: axis *= -1
    else:
        if axis[1] < 0: axis *= -1
    s_raw = (xy - center) @ axis
    return center, axis, np.nanmin(s_raw), s_raw - np.nanmin(s_raw)

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                res = pd.read_excel(file, sheet_name="res_L3")
                file.seek(0)
                assay_raw = pd.read_excel(file, sheet_name="Assay L3")
                
                for col in ["X", "Y", "Z", "RES"]:
                    res[col] = pd.to_numeric(res[col], errors="coerce")
                    
                center, axis, s0, s_res = compute_profile_axis(res["X"].to_numpy(), res["Y"].to_numpy())
                res["Profile_m"] = s_res
                
                # Baseline MCI
                arr = res["RES"].to_numpy(dtype=float)
                valid = np.isfinite(arr) & (arr > 0)
                p_low, p_high = float(np.nanpercentile(arr[valid], 5.0)), float(np.nanpercentile(arr[valid], 95.0))
                mci_base = np.clip(100.0 * ((np.log10(arr) - math.log10(p_low)) / (math.log10(p_high) - math.log10(p_low))), 0, 100)
                res["MCI_RES_Base"] = mci_base
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 5))
                xg, zg, mg = res["Profile_m"].to_numpy(), res["Z"].to_numpy(), mci_base
                good = np.isfinite(xg) & np.isfinite(zg) & np.isfinite(mg)
                
                triang = mtri.Triangulation(xg[good], zg[good])
                cntr = ax.tricontourf(triang, mg[good], levels=np.linspace(0, 100, 21))
                fig.colorbar(cntr, ax=ax, label="MCI (0-100)")
                ax.set_xlabel("Jarak Profil (m)")
                ax.set_ylabel("Elevasi Z (m)")
                ax.grid(alpha=0.3)
                
                import base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)
                
    return render_template_string(HTML_TEMPLATE, img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)