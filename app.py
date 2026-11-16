"""
app.py — Flask Application
WHR 2024: Perbandingan DBSCAN & GMM
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, jsonify, session
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif untuk server
import matplotlib.pyplot as plt

from analysis import run_full_analysis

app = Flask(__name__)
app.secret_key = 'skripsi_amelya_2026' 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'WHR2024.csv')

def get_analysis(eps=None, min_s=None, gmm_n=None, gmm_cov='full'):
    hasil_analisis = run_full_analysis(CSV_PATH, eps, min_s, gmm_n, gmm_cov)
    # Sapu bersih memori grafik global setelah selesai
    plt.close('all')  
    return hasil_analisis

# ── Halaman Utama ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── Dashboard ──────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/run', methods=['POST'])
def api_run():
    """Endpoint utama: jalankan clustering dengan parameter dari form."""
    data = request.get_json(silent=True) or {}
    try:
        eps = float(data.get('eps'))  if data.get('eps')   else None
        min_s = int(data.get('min_s'))  if data.get('min_s') else None
        gmm_n = int(data.get('gmm_n'))  if data.get('gmm_n') else None
        gmm_cov = data.get('gmm_cov', 'full')

        session['eps'] = eps
        session['min_s'] = min_s
        session['gmm_n'] = gmm_n
        session['gmm_cov'] = gmm_cov

    except (ValueError, TypeError):
        return jsonify({'error': 'Parameter tidak valid'}), 400
    
    res = get_analysis(eps, min_s, gmm_n, gmm_cov)
    db  = res['db_res']
    gm  = res['gmm_res']

    return jsonify({
        'dbscan': {
            'eps': db['eps'], 'min_samples': db['min_samples'],
            'n_clusters': db['n_clusters'], 'n_noise': db['n_noise'],
            'silhouette': db['silhouette'],
            'davies_bouldin': db['davies_bouldin'],
            'calinski_harabasz': db['calinski_harabasz'],
        },
        'gmm': {
            'n_components': gm['n_components'],
            'covariance_type': gm['covariance_type'],
            'mean_prob': gm['mean_prob'],
            'silhouette': gm['silhouette'],
            'davies_bouldin': gm['davies_bouldin'],
            'calinski_harabasz': gm['calinski_harabasz'],
        },
        'plots': res['plots'],
        'n_countries': res['n_countries'],
        'median_ladder': res['median_ladder'],
    })

# ── Hasil Akhir ────────────────────────────────────────────
@app.route('/hasil')
def hasil():
    return render_template('hasil.html')

@app.route('/api/hasil')
def api_hasil():
    """Data lengkap untuk halaman Hasil Akhir."""
    eps = session.get('eps', None)
    min_s = session.get('min_s', None)
    gmm_n = session.get('gmm_n', None)
    gmm_cov = session.get('gmm_cov', 'full')

    res = get_analysis(eps, min_s, gmm_n, gmm_cov)
    db  = res['db_res']
    gm  = res['gmm_res']

    # Tentukan pemenang tiap metrik
    winner = {}
    winner['silhouette']       = 'DBSCAN' if (db['silhouette'] or 0) > (gm['silhouette'] or 0) else 'GMM'
    winner['davies_bouldin']   = 'DBSCAN' if (db['davies_bouldin'] or 999) < (gm['davies_bouldin'] or 999) else 'GMM'
    winner['calinski_harabasz']= 'DBSCAN' if (db['calinski_harabasz'] or 0) > (gm['calinski_harabasz'] or 0) else 'GMM'

    return jsonify({
        'dbscan': {
            'eps': db['eps'], 'min_samples': db['min_samples'],
            'n_clusters': db['n_clusters'], 'n_noise': db['n_noise'],
            'silhouette': db['silhouette'],
            'davies_bouldin': db['davies_bouldin'],
            'calinski_harabasz': db['calinski_harabasz'],
        },
        'gmm': {
            'n_components': gm['n_components'],
            'covariance_type': gm['covariance_type'],
            'mean_prob': gm['mean_prob'],
            'silhouette': gm['silhouette'],
            'davies_bouldin': gm['davies_bouldin'],
            'calinski_harabasz': gm['calinski_harabasz'],
        },
        'winner': winner,
        'plots': res['plots'],
        'anomali_summary': res['anomali_summary'],
        'n_countries': res['n_countries'],
    })

# ── Laporan ────────────────────────────────────────────────
@app.route('/laporan')
def laporan():
    return render_template('laporan.html')

@app.route('/api/laporan', methods=['GET'])
def api_laporan():
    """Data lengkap untuk halaman Laporan."""
    eps = session.get('eps', None)
    min_s = session.get('min_s', None)
    gmm_n = session.get('gmm_n', None)
    gmm_cov = session.get('gmm_cov', 'full')

    res = get_analysis(eps, min_s, gmm_n, gmm_cov)
    db  = res['db_res']
    gm  = res['gmm_res']
    cat = res['cat_summary']
    an  = res['anomali_summary']
    return jsonify({
        'dbscan': {
            'eps': db['eps'], 'min_samples': db['min_samples'],
            'n_clusters': db['n_clusters'], 'n_noise': db['n_noise'],
            'silhouette': db['silhouette'],
            'davies_bouldin': db['davies_bouldin'],
            'calinski_harabasz': db['calinski_harabasz'],
        },
        'gmm': {
            'n_components': gm['n_components'],
            'covariance_type': gm['covariance_type'],
            'mean_prob': gm['mean_prob'],
            'silhouette': gm['silhouette'],
            'davies_bouldin': gm['davies_bouldin'],
            'calinski_harabasz': gm['calinski_harabasz'],
        },
        'cat_summary': cat,
        'anomali_summary': an,
        'plots': res['plots'],
        'n_countries': res['n_countries'],
        'median_ladder': res['median_ladder'],
    })

# ── Tentang ────────────────────────────────────────────────
@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

# ── Download CSV ────────────────────────────────────────────
@app.route('/api/download_csv')
def download_csv():
    res = get_analysis()
    import io
    df = res['df']
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    from flask import Response
    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=Hasil_Clustering_WHR2024.csv"}
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False) # Host ambil dari Alamat IP lokal untuk akses dari jaringan yang sama