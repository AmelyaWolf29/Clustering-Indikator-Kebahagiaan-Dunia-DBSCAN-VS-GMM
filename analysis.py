"""
analysis.py — Engine clustering DBSCAN & GMM untuk WHR 2024
Dapat dipanggil oleh Flask routes sebagai modul.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

import base64, io, itertools, warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'Log GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def kategorisasi_gmm_univariat(df, kolom, lbl_rendah, lbl_sedang, lbl_tinggi):
    sc = StandardScaler()
    X1 = sc.fit_transform(df[[kolom]])
    g  = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    g.fit(X1)
    lbl = g.predict(X1)
    means = df.groupby(lbl)[kolom].mean().sort_values()
    idx   = means.index.tolist()
    return pd.Series(lbl).map({idx[0]: lbl_rendah, idx[1]: lbl_sedang, idx[2]: lbl_tinggi}).values

def evaluate(X_scaled, labels):
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return None, None, None
    Xm, lm = X_scaled[mask], labels[mask]
    return (round(silhouette_score(Xm, lm), 4),
            round(davies_bouldin_score(Xm, lm), 4),
            round(calinski_harabasz_score(Xm, lm), 4))

# ─────────────────────────────────────────────
# Load Dataset & Preprocessing Data
# ─────────────────────────────────────────────
def load_data(csv_path='WHR2024.csv'):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df

def preprocess(df):
    X = df[FEATURES].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return np.array(X_scaled, dtype=np.float64), scaler

# ─────────────────────────────────────────────
# DBSCAN 
# ─────────────────────────────────────────────
def run_dbscan(X_scaled, eps=None, min_samples=None, auto_tune=False):
    if eps is None or min_samples is None:
        best_sil = -1; best_eps = 1.3; best_min = 6
        for e, ms in itertools.product(np.arange(0.5, 2.1, 0.1), range(2, 11)):
            db = DBSCAN(eps=round(e, 1), min_samples=ms)
            lbl = db.fit_predict(X_scaled)
            n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
            if n_cl >= 2:
                sil = silhouette_score(X_scaled, lbl)
                if sil > best_sil:
                    best_sil = sil; best_eps = round(e, 1); best_min = ms
        eps, min_samples = best_eps, best_min

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    sil, dbi, chi = evaluate(X_scaled, labels)
    return {
        'labels': labels, 'eps': eps, 'min_samples': min_samples,
        'n_clusters': n_clusters, 'n_noise': n_noise,
        'silhouette': sil, 'davies_bouldin': dbi, 'calinski_harabasz': chi
    }

# ─────────────────────────────────────────────
# GMM
# ─────────────────────────────────────────────
def run_gmm(X_scaled, n_components=None, covariance_type='full'):
    # Hitung skor BIC setiap kali tombol Proses ditekan
    # (Agar kurva BIC di web selalu punya gambar)
    bics = []
    for n in range(2, 11):
        g = GaussianMixture(n, covariance_type=covariance_type, random_state=42)
        g.fit(X_scaled)
        bics.append(g.bic(X_scaled))
        
    # Jika web tidak mengirim angka dari parameter tersebut, pilih otomatis yang paling optimal
    if n_components is None or n_components == "":
        n_components = range(2, 11)[np.argmin(bics)]
        
    # Simpan variabel bics ke dalam variabel bic_scores untuk digambar
    bic_scores = bics

    gmm = GaussianMixture(n_components=int(n_components), covariance_type=covariance_type, random_state=42)
    gmm.fit(X_scaled)
    labels   = gmm.predict(X_scaled)
    prob_max = gmm.predict_proba(X_scaled).max(axis=1)
    sil, dbi, chi = evaluate(X_scaled, labels)
    
    return {
        'labels': labels, 'n_components': n_components,
        'covariance_type': covariance_type,
        'prob_max': prob_max, 'mean_prob': round(float(prob_max.mean()), 4),
        'silhouette': sil, 'davies_bouldin': dbi, 'calinski_harabasz': chi,
        'bic_scores': bic_scores
    }

# ─────────────────────────────────────────────
# Kategorisasi dan Analisis Anomali
# ─────────────────────────────────────────────
def build_categories(df):
    df = df.copy()
    df['GDP_Kategori']        = kategorisasi_gmm_univariat(df,'Log GDP per capita','Miskin','Menengah','Kaya')
    df['Sosial_Kategori']     = kategorisasi_gmm_univariat(df,'Social support','Sosial Rendah','Sosial Sedang','Sosial Tinggi')
    df['Healthy_Kategori']    = kategorisasi_gmm_univariat(df,'Healthy life expectancy','Sehat Rendah','Sehat Sedang','Sehat Tinggi')
    df['Freedom_Kategori']    = kategorisasi_gmm_univariat(df,'Freedom to make life choices','Bebas Rendah','Bebas Sedang','Bebas Tinggi')
    df['Generosity_Kategori'] = kategorisasi_gmm_univariat(df,'Generosity','Kurang Dermawan','Netral','Dermawan')
    df['Corruption_Kategori'] = kategorisasi_gmm_univariat(df,'Perceptions of corruption','Tinggi Korupsi','Korupsi Sedang','Rendah Korupsi')

    med = df['Ladder score'].median()
    df['Anomali_GDP'] = 'Normal'
    df.loc[(df['GDP_Kategori']=='Miskin')&(df['Ladder score']>med),'Anomali_GDP'] = 'Miskin tapi Senang'
    df.loc[(df['GDP_Kategori']=='Kaya')  &(df['Ladder score']<med),'Anomali_GDP'] = 'Kaya tapi Sedih'

    df['Anomali_Korupsi'] = 'Normal'
    df.loc[(df['Corruption_Kategori']=='Tinggi Korupsi')&(df['Ladder score']>med),'Anomali_Korupsi'] = 'Korupsi Tinggi tapi Bahagia'
    df.loc[(df['Corruption_Kategori']=='Rendah Korupsi')&(df['Ladder score']<med),'Anomali_Korupsi'] = 'Bersih tapi Tidak Bahagia'

    df['Anomali_Generosity'] = 'Normal'
    df.loc[(df['GDP_Kategori']=='Miskin')&(df['Generosity_Kategori']=='Dermawan'),'Anomali_Generosity'] = 'Miskin tapi Dermawan'
    df.loc[(df['GDP_Kategori']=='Kaya')  &(df['Generosity_Kategori']=='Kurang Dermawan'),'Anomali_Generosity'] = 'Kaya tapi Pelit'
    return df

# ─────────────────────────────────────────────
# Visualisasi Data
# ─────────────────────────────────────────────
CLUSTER_COLORS = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800','#00BCD4']
NOISE_COLOR    = '#BDBDBD'

def plot_clusters_side_by_side(X_scaled, db_labels, gmm_labels, country_names, pca=None):
    if pca is None:
        pca = PCA(n_components=2, random_state=42)
        X2d = pca.fit_transform(X_scaled)
    else:
        X2d = pca.transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, labels, title in zip(axes, [db_labels, gmm_labels], ['DBSCAN','GMM']):
        unique = sorted(set(labels))
        ci = 0
        for lab in unique:
            mask = labels == lab
            if lab == -1:
                color, lname, marker, alpha = NOISE_COLOR, 'Noise/Outlier', 'x', 0.5
            else:
                color, lname, marker, alpha = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)], f'Cluster {lab}', 'o', 0.85
                ci += 1
            ax.scatter(X2d[mask,0], X2d[mask,1], c=color, label=lname,
                       marker=marker, s=55, alpha=alpha, edgecolors='k' if lab!=-1 else 'none', linewidths=0.4)

        var1 = pca.explained_variance_ratio_[0]*100
        var2 = pca.explained_variance_ratio_[1]*100
        ax.set_title(f'Cluster {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({var1:.1f}%)')
        ax.set_ylabel(f'PC2 ({var2:.1f}%)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)

    plt.suptitle('Perbandingan Cluster DBSCAN vs GMM (Reduksi PCA 2D)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_evaluation_bars(db_res, gmm_res):
    metrics  = ['Silhouette Score\n(↑ lebih baik)', 'Davies-Bouldin Index\n(↓ lebih baik)', 'Calinski-Harabasz\n(↑ lebih baik)']
    sil_db = db_res['silhouette'] or 0
    dbi_db = db_res['davies_bouldin'] or 0
    chi_db = db_res['calinski_harabasz'] or 0
    
    sil_gm = gmm_res['silhouette'] or 0
    dbi_gm = gmm_res['davies_bouldin'] or 0
    chi_gm = gmm_res['calinski_harabasz'] or 0

    db_vals  = [sil_db, dbi_db, chi_db]
    gm_vals  = [sil_gm, dbi_gm, chi_gm]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, metric, dv, gv in zip(axes, metrics, db_vals, gm_vals):
        bars = ax.bar(['DBSCAN','GMM'], [dv, gv],
                      color=['#2196F3','#FF5722'], edgecolor='black', linewidth=0.8, width=0.5)
        for bar, val in zip(bars, [dv, gv]):
            # max(dv, gv) sekarang aman karena tidak mungkin berisi None
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(dv,gv)*0.03,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(metric, fontsize=10, fontweight='bold')
        ax.set_ylabel('Nilai')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Perbandingan Metrik Evaluasi Clustering', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_anomali(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    med = df['Ladder score'].median()

    # GDP vs Kebahagiaan
    ax = axes[0]
    cmap = {'Normal':'#90CAF9','Miskin tapi Senang':'#4CAF50','Kaya tapi Sedih':'#F44336'}
    for t, c in cmap.items():
        mask = df['Anomali_GDP'] == t
        ax.scatter(df.loc[mask,'Log GDP per capita'], df.loc[mask,'Ladder score'],
                   c=c, label=t, alpha=0.8, s=55, edgecolors='k', linewidths=0.4)
    ax.axhline(y=med, color='gray', linestyle='--', linewidth=1.2, label=f'Median ({med:.2f})')
    for _, row in df[df['Anomali_GDP']!='Normal'].iterrows():
        ax.annotate(row['Country name'], (row['Log GDP per capita'], row['Ladder score']),
                    fontsize=6.5, xytext=(3,3), textcoords='offset points')
    ax.set_xlabel('Log GDP per Capita'); ax.set_ylabel('Ladder Score')
    ax.set_title('Anomali: GDP vs Kebahagiaan', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # Korupsi vs Kebahagiaan
    ax2 = axes[1]
    cmap2 = {'Normal':'#90CAF9','Korupsi Tinggi tapi Bahagia':'#FF9800','Bersih tapi Tidak Bahagia':'#9C27B0'}
    for t, c in cmap2.items():
        mask = df['Anomali_Korupsi'] == t
        ax2.scatter(df.loc[mask,'Perceptions of corruption'], df.loc[mask,'Ladder score'],
                    c=c, label=t, alpha=0.8, s=55, edgecolors='k', linewidths=0.4)
    ax2.axhline(y=med, color='gray', linestyle='--', linewidth=1.2)
    for _, row in df[df['Anomali_Korupsi']!='Normal'].iterrows():
        ax2.annotate(row['Country name'], (row['Perceptions of corruption'], row['Ladder score']),
                     fontsize=6.5, xytext=(3,3), textcoords='offset points')
    ax2.set_xlabel('Perceptions of Corruption'); ax2.set_ylabel('Ladder Score')
    ax2.set_title('Anomali: Korupsi vs Kebahagiaan', fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    plt.suptitle('Analisis Anomali Indikator Kebahagiaan Dunia', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_bic(bic_scores, best_n):
    if not bic_scores:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(2, 11), bic_scores, 'bo-', linewidth=2, markersize=7)
    ax.axvline(x=best_n, color='red', linestyle='--', linewidth=2, label=f'Optimal n={best_n}')
    ax.set_xlabel('Jumlah Komponen'); ax.set_ylabel('BIC Score')
    ax.set_title('Pemilihan n_components GMM via BIC', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

# ─────────────────────────────────────────────
# Main Pipeline — dipanggil oleh Flask
# ─────────────────────────────────────────────
def run_full_analysis(csv_path='WHR2024.csv',
                      dbscan_eps=None, dbscan_min=None,
                      gmm_n=None, gmm_cov='full'):
    df = load_data(csv_path)
    X_scaled, scaler  = preprocess(df)

    auto = (dbscan_eps is None or dbscan_min is None)
    db_res  = run_dbscan(X_scaled, dbscan_eps, dbscan_min, auto_tune=auto)
    gmm_res = run_gmm(X_scaled, gmm_n, gmm_cov)

    df['Cluster_DBSCAN'] = db_res['labels']
    df['Cluster_GMM']    = gmm_res['labels']
    df['GMM_Prob_Max']   = gmm_res['prob_max']
    df = build_categories(df)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)

    plots = {
        'cluster_compare' : plot_clusters_side_by_side(X_scaled, db_res['labels'], gmm_res['labels'],
                                                        df['Country name'].tolist(), pca),
        'eval_bars'        : plot_evaluation_bars(db_res, gmm_res),
        'anomali'          : plot_anomali(df),
        'bic_curve'        : plot_bic(gmm_res['bic_scores'], gmm_res['n_components']),
    }

    # Top 10 per kategori
    cat_summary = {}
    for col, cats in [
        ('GDP_Kategori',        ['Miskin','Menengah','Kaya']),
        ('Sosial_Kategori',     ['Sosial Rendah','Sosial Sedang','Sosial Tinggi']),
        ('Healthy_Kategori',    ['Sehat Rendah','Sehat Sedang','Sehat Tinggi']),
        ('Freedom_Kategori',    ['Bebas Rendah','Bebas Sedang','Bebas Tinggi']),
        ('Generosity_Kategori', ['Kurang Dermawan','Netral','Dermawan']),
        ('Corruption_Kategori', ['Tinggi Korupsi','Korupsi Sedang','Rendah Korupsi']),
    ]:
        cat_summary[col] = {}
        for kat in cats:
            sub = df[df[col] == kat].sort_values('Ladder score', ascending=False)
            cat_summary[col][kat] = sub[['Country name','Ladder score'] + FEATURES].head(10).to_dict('records')

    anomali_summary = {}
    for col, types in [
        ('Anomali_GDP', ['Miskin tapi Senang','Kaya tapi Sedih']),
        ('Anomali_Korupsi', ['Korupsi Tinggi tapi Bahagia','Bersih tapi Tidak Bahagia']),
        ('Anomali_Generosity', ['Miskin tapi Dermawan','Kaya tapi Pelit']),
    ]:
        anomali_summary[col] = {}
        for t in types:
            sub = df[df[col] == t]
            anomali_summary[col][t] = sub[['Country name','Ladder score','Log GDP per capita',
                                           'Perceptions of corruption','Generosity']].to_dict('records')

    return {
        'df': df,
        'db_res': db_res,
        'gmm_res': gmm_res,
        'plots': plots,
        'cat_summary': cat_summary,
        'anomali_summary': anomali_summary,
        'n_countries': len(df),
        'median_ladder': round(df['Ladder score'].median(), 3),
    }
