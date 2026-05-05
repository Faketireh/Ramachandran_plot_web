import streamlit as st
import os
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import colors
from Bio import PDB
import urllib.request
import urllib.error

# ─────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ramachandran Plot Generator",
    layout="wide"
)

# Custom CSS for a clean, premium Antigravity-inspired look
st.markdown("""
    <style>
    /* Clean white background */
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    /* Sidebar styling: subtle background and vertical centering */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        padding: 2rem;
    }

    /* Modern button styling with subtle blue accent */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3.2em;
        background-color: #0066ff;
        color: white;
        border: none;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #0052cc;
    }
    
    .stDownloadButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3.2em;
        background-color: #2da44e;
        color: white;
        border: none;
        font-weight: 500;
    }

    /* Heading typography */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    /* Statistics cards with subtle shadows */
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Contact info styling */
    .contact-info {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e9ecef;
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
    }
    .contact-info b { color: #0066ff; }
    .contact-info a { text-decoration: none; color: inherit; }
    .contact-info a:hover { color: #0066ff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Ramachandran Plot Generator")
st.markdown("Generate RAMPAGE-style Ramachandran plots by entering a PDB ID or uploading a structure file.")

# ─────────────────────────────────────────────
#  CONSTANTS & DATA LOADING
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rama500_data")

RAMA_PREFS = {
    "General": {
        "file": "rama500-general.data",
        "cmap": colors.ListedColormap(['#FFFFFF', '#F1F5F9', '#3B82F6']),
    },
    "GLY": {
        "file": "rama500-gly-sym.data",
        "cmap": colors.ListedColormap(['#FFFFFF', '#F1F5F9', '#60A5FA']),
    },
    "PRO": {
        "file": "rama500-pro.data",
        "cmap": colors.ListedColormap(['#FFFFFF', '#F1F5F9', '#3B82F6']),
    },
    "PRE-PRO": {
        "file": "rama500-prepro.data",
        "cmap": colors.ListedColormap(['#FFFFFF', '#F1F5F9', '#3B82F6']),
    },
}

THRESH_FAV = 0.02
THRESH_ALL = 0.0005

@st.cache_data
def load_rama_grids():
    rama_values = {}
    for key, val in RAMA_PREFS.items():
        grid = np.zeros((360, 360), dtype=np.float64)
        file_path = os.path.join(DATA_DIR, val["file"])
        if not os.path.exists(file_path):
            st.error(f"Missing data file: {val['file']}")
            return None
        with open(file_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                phi_v = int(float(parts[0]))
                psi_v = int(float(parts[1]))
                prob  = float(parts[2])
                grid[psi_v + 180][phi_v + 180] \
                    = grid[psi_v + 179][phi_v + 179] \
                    = grid[psi_v + 179][phi_v + 180] \
                    = grid[psi_v + 180][phi_v + 179] \
                    = prob
        rama_values[key] = grid
    return rama_values

rama_values = load_rama_grids()

# ─────────────────────────────────────────────
#  SIDEBAR - INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Input Options")
    
    input_mode = st.radio("Choose Input Method:", ["PDB ID", "Upload File"])
    
    pdb_file_content = None
    pdb_id = ""
    
    if input_mode == "PDB ID":
        pdb_id = st.text_input("Enter 4-letter PDB ID (e.g., 1UBQ):", max_chars=4).upper()
        if pdb_id:
            if len(pdb_id) == 4:
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                try:
                    with urllib.request.urlopen(url) as response:
                        pdb_file_content = response.read().decode('utf-8')
                    st.success(f"Fetched {pdb_id} from PDB database.")
                except Exception:
                    st.error(f"Failed to download {pdb_id}. Please check the ID or your connection.")
            else:
                st.warning("Please enter a valid 4-character ID.")
    
    else:
        uploaded_file = st.file_uploader("Choose a PDB file", type=["pdb"])
        if uploaded_file:
            pdb_file_content = uploaded_file.read().decode('utf-8')
            pdb_id = uploaded_file.name.split('.')[0].upper()
            st.success(f"Uploaded {uploaded_file.name}")

    # Branding / Contact Info
    st.markdown(f"""
        <div class="contact-info">
            Protein Analysis Project<br>
            By <b>Tirth Patel</b><br>
            <a href="mailto:tirthtirth10@gmail.com">tirthtirth10@gmail.com</a>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CORE LOGIC
# ─────────────────────────────────────────────
if pdb_file_content and rama_values:
    # Save content to a temporary file for Biopython parser
    with io.StringIO(pdb_file_content) as f:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", f)

    # Extract angles
    data = {k: {"phi": [], "psi": [], "labels": []} for k in RAMA_PREFS}
    total_res = 0

    for model in structure:
        for chain in model:
            polys = PDB.PPBuilder().build_peptides(chain)
            for poly in polys:
                res_list = list(poly)
                phi_psi  = poly.get_phi_psi_list()
                for i, residue in enumerate(res_list):
                    phi, psi = phi_psi[i]
                    if phi is None or psi is None: continue
                    
                    res_name = residue.resname.strip()
                    res_num  = residue.id[1]
                    label    = f"{chain.id}{res_num} {res_name[:3]}"

                    if i + 1 < len(res_list) and res_list[i + 1].resname == "PRO":
                        aa_type = "PRE-PRO"
                    elif res_name == "PRO":
                        aa_type = "PRO"
                    elif res_name == "GLY":
                        aa_type = "GLY"
                    else:
                        aa_type = "General"

                    data[aa_type]["phi"].append(math.degrees(phi))
                    data[aa_type]["psi"].append(math.degrees(psi))
                    data[aa_type]["labels"].append(label)
                    total_res += 1

    if total_res == 0:
        st.error("No valid φ/ψ angles found in the structure.")
    else:
        # Classification
        buckets = {
            "fav":     {"gen": {"x": [], "y": []}, "gly": {"x": [], "y": []}},
            "allowed": {"gen": {"x": [], "y": []}, "gly": {"x": [], "y": []}},
            "outlier": {"gen": {"x": [], "y": [], "lbl": []}, "gly": {"x": [], "y": [], "lbl": []}},
        }

        for aa_type, vals in data.items():
            is_gly = (aa_type == "GLY")
            sub = "gly" if is_gly else "gen"
            grid = rama_values[aa_type]

            for phi_v, psi_v, lbl in zip(vals["phi"], vals["psi"], vals["labels"]):
                gi = min(max(int(psi_v) + 180, 0), 359)
                gj = min(max(int(phi_v) + 180, 0), 359)
                prob = grid[gi][gj]

                if prob >= THRESH_FAV:
                    cat = "fav"
                elif prob >= THRESH_ALL:
                    cat = "allowed"
                else:
                    cat = "outlier"

                buckets[cat][sub]["x"].append(phi_v)
                buckets[cat][sub]["y"].append(psi_v)
                if cat == "outlier":
                    buckets[cat][sub]["lbl"].append(lbl)

        # Statistics
        n_fav     = len(buckets["fav"]["gen"]["x"]) + len(buckets["fav"]["gly"]["x"])
        n_allowed = len(buckets["allowed"]["gen"]["x"]) + len(buckets["allowed"]["gly"]["x"])
        n_outlier = len(buckets["outlier"]["gen"]["x"]) + len(buckets["outlier"]["gly"]["x"])
        
        pct_fav = 100 * n_fav / total_res
        pct_allowed = 100 * n_allowed / total_res
        pct_outlier = 100 * n_outlier / total_res

        # UI Layout
        col1, col2 = st.columns([2, 1])

        with col1:
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
            ax.set_facecolor('white')
            
            # Text and label colors for light mode
            label_color = '#1e293b'
            grid_color = '#e2e8f0'
            def make_region_grid(aa_type):
                grid = rama_values[aa_type]
                region = np.zeros_like(grid, dtype=np.int8)
                region[grid >= THRESH_ALL] = 1
                region[grid >= THRESH_FAV] = 2
                return region.astype(float)

            bg_bounds = [0, 0.5, 1.5, 2.5]
            
            ax.imshow(make_region_grid("General"), cmap=RAMA_PREFS["General"]["cmap"],
                      norm=colors.BoundaryNorm(bg_bounds, 3), extent=(-180, 180, 180, -180),
                      aspect='auto', alpha=1.0, zorder=0)
            
            ax.imshow(make_region_grid("GLY"), cmap=RAMA_PREFS["GLY"]["cmap"],
                      norm=colors.BoundaryNorm(bg_bounds, 3), extent=(-180, 180, 180, -180),
                      aspect='auto', alpha=0.6, zorder=1)

            # Grid lines
            for v in range(-180, 181, 45):
                ax.axhline(v, color=grid_color, lw=0.5, ls='--', zorder=2)
                ax.axvline(v, color=grid_color, lw=0.5, ls='--', zorder=2)
            ax.axhline(0, color='#94a3b8', lw=1.0, zorder=2)
            ax.axvline(0, color='#94a3b8', lw=1.0, zorder=2)

            # Scatter
            ax.scatter(buckets["fav"]["gen"]["x"], buckets["fav"]["gen"]["y"], marker='s', s=14, color='#1e293b', alpha=0.8, zorder=5)
            ax.scatter(buckets["fav"]["gly"]["x"], buckets["fav"]["gly"]["y"], marker='s', s=14, color='#ea580c', alpha=0.8, zorder=5)
            ax.scatter(buckets["allowed"]["gen"]["x"], buckets["allowed"]["gen"]["y"], marker='^', s=16, color='#475569', alpha=0.6, zorder=5)
            ax.scatter(buckets["allowed"]["gly"]["x"], buckets["allowed"]["gly"]["y"], marker='^', s=16, color='#f97316', alpha=0.6, zorder=5)

            for sub in ["gen", "gly"]:
                for xv, yv, lbl in zip(buckets["outlier"][sub]["x"], buckets["outlier"][sub]["y"], buckets["outlier"][sub]["lbl"]):
                    ax.plot(xv, yv, 'x', ms=8, color='#cc0000', mew=2.0, zorder=8)
                    ax.annotate(lbl, (xv, yv), textcoords="offset points", xytext=(5, 5), fontsize=7, color='#cc0000', fontweight='bold', zorder=9)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_xlabel("φ (degrees)", color=label_color)
            ax.set_ylabel("ψ (degrees)", color=label_color)
            ax.tick_params(axis='both', colors=label_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(label_color)
                
            ax.set_title(f"Ramachandran Plot - {pdb_id}", color=label_color, fontweight='bold', pad=15)
            
            st.pyplot(fig)

            # Download button
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            st.download_button(
                label="📥 Download Plot as PNG",
                data=buf.getvalue(),
                file_name=f"{pdb_id}_ramachandran.png",
                mime="image/png"
            )

        with col2:
            st.subheader("Statistical Summary")
            st.metric("Total Residues", total_res)
            
            st.markdown("---")
            st.write(f"**Favoured Region:** {n_fav} ({pct_fav:.1f}%)")
            st.progress(pct_fav / 100)
            
            st.write(f"**Allowed Region:** {n_allowed} ({pct_allowed:.1f}%)")
            st.progress(pct_allowed / 100)
            
            st.write(f"**Outliers:** {n_outlier} ({pct_outlier:.1f}%)")
            st.progress(pct_outlier / 100)

            st.markdown("---")
            st.info("""
            **Thresholds:**
            - Favoured: > 2%
            - Allowed: > 0.05%
            - Outlier: < 0.05%
            """)

else:
    st.info("👈 Please enter a PDB ID or upload a file in the sidebar to begin.")

st.markdown("---")
st.caption("Developed for structural biology analysis. Data source: Top500 high-resolution protein set.")
