import streamlit as st
import streamlit.components.v1 as components
import os, io, math, base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import colors
from Bio import PDB
import urllib.request
import plotly.graph_objects as go

st.set_page_config(page_title="Ramachandran Plot Generator", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
:root { --primary:#0066ff; --primary-soft:rgba(0,102,255,0.06); --border:#e2e8f0; --muted:#64748b; }
* { font-family:'Outfit',sans-serif; }
.stApp { background:#fff; }
[data-testid="stSidebar"] {
    background:#f8f9fa; border-right:1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    display:flex; flex-direction:column; justify-content:center;
    min-height:100vh; padding:2rem 1rem; gap:0.5rem;
}
.stButton>button { width:100%; border-radius:10px; height:3.2em; background:var(--primary); color:white; border:none; font-weight:500; transition:all 0.2s; }
.stButton>button:hover { background:#0052cc; transform:translateY(-1px); }
.metric-card { background:#fff; border:1px solid var(--border); border-radius:16px; padding:1.25rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom:1rem; }
.metric-label { font-size:0.85rem; color:var(--muted); font-weight:500; text-transform:uppercase; margin-bottom:0.5rem; }
.metric-value { font-size:2rem; color:var(--primary); font-weight:700; line-height:1; }
.metric-sub { font-size:0.85rem; color:var(--muted); margin-top:0.5rem; }
.hero-wrap { display:flex; justify-content:center; align-items:center; min-height:80vh; }
.hero-section { padding:5rem 4rem; text-align:center; background:linear-gradient(160deg,var(--primary-soft) 0%,transparent 80%); border-radius:40px; max-width:900px; width:100%; border:1px solid var(--border); }
.hero-title { font-size:6rem; font-weight:800; color:#1e293b; margin-bottom:1.5rem; letter-spacing:-0.04em; line-height:1; }
.hero-sub { font-size:1.4rem; color:var(--muted); line-height:1.7; max-width:700px; margin:0 auto 2rem; }
.hero-chips { display:flex; gap:0.75rem; justify-content:center; flex-wrap:wrap; }
.chip { background:white; border:1px solid var(--border); border-radius:99px; padding:0.5rem 1.25rem; font-size:0.9rem; color:var(--muted); }
.contact-info { padding:1.5rem; background:var(--primary-soft); border-radius:16px; text-align:center; font-size:0.85rem; color:var(--muted); }
.js-plotly-plot .plotly .modebar-btn svg { width:20px!important; height:20px!important; }
</style>
""", unsafe_allow_html=True)

# ── COLORS ─────────────────────────────────────────────────────────────────
C_GEN_FAV="#1e88e5"; C_GEN_ALL="#90caf9"
C_GLY_FAV="#fb8c00"; C_GLY_ALL="#ffe082"
C_PT_GEN="black";  C_PT_GLY="#d37211"; C_OUTLIER="red"

# ── DATA ───────────────────────────────────────────────────────────────────
def get_data_path():
    b=os.path.dirname(os.path.abspath(__file__))
    p=os.path.join(b,"rama500_data")
    return p if os.path.exists(p) else os.path.join(os.path.dirname(b),"rama500_data")

DATA_DIR=get_data_path()
RAMA_FILES={"General":"rama500-general.data","GLY":"rama500-gly-sym.data",
             "PRO":"rama500-pro.data","PRE-PRO":"rama500-prepro.data"}
THRESH_FAV,THRESH_ALL=0.02,0.0005

@st.cache_data
def load_grids():
    out={}
    for k,fname in RAMA_FILES.items():
        g=np.zeros((360,360)); fp=os.path.join(DATA_DIR,fname)
        if not os.path.exists(fp): return None
        with open(fp) as f:
            for line in f:
                if line.startswith("#"): continue
                p=line.split(); phi,psi,prob=int(float(p[0])),int(float(p[1])),float(p[2])
                g[psi+180][phi+180]=g[psi+179][phi+179]=g[psi+179][phi+180]=g[psi+180][phi+179]=prob
        out[k]=g
    return out
grids=load_grids()

# ── STATE MANAGEMENT ───────────────────────────────────────────────────────
if 'pdb_content' not in st.session_state:
    st.session_state.pdb_content = None
    st.session_state.pdb_id = ""

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Input Options")
    mode=st.radio("Choose Input Method:",["PDB ID","Upload File"])
    
    if mode=="PDB ID":
        pdb_input=st.text_input("Enter 4-letter PDB ID:",max_chars=4).upper()
        if st.button("Fetch PDB"):
            if pdb_input and len(pdb_input)==4:
                try:
                    with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb_input}.pdb") as r:
                        st.session_state.pdb_content=r.read().decode()
                        st.session_state.pdb_id=pdb_input
                    st.success(f"Fetched {pdb_input}")
                except: st.error("Fetch failed.")
    else:
        uf=st.file_uploader("Choose PDB file",type=["pdb"])
        if uf:
            st.session_state.pdb_content=uf.read().decode()
            st.session_state.pdb_id=uf.name.split('.')[0].upper()
            st.success("Uploaded")
            
    st.markdown("---")
    if st.button("Try with Demo Protein (3D11)"):
        try:
            with urllib.request.urlopen("https://files.rcsb.org/download/3D11.pdb") as r:
                st.session_state.pdb_content=r.read().decode()
                st.session_state.pdb_id="3D11"
            st.success("Loaded 3D11")
        except: st.error("Failed.")
    st.markdown('<div class="contact-info">Protein Analysis Project<br>By <b>Tirth Patel</b></div>',unsafe_allow_html=True)

pdb_content = st.session_state.pdb_content
pdb_id = st.session_state.pdb_id

# ── HERO ───────────────────────────────────────────────────────────────────
if not pdb_content:
    st.markdown("""
    <div class="hero-wrap"><div class="hero-section">
      <div class="hero-title">RamaPlot</div>
      <p class="hero-sub">High-resolution Ramachandran plot generator using the RAMPAGE-standard Top500 dataset. Visualize and validate protein backbone geometry with interactive precision.</p>
      <div class="hero-chips">
        <span class="chip">Top500 Dataset</span>
        <span class="chip">RAMPAGE Standard</span>
        <span class="chip">Interactive Plotly</span>
        <span class="chip">300 DPI Export</span>
      </div>
    </div></div>""", unsafe_allow_html=True)
    st.stop()

# ── PARSE ──────────────────────────────────────────────────────────────────
with io.StringIO(pdb_content) as f:
    structure=PDB.PDBParser(QUIET=True).get_structure("p",f)
aa_data={k:{"phi":[],"psi":[],"lbl":[]} for k in RAMA_FILES}
total=0
for model in structure:
    for chain in model:
        for poly in PDB.PPBuilder().build_peptides(chain):
            rlist,pp=list(poly),poly.get_phi_psi_list()
            for i,res in enumerate(rlist):
                phi,psi=pp[i]
                if phi is None or psi is None: continue
                lbl=f"{chain.id}{res.id[1]} {res.resname.strip()}"
                if i+1<len(rlist) and rlist[i+1].resname=="PRO": t="PRE-PRO"
                elif res.resname=="PRO": t="PRO"
                elif res.resname=="GLY": t="GLY"
                else: t="General"
                aa_data[t]["phi"].append(math.degrees(phi))
                aa_data[t]["psi"].append(math.degrees(psi))
                aa_data[t]["lbl"].append(lbl); total+=1

if total==0: st.error("No valid phi/psi angles found."); st.stop()

# ── CLASSIFY ───────────────────────────────────────────────────────────────
bkt={c:{s:{"x":[],"y":[],"lbl":[]} for s in ["gen","gly"]} for c in ["fav","allowed","outlier"]}
for atype,vals in aa_data.items():
    sub="gly" if atype=="GLY" else "gen"; g=grids[atype]
    for phi,psi,lbl in zip(vals["phi"],vals["psi"],vals["lbl"]):
        gi,gj=min(max(int(psi)+180,0),359),min(max(int(phi)+180,0),359)
        prob=g[gi][gj]
        cat="fav" if prob>=THRESH_FAV else("allowed" if prob>=THRESH_ALL else "outlier")
        bkt[cat][sub]["x"].append(phi); bkt[cat][sub]["y"].append(psi)
        bkt[cat][sub]["lbl"].append(lbl)

nfav=len(bkt["fav"]["gen"]["x"])+len(bkt["fav"]["gly"]["x"])
nall=len(bkt["allowed"]["gen"]["x"])+len(bkt["allowed"]["gly"]["x"])
nout=len(bkt["outlier"]["gen"]["x"])+len(bkt["outlier"]["gly"]["x"])
pfav,pall,pout=100*nfav/total,100*nall/total,100*nout/total

col1,col2=st.columns([2,1])
with col1:
    st.markdown(f"### Ramachandran Plot — {pdb_id}")
    xs=np.linspace(-180,180,360)
    fig=go.Figure()

    # Regions
    for key,c_fav,c_all,op in [("General",C_GEN_FAV,C_GEN_ALL,1.0),("GLY",C_GLY_FAV,C_GLY_ALL,0.5)]:
        z=np.zeros((360,360)); g=grids[key]
        z[g>=THRESH_ALL]=1; z[g>=THRESH_FAV]=2
        fig.add_trace(go.Heatmap(z=z,x=xs,y=xs,
            colorscale=[[0,"white"],[0.5,c_all],[1,c_fav]],
            showscale=False,hoverinfo='skip',opacity=op,zmin=0,zmax=2))

    # Grid
    for v in range(-180,181,45):
        kw=dict(color="#111",width=1.5) if v==0 else dict(color="#888",width=0.5,dash="dot")
        fig.add_shape(type="line",x0=v,y0=-180,x1=v,y1=180,line=kw)
        fig.add_shape(type="line",x0=-180,y0=v,x1=180,y1=v,line=kw)
    fig.add_shape(type="rect",x0=-180,y0=-180,x1=180,y1=180,line=dict(color="#111",width=2))

    # Legend: region colors (dummy squares with fill)
    for name,col in [
        ("Favoured Region (General)", C_GEN_FAV),
        ("Allowed Region (General)",  C_GEN_ALL),
        ("Favoured Region (Gly)",     C_GLY_FAV),
        ("Allowed Region (Gly)",      C_GLY_ALL),
    ]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',name=name,
            marker=dict(symbol='square',size=14,color=col,opacity=0.9,line=dict(color="#555",width=0.5))))

    # Legend: point types
    for name,sym,col in [
        ("General/Pre-Pro/Proline Favoured","square",     C_PT_GEN),
        ("General/Pre-Pro/Proline Allowed", "triangle-up",C_PT_GEN),
        ("Glycine Favoured",                "square",     C_PT_GLY),
        ("Glycine Allowed",                 "triangle-up",C_PT_GLY),
        ("Outlier",                         "x",          C_OUTLIER),
    ]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',name=name,
            marker=dict(symbol=sym,size=11,color=col,line=dict(color=col,width=2) if sym=="x" else dict(width=0))))

    # Data
    for cat,sub,sym,col,sz in [
        ("fav","gen","square",C_PT_GEN,4),("fav","gly","square",C_PT_GLY,4),
        ("allowed","gen","triangle-up",C_PT_GEN,4),("allowed","gly","triangle-up",C_PT_GLY,4),
        ("outlier","gen","x",C_OUTLIER,6),("outlier","gly","x",C_OUTLIER,6),
    ]:
        if bkt[cat][sub]["x"]:
            fig.add_trace(go.Scatter(x=bkt[cat][sub]["x"],y=bkt[cat][sub]["y"],
                mode='markers',showlegend=False,
                marker=dict(symbol=sym,size=sz,color=col,
                    line=dict(color=col,width=1.8) if sym=="x" else dict(width=0)),
                text=bkt[cat][sub]["lbl"],
                hovertemplate="%{text}<br>φ=%{x:.1f}°, ψ=%{y:.1f}°<extra></extra>"))

    fig.update_layout(
        xaxis=dict(title="φ (degrees)",range=[-180,180],dtick=45,showgrid=False,
                   zeroline=False,mirror=True,showline=True,linecolor="#111",linewidth=2,constrain="domain"),
        yaxis=dict(title="ψ (degrees)",range=[-180,180],dtick=45,showgrid=False,
                   zeroline=False,scaleanchor="x",scaleratio=1,
                   mirror=True,showline=True,linecolor="#111",linewidth=2),
        template="plotly_white",height=730,plot_bgcolor="white",dragmode="pan",
        legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center",
                    font=dict(size=10),bgcolor="white",bordercolor="#e2e8f0",borderwidth=1),
        margin=dict(l=60,r=20,t=60,b=160),
    )
    st.plotly_chart(fig,use_container_width=True,config={
        "scrollZoom":True,"displaylogo":False,
        "modeBarButtonsToRemove":["select2d","lasso2d"],
        "toImageButtonOptions":{"format":"png","filename":f"{pdb_id}_ramaplot","scale":3}
    })

    # ── MATPLOTLIB PNG ─────────────────────────────────────────────────────
    fig_s,ax=plt.subplots(figsize=(10,10),facecolor='white')
    norm=colors.BoundaryNorm([0,0.5,1.5,2.5],3)
    def rg(k):
        g=grids[k]; r=np.zeros_like(g)
        r[g>=THRESH_ALL]=1; r[g>=THRESH_FAV]=2; return r
    ax.imshow(rg("General"),cmap=colors.ListedColormap(['white',C_GEN_ALL,C_GEN_FAV]),
              norm=norm,extent=(-180,180,180,-180),aspect='auto',alpha=1.0,zorder=0)
    ax.imshow(rg("GLY"),cmap=colors.ListedColormap(['white',C_GLY_ALL,C_GLY_FAV]),
              norm=norm,extent=(-180,180,180,-180),aspect='auto',alpha=0.55,zorder=1)
    ax.set_xlim(-180,180); ax.set_ylim(-180,180)
    for v in range(-180,181,45):
        lw,ls,c=(1.5,"solid","#111") if v==0 else (0.5,"--","#888")
        ax.axhline(v,color=c,lw=lw,ls=ls,zorder=2); ax.axvline(v,color=c,lw=lw,ls=ls,zorder=2)
    for sp in ax.spines.values(): sp.set_linewidth(2); sp.set_color("#111")
    ax.scatter(bkt["fav"]["gen"]["x"],bkt["fav"]["gen"]["y"],marker='s',s=6,color=C_PT_GEN,zorder=5)
    ax.scatter(bkt["fav"]["gly"]["x"],bkt["fav"]["gly"]["y"],marker='s',s=6,color=C_PT_GLY,zorder=5)
    ax.scatter(bkt["allowed"]["gen"]["x"],bkt["allowed"]["gen"]["y"],marker='^',s=8,color=C_PT_GEN,alpha=0.8,zorder=5)
    ax.scatter(bkt["allowed"]["gly"]["x"],bkt["allowed"]["gly"]["y"],marker='^',s=8,color=C_PT_GLY,alpha=0.8,zorder=5)
    for sub in ["gen","gly"]:
        for x,y,l in zip(bkt["outlier"][sub]["x"],bkt["outlier"][sub]["y"],bkt["outlier"][sub]["lbl"]):
            ax.plot(x,y,'x',ms=5,color=C_OUTLIER,mew=1.5,zorder=10)
            ax.annotate(l,(x,y),xytext=(3,3),textcoords="offset points",fontsize=7,color=C_OUTLIER,weight='bold')
    stats=(f"Favoured (~98.0% expected) : {nfav} ({pfav:.1f}%)\n"
           f"Allowed  (~2.0% expected)  : {nall} ({pall:.1f}%)\n"
           f"Outlier                    : {nout} ({pout:.1f}%)")
    ax.text(0.02,0.02,stats,transform=ax.transAxes,fontsize=8,family='monospace',
            va='bottom',bbox=dict(boxstyle='round',fc='white',alpha=0.88,ec='#ccc'))
    ax.set_xlabel("φ (degrees)",fontsize=12); ax.set_ylabel("ψ (degrees)",fontsize=12)
    ax.set_title(f"Ramachandran Plot — {pdb_id}",fontsize=14,weight='bold',pad=15)
    handles=[
        mpatches.Patch(color=C_GEN_FAV,label='General Favoured Region'),
        mpatches.Patch(color=C_GEN_ALL,label='General Allowed Region'),
        mpatches.Patch(color=C_GLY_FAV,label='Glycine Favoured Region'),
        mpatches.Patch(color=C_GLY_ALL,label='Glycine Allowed Region'),
        mlines.Line2D([],[],color=C_PT_GEN,marker='s',ls='',ms=8,label='General Favoured'),
        mlines.Line2D([],[],color=C_PT_GEN,marker='^',ls='',ms=8,label='General Allowed'),
        mlines.Line2D([],[],color=C_PT_GLY,marker='s',ls='',ms=8,label='Glycine Favoured'),
        mlines.Line2D([],[],color=C_PT_GLY,marker='^',ls='',ms=8,label='Glycine Allowed'),
        mlines.Line2D([],[],color=C_OUTLIER,marker='x',ls='',ms=9,mew=2,label='Outlier'),
    ]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,-0.09),
              ncol=3,fontsize=8,frameon=True,facecolor='white',edgecolor='#ccc')
    plt.tight_layout(rect=[0,0.08,1,1])
    if 'png_bytes' not in st.session_state or st.session_state.get('last_png_pdb') != pdb_id:
        buf=io.BytesIO()
        fig_s.savefig(buf,format="png",dpi=300,bbox_inches='tight')
        buf.seek(0)
        st.session_state.png_bytes = buf.getvalue()
        st.session_state.last_png_pdb = pdb_id
    plt.close(fig_s)

    # ── DOWNLOAD via HTML Base64 Anchor (Bypasses Streamlit backend) ──
    b64 = base64.b64encode(st.session_state.png_bytes).decode()
    href = f'''
    <style>
        #dl-btn-link {{
            color: #ffffff !important;
            text-decoration: none !important;
        }}
        #dl-btn-link:hover, #dl-btn-link:visited, #dl-btn-link:active {{
            color: #ffffff !important;
            text-decoration: none !important;
        }}
    </style>
    <div style="display:flex;justify-content:center;margin:8px 0;">
        <a id="dl-btn-link" href="data:image/png;base64,{b64}" download="{pdb_id}_ramaplot.png" style="
            display:inline-block; padding:14px 48px; background-color:#2da44e; 
            color:#ffffff !important; text-decoration:none !important; border-radius:10px; 
            font-size:1rem; font-weight:600; font-family:'Outfit',sans-serif;
            box-shadow:0 4px 12px rgba(45,164,78,0.3);">
            Download Plot as PNG
        </a>
    </div>'''
    st.markdown(href, unsafe_allow_html=True)

with col2:
    st.markdown("### Statistics")
    for label, val, sub, thresh_text, is_met in [
        ("Total Residues", str(total), pdb_id, "", True),
        ("Favoured", f"{pfav:.1f}%", f"{nfav} residues", "| >98.0%", pfav >= 98.0),
        ("Allowed", f"{pall:.1f}%", f"{nall} residues", "| <2.0%", pall <= 2.0),
        ("Outliers", f"{pout:.1f}%", f"{nout} residues", "| 0.0%", pout == 0.0),
    ]:
        bg = "rgba(34, 197, 94, 0.08)" if is_met else "rgba(239, 68, 68, 0.08)"
        if label == "Total Residues": bg = "#fff"
        
        c = "#e53935" if label == "Outliers" else "var(--primary)"
        st.markdown(f"""<div class="metric-card" style="background:{bg}">
            <div class="metric-label">{label} <span style="opacity:0.6;margin-left:4px">{thresh_text}</span></div>
            <div class="metric-value" style="color:{c}">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")

st.markdown("---")
st.caption("Structural biology tool. Data: Top500 high-resolution set.")
