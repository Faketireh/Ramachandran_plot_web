# 🧬 Ramachandran Plot Generator

A premium, high-precision web application for generating RAMPAGE-style Ramachandran plots. This tool analyzes protein structures to visualize the backbone dihedral angles (φ and ψ) and provides a statistical breakdown of residue distributions based on high-resolution data.

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

## Features

- **Multi-Source Input**: Fetch protein structures directly from the RCSB PDB database via ID or upload your own `.pdb` files.
- **RAMPAGE-Style Aesthetics**: High-quality plots with clear demarcations for Favoured, Allowed, and Outlier regions.
- **Amino Acid Specificity**: Specialized distribution grids for General, Glycine (GLY), Proline (PRO), and Pre-Proline residues.
- **Statistical Summary**: Instant calculation of residue percentages in each region (Favoured, Allowed, Outliers).
- **Export Options**: Download publication-ready plots as high-resolution PNG files.
- **Premium UI**: Modern, clean, and responsive interface optimized for both Light and Dark modes.

## Quick Start

### Deployed Version
Access the live application here: [Ramachandran Plot Web](https://ramachandranplotweb-2fftu6jxqyadbwufsqt7yj.streamlit.app/)

### Local Installation
If you'd like to run the generator locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ramachandranplotweb.git
   cd ramachandranplotweb
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Technical Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Structural Analysis**: [Biopython (Bio.PDB)](https://biopython.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/)
- **Data Handling**: [NumPy](https://numpy.org/)

## Data Source

The distribution regions are calculated based on the **Top500** high-resolution protein set, ensuring that the probability density grids used for classification are scientifically accurate and up-to-date.

## Contributing

Contributions are welcome! If you have suggestions for new features or find any bugs:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📧 Contact

**Tirth Patel** - [tirthtirth10@gmail.com](mailto:tirthtirth10@gmail.com)

Project Link: [https://github.com/your-username/ramachandranplotweb](https://github.com/your-username/ramachandranplotweb)
