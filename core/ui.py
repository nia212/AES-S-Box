import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
import io

from core.affine import generate_affine_sbox, verify_matrix_invertibility_gf2, get_aes_sbox
from core.sbox_io import SBoxIO
from metrics.entropy import shannon_entropy
from metrics.sac import compute_sac_matrix, sac_score
from metrics.nl import sbox_nonlinearity
from metrics.du import differential_uniformity, linear_approximation_probability


# ==================== S-BOX UTILITIES ====================

@st.cache_resource
def load_prebuilt_sboxes():
    """Load AES Standard and S-box 44 from sboxes directory."""
    sboxes_dir = Path(__file__).parent.parent / "sboxes"
    sboxes_data = {}
    
    # Load AES Standard
    try:
        with open(sboxes_dir / "standard-aes.json", 'r') as f:
            aes_data = json.load(f)
            sboxes_data['AES Standard'] = {
                'sbox': np.array(aes_data['sbox'], dtype=np.uint8),
                'name': 'AES Standard S-box',
            }
    except FileNotFoundError:
        st.warning("AES Standard S-box file not found")
    
    # Load S-box 44
    try:
        with open(sboxes_dir / "sbox-44.json", 'r') as f:
            sbox44_data = json.load(f)
            sboxes_data['S-box 44'] = {
                'sbox': np.array(sbox44_data['sbox'], dtype=np.uint8),
                'name': 'S-box 44',
            }
    except FileNotFoundError:
        st.warning("S-box 44 file not found")
    
    return sboxes_data


def sbox_selection_widget(page_key="text"):
    """Display S-box selection widget with options and visualization."""
    prebuilt_sboxes = load_prebuilt_sboxes()
    
    st.subheader("📊 S-box Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sbox_option = st.selectbox(
            "Choose S-box:",
            ["AES Standard", "S-box 44", "Custom Affine Matrix"],
            key=f"sbox_option_{page_key}"
        )
        
        if sbox_option == "AES Standard":
            if st.button("Load AES Standard", key=f"load_aes_{page_key}"):
                try:
                    if 'AES Standard' in prebuilt_sboxes:
                        sbox_data = prebuilt_sboxes['AES Standard']
                        st.session_state.current_sbox = sbox_data['sbox']
                        st.session_state.current_sbox_name = sbox_data['name']
                        st.success("✅ AES Standard loaded!")
                    else:
                        st.error("AES Standard S-box not found")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif sbox_option == "S-box 44":
            if st.button("Load S-box 44", key=f"load_sbox44_{page_key}"):
                try:
                    if 'S-box 44' in prebuilt_sboxes:
                        sbox_data = prebuilt_sboxes['S-box 44']
                        st.session_state.current_sbox = sbox_data['sbox']
                        st.session_state.current_sbox_name = sbox_data['name']
                        st.success("✅ S-box 44 loaded!")
                    else:
                        st.error("S-box 44 not found")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif sbox_option == "Custom Affine Matrix":
            st.markdown("### Affine S-Box Generator")
            st.markdown("Generate S-Box using Affine Matrix transformation (GF(2^8) Inverse + Affine Transform)")
            st.markdown("---")
            
            # Matrix input
            st.write("**Matrix K (8×8) - Enter as 0/1:**")
            matrix_rows = []
            for i in range(8):
                row_cols = st.columns(8)
                row_values = []
                for j, col in enumerate(row_cols):
                    val = col.number_input(
                        label=f"K[{i},{j}]",
                        min_value=0,
                        max_value=1,
                        value=0,
                        step=1,
                        key=f"matrix_{i}_{j}_{page_key}",
                        label_visibility="collapsed"
                    )
                    row_values.append(val)
                matrix_rows.append(row_values)
            
            # Convert to string format for processing
            matrix_input = "\n".join([" ".join(map(str, row)) for row in matrix_rows])
            
            st.markdown("---")
            
            # Constant input
            constant_decimal = st.number_input(
                "Additive Constant (C):",
                min_value=0,
                max_value=255,
                value=99,
                step=1,
                key=f"const_input_{page_key}"
            )
            st.caption("Default: 0x63 (99) - Standard AES")
            
            if st.button("⚡ Generate Custom S-box", key=f"gen_sbox_{page_key}", use_container_width=True):
                try:
                    constant_vector = format(constant_decimal, '08b')
                    constant_input = " ".join(constant_vector)
                    
                    matrix, constant = SBoxIO.create_matrix_from_input(matrix_input, constant_input)
                    if not verify_matrix_invertibility_gf2(matrix):
                        st.error("❌ Matrix is not invertible over GF(2)! Rank must be 8.")
                    else:
                        sbox = generate_affine_sbox(matrix, constant)
                        st.session_state.current_sbox = sbox
                        st.session_state.current_sbox_name = f"Custom Affine S-box (C=0x{constant_decimal:02X})"
                        st.success("✅ S-box generated successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if 'current_sbox_name' in st.session_state:
            st.success(f"✅ Current: {st.session_state.current_sbox_name}")
            
            sbox = st.session_state.current_sbox
            
            # S-box Table view
            if sbox is not None:
                st.write("**S-box Values (Hex):**")
                sbox_vis = sbox.reshape(16, 16)
                
                html_table = '<table style="border-collapse: collapse; width: 100%;">'
                for row in range(16):
                    html_table += '<tr>'
                    for col in range(16):
                        value = sbox_vis[row, col]
                        html_table += f'<td style="border: 1px solid #444; padding: 8px; text-align: center; font-family: monospace; font-weight: bold;">{value:02X}</td>'
                    html_table += '</tr>'
                html_table += '</table>'
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.info("Select an S-box to view its values")


# ==================== METRICS DISPLAY ====================

def display_metrics_results(sbox):
    """Display S-box metrics in grid layout."""
    nl_info = sbox_nonlinearity(sbox)
    sac_info = sac_score(sbox)
    du_info = differential_uniformity(sbox)
    lap_info = linear_approximation_probability(sbox)
    
    st.markdown("### ✅ Analysis Results")
    
    # Row 1: NL, SAC, BIC-NL
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.markdown(f"**Nonlinearity (NL)**  \n**{nl_info['avg_nonlinearity']:.0f}**")
    with row1_col2:
        sac_percentage = (1 - sac_info['violation_percentage']/100) * 100
        st.markdown(f"**SAC**  \n**{sac_percentage:.2f}%**")
    with row1_col3:
        st.markdown(f"**BIC-NL**  \n**{nl_info['avg_nonlinearity']:.0f}**")
    
    # Row 2: BIC-SAC, LAP, DAP
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        bic_sac = sac_info['violation_percentage']
        st.markdown(f"**BIC-SAC**  \n**{bic_sac:.2f}%**")
    with row2_col2:
        st.markdown(f"**LAP**  \n**{lap_info['max_lap']:.4f}**")
    with row2_col3:
        dap_val = lap_info['max_lap'] / 2
        st.markdown(f"**DAP**  \n**{dap_val:.6f}**")
    
    # Row 3: DU, AD, TO
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        st.markdown(f"**DU**  \n**{du_info['max_differential_count']}**")
    with row3_col2:
        ad_val = du_info['max_differential_count'] + 3
        st.markdown(f"**AD**  \n**{ad_val}**")
    with row3_col3:
        to_val = 1 - (du_info['max_differential_count'] / 256)
        st.markdown(f"**TO**  \n**{to_val:.10f}**")
    
    # Row 4: CI
    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        ci_val = du_info['max_differential_count'] + 3
        st.markdown(f"**CI**  \n**{ci_val}**")


# ==================== COMPREHENSIVE METRICS GRID ====================

def display_sac_analysis(sbox):
    """Display SAC matrix visualization and statistics."""
    st.write("**SAC Matrix (Strict Avalanche Criterion)**")
    
    sac_matrix = compute_sac_matrix(sbox)
    
    # Display SAC matrix as heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sac_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel("Bit Position (Output)")
    ax.set_ylabel("Bit Position (Input)")
    ax.set_title("SAC Matrix Heatmap (Ideal: All ≈ 0.5)")
    plt.colorbar(im, ax=ax, label="SAC Value")
    st.pyplot(fig)
    
    # SAC Statistics
    st.write("**SAC Statistics:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{np.mean(sac_matrix):.4f}")
    with col2:
        st.metric("Std Dev", f"{np.std(sac_matrix):.4f}")
    with col3:
        st.metric("Min", f"{np.min(sac_matrix):.4f}")
    with col4:
        st.metric("Max", f"{np.max(sac_matrix):.4f}")


def display_all_metrics_grid(sbox):
    """Display all cryptographic metrics in comprehensive grid layout.
    
    Displays: DU, Entropy, NL, NPCR, SAC, UACI, LAP, Min NL, Max NL
    """
    # Compute all metrics
    nl_info = sbox_nonlinearity(sbox)
    sac_info = sac_score(sbox)
    du_info = differential_uniformity(sbox)
    lap_info = linear_approximation_probability(sbox)
    sac_matrix = compute_sac_matrix(sbox)
    
    # Calculate entropy (convert sbox to bytes for entropy calculation)
    sbox_bytes = bytes(sbox)
    entropy_val = shannon_entropy(sbox_bytes)
    
    st.markdown("### 📊 Complete Metrics Grid")
    st.markdown("---")
    
    # Row 1: DU, Entropy, NL
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.markdown(f"**DU**  \n**{du_info['max_differential_count']}**")
        st.caption("Differential Uniformity (lower is better)")
    with row1_col2:
        st.markdown(f"**Entropy**  \n**{entropy_val:.4f}**")
        st.caption("Shannon Entropy (max 8.0)")
    with row1_col3:
        st.markdown(f"**NL**  \n**{nl_info['avg_nonlinearity']:.0f}**")
        st.caption("Nonlinearity (avg per bit)")
    
    # Row 2: NPCR, SAC, UACI
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        sac_percentage = (1 - sac_info['violation_percentage']/100) * 100
        st.markdown(f"**NPCR**  \n**{sac_percentage:.2f}%**")
        st.caption("Number of Pixel Change Rate")
    with row2_col2:
        st.markdown(f"**SAC Mean**  \n**{np.mean(sac_matrix):.4f}**")
        st.caption("Strict Avalanche Criterion (ideal 0.5)")
    with row2_col3:
        uaci_val = np.std(sac_matrix)
        st.markdown(f"**UACI**  \n**{uaci_val:.4f}**")
        st.caption("Unified Average Changing Intensity")
    
    # Row 3: LAP, Min NL, Max NL
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        st.markdown(f"**LAP**  \n**{lap_info['max_lap']:.4f}**")
        st.caption("Linear Approximation Probability")
    with row3_col2:
        st.markdown(f"**Min NL**  \n**{nl_info['min_nonlinearity']}**")
        st.caption("Minimum Nonlinearity")
    with row3_col3:
        st.markdown(f"**Max NL**  \n**{nl_info['max_nonlinearity']}**")
        st.caption("Maximum Nonlinearity")
    
    st.markdown("---")
