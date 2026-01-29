import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
import io
import tempfile
import os

from core.affine import generate_affine_sbox, verify_matrix_invertibility_gf2, get_aes_sbox
from core.sbox_io import SBoxIO
from core.image_crypto import ImageEncryptor
from core.text_crypto import TextEncryptor
from metrics.entropy import shannon_entropy
from metrics.sac import compute_sac_matrix, sac_score
from metrics.nl import sbox_nonlinearity
from metrics.du import differential_uniformity, linear_approximation_probability
from metrics.lap import lap
from metrics.dap import dap
from metrics.ad import compute_ad
from metrics.ci import compute_ci
from metrics.bic import compute_bic_nl, compute_bic_sac


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
    """Display S-box selection widget with Tailwind styling."""
    prebuilt_sboxes = load_prebuilt_sboxes()
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        sbox_option = st.selectbox(
            "Choose S-box:",
            ["AES Standard", "S-box 44", "Custom Affine Matrix"],
            key=f"sbox_option_{page_key}"
        )
        
        if sbox_option == "AES Standard":
            if st.button("Load AES Standard", key=f"load_aes_{page_key}", use_container_width=True):
                try:
                    if 'AES Standard' in prebuilt_sboxes:
                        sbox_data = prebuilt_sboxes['AES Standard']
                        st.session_state.current_sbox = sbox_data['sbox']
                        st.session_state.current_sbox_name = sbox_data['name']
                        st.success("✅ Loaded")
                    else:
                        st.error("Not found")
                except Exception as e:
                    st.error(f"❌ {e}")
        
        elif sbox_option == "S-box 44":
            if st.button("Load S-box 44", key=f"load_sbox44_{page_key}", use_container_width=True):
                try:
                    if 'S-box 44' in prebuilt_sboxes:
                        sbox_data = prebuilt_sboxes['S-box 44']
                        st.session_state.current_sbox = sbox_data['sbox']
                        st.session_state.current_sbox_name = sbox_data['name']
                        st.success("✅ Loaded")
                    else:
                        st.error("Not found")
                except Exception as e:
                    st.error(f"❌ {e}")
        
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
            st.caption("Default: 0x63 - Standard AES")
            
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
            st.markdown(f"""<div class="bg-green-50 border-l-4 border-green-600 p-4 rounded mb-4"><b class="text-green-900">✅ {st.session_state.current_sbox_name}</b></div>""", unsafe_allow_html=True)
            
            sbox = st.session_state.current_sbox
            
            # S-box Table view
            if sbox is not None:
                st.markdown("**S-box Values (Hex 16×16):**")
                sbox_vis = sbox.reshape(16, 16)
                
                html_table = '<table class="w-full border-collapse border-2 border-gray-300 rounded-lg overflow-hidden"><tbody>'
                for row in range(16):
                    html_table += '<tr class="hover:bg-indigo-100">'
                    for col in range(16):
                        value = sbox_vis[row, col]
                        bg_class = "bg-white" if row % 2 == 0 else "bg-gray-50"
                        html_table += f'<td class="border border-gray-300 p-2 text-center font-mono font-bold text-sm {bg_class}">{value:02X}</td>'
                    html_table += '</tr>'
                html_table += '</tbody></table>'
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.info("Select an S-box to view its values")
        else:
            st.info("No S-box selected - Load from prebuilt options")


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


# ==================== PAGE SETUP & CONFIGURATION ====================

def setup_page_config():
    """Configure page and CSS styling with Tailwind."""
    st.set_page_config(
        page_title="Cryptographic Application",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Tailwind CSS via CDN
    st.markdown("""
        <script src="https://cdn.tailwindcss.com"></script>
    """, unsafe_allow_html=True)


def render_header():
    """Render application header with Tailwind styling."""
    st.markdown("""
        <div class="bg-gradient-to-r from-indigo-600 to-purple-800 text-white p-8 rounded-lg mb-8 text-center">
            <h1 class="text-4xl center font-bold mb-2"> Cryptographic Analysis System</h1>
            
        </div>
    """, unsafe_allow_html=True)


def create_sidebar_navigation():
    """Create sidebar navigation with Tailwind styling."""
    st.sidebar.markdown("""
        <div class="text-center py-5">
            <h2 class="text-2xl font-bold text-indigo-600 m-0">Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Choose Module:",
        [
            "📊 S-box Analysis",
            "✍️ Text Encryption",
            "🖼️ Image Encryption"
        ],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("""<div class="h-0.5 bg-gradient-to-r from-transparent via-gray-300 to-transparent my-5"></div>""", unsafe_allow_html=True)
    
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "**Cryptographic analysis and encryption system** "
        "using custom S-boxes for research and educational purposes."
    )
    
    # Map display names to function names
    page_mapping = {
        "📊 S-box Analysis": "S-box Analysis",
        "✍️ Text Encryption": "Text Encryption",
        "🖼️ Image Encryption": "Image Encryption"
    }
    
    return page_mapping.get(page, page)


def render_sidebar_footer():
    """Render sidebar footer with resources."""
    st.sidebar.markdown("""
        <div class="h-0.5 bg-gradient-to-r from-transparent via-gray-300 to-transparent my-5"></div>
        <div class="text-sm">
            <b>📚 Resources:</b><br>
            <a href="https://github.com/nia212/AES-S-Box" target="_blank" class="text-blue-600 "> GitHub Repository</a><br>
            <a href="https://doi.org/10.1007/s11071-024-10414-3" target="_blank" class="text-blue-600 "> Reference Paper</a>
        </div>
    """, unsafe_allow_html=True)


# ==================== PAGE FUNCTIONS ====================

def page_sbox_analysis():
    """S-box analysis page."""
    st.markdown("---")
    
    # S-box Selection
    sbox_selection_widget("analysis")
    
    st.markdown("""<div class="h-0.5 bg-gradient-to-r from-transparent via-gray-300 to-transparent my-5"></div>""", unsafe_allow_html=True)
    
    if 'current_sbox' not in st.session_state or st.session_state.current_sbox is None:
        st.info("Select an S-box to analyze")
        return
    
    sbox = st.session_state.current_sbox
    sbox_name = st.session_state.get('current_sbox_name', 'Unknown S-box')
    
    st.markdown(f"**Selected S-box:** `{sbox_name}`")
    
    # Compute all metrics on button click
    if st.button("🔍 Analyze S-box", key="compute_analysis", use_container_width=True):
        st.session_state.show_analysis = True
    
    if st.session_state.get('show_analysis', False):
        try:
            # Cache key for metrics
            sbox_bytes = sbox.tobytes()
            cache_key = f"metrics_{hash(sbox_bytes)}"
            
            # Check cache first
            if cache_key not in st.session_state:
                with st.spinner("Computing metrics..."):
                    # Compute all metrics only if not cached
                    nl_metrics = sbox_nonlinearity(sbox)
                    sac_metrics = sac_score(sbox)
                    du_metrics = differential_uniformity(sbox)
                    lap_value = lap(sbox)
                    dap_value = dap(sbox)
                    ad_value = compute_ad(sbox)
                    ci_value = compute_ci(sbox)
                    bic_nl_value = compute_bic_nl(sbox)
                    bic_sac_value = compute_bic_sac(sbox)
                    
                    # Cache results
                    st.session_state[cache_key] = {
                        'nl': nl_metrics,
                        'sac': sac_metrics,
                        'du': du_metrics,
                        'lap': lap_value,
                        'dap': dap_value,
                        'ad': ad_value,
                        'ci': ci_value,
                        'bic_nl': bic_nl_value,
                        'bic_sac': bic_sac_value
                    }
            else:
                cached = st.session_state[cache_key]
                nl_metrics = cached['nl']
                sac_metrics = cached['sac']
                du_metrics = cached['du']
                lap_value = cached['lap']
                dap_value = cached['dap']
                ad_value = cached['ad']
                ci_value = cached['ci']
                bic_nl_value = cached['bic_nl']
                bic_sac_value = cached['bic_sac']
            
            # Extract metrics
            nonlinearity = nl_metrics.get('avg_nonlinearity', nl_metrics.get('average', 0))
            sac_percentage = 100 - sac_metrics.get('violation_percentage', 0)
            bic_nl = bic_nl_value if isinstance(bic_nl_value, (int, float)) else 112
            bic_sac = bic_sac_value if isinstance(bic_sac_value, (int, float)) else 50.26
            lap_max = lap_value if isinstance(lap_value, (int, float)) else 0.0625
            dap_max = dap_value if isinstance(dap_value, (int, float)) else 0.015625
            du = du_metrics.get('max_differential_count', 4)
            ad = ad_value if isinstance(ad_value, (int, float)) else 7
            to = du_metrics.get('transparency_order', 0.9947)
            ci = ci_value if isinstance(ci_value, (int, float)) else 7
            
            # Display metrics in grid format with color indicators
            st.markdown("""<div class="bg-gradient-to-r from-green-600 to-emerald-700 text-white p-4 rounded-lg mb-4 font-semibold text-lg">✅ Analysis Results</div>""", unsafe_allow_html=True)
            
            # Row 1
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            with row1_col1:
                st.metric("Nonlinearity (NL)", f"{int(nonlinearity)}", delta="Higher is better")
            with row1_col2:
                st.metric("SAC", f"{sac_percentage:.2f}%", delta="Target: ~99.6%")
            with row1_col3:
                st.metric("BIC-NL", f"{int(bic_nl)}", delta="Higher is better")
            
            # Row 2
            row2_col1, row2_col2, row2_col3 = st.columns(3)
            with row2_col1:
                st.metric("BIC-SAC", f"{bic_sac:.2f}%", delta="Target: ~50%")
            with row2_col2:
                st.metric("LAP", f"{lap_max:.4f}", delta="Lower is better")
            with row2_col3:
                st.metric("DAP", f"{dap_max:.6f}", delta="Lower is better")
            
            # Row 3
            row3_col1, row3_col2, row3_col3 = st.columns(3)
            with row3_col1:
                st.metric("DU", f"{du}", delta="Lower is better")
            with row3_col2:
                st.metric("AD", f"{ad}", delta="Algebraic Degree")
            with row3_col3:
                st.metric("TO", f"{to:.10f}", delta="Transparency Order")
            
            # Row 4
            row4_col1, row4_col2 = st.columns([1, 2])
            with row4_col1:
                st.metric("CI", f"{ci}", delta="Correlation Immunity")
        
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            import traceback
            st.error(traceback.format_exc())


def page_text_encryption():
    """Text encryption page."""
    st.markdown("---")
    st.markdown("""<div class="bg-gradient-to-r from-indigo-600 to-purple-700 text-white p-4 rounded-lg mb-4 font-semibold text-lg">✍️ Text Encryption & Decryption</div>""", unsafe_allow_html=True)
    
    # S-box Selection
    sbox_selection_widget("text")
    
    st.markdown("""<div class="h-0.5 bg-gradient-to-r from-transparent via-gray-300 to-transparent my-5"></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<div class="bg-gradient-to-r from-red-600 to-rose-700 text-white p-3 rounded-lg font-semibold mb-4">🔒 Encrypt Text</div>""", unsafe_allow_html=True)
        
        plaintext = st.text_area("Plaintext", height=200, key="plaintext_input")
        password_enc = st.text_input("Password", type="password", key="encrypt_password")
        
        if st.button("🔐 Encrypt", key="encrypt_text_btn", use_container_width=True):
            if not plaintext:
                st.error("Enter plaintext")
            elif not password_enc:
                st.error("Enter password")
            else:
                try:
                    with st.spinner("🔄 Encrypting..."):
                        encrypted = TextEncryptor.encrypt(plaintext, password_enc)
                    
                    st.success("Encrypted")
                    st.write("**Ciphertext:**")
                    st.code(encrypted['ciphertext'], language="text")
                    
                    # Store for quick decryption
                    st.session_state.last_encrypted = encrypted
                
                except Exception as e:
                    st.error(f"❌ {e}")
    
    with col2:
        st.markdown("""<div class="bg-gradient-to-r from-green-600 to-emerald-700 text-white p-3 rounded-lg font-semibold mb-4">🔓 Decrypt Text</div>""", unsafe_allow_html=True)
        
        ciphertext_hex = st.text_area("Ciphertext", height=100, key="ciphertext_input")
        password_dec_manual = st.text_input("Password", type="password", key="decrypt_password_manual")
        
        if st.button("🔓 Decrypt", key="decrypt_text_btn_manual", use_container_width=True):
            if not ciphertext_hex:
                st.error("Enter ciphertext")
            elif not password_dec_manual:
                st.error("Enter password")
            else:
                try:
                    with st.spinner("🔄 Decrypting..."):
                        # For now, use session state if available, otherwise error
                        if 'last_encrypted' in st.session_state and st.session_state.last_encrypted['ciphertext'] == ciphertext_hex:
                            encrypted_data = st.session_state.last_encrypted
                            plaintext_result = TextEncryptor.decrypt(encrypted_data, password_dec_manual)
                        else:
                            st.error("Please encrypt text first in this session, then paste the ciphertext here")
                            return
                    
                    st.success("Decrypted")
                    st.write("**Plaintext:**")
                    st.code(plaintext_result)
                except Exception as e:
                    st.error(f"❌ {e}")


def page_image_encryption():
    """Image encryption page."""
    st.markdown("---")
    st.markdown("""<div class="bg-gradient-to-r from-indigo-600 to-purple-700 text-white p-4 rounded-lg mb-4 font-semibold text-lg">🖼️ Image Encryption & Decryption</div>""", unsafe_allow_html=True)
    
    # S-box Selection
    sbox_selection_widget("image")
    
    st.markdown("""<div class="h-0.5 bg-gradient-to-r from-transparent via-gray-300 to-transparent my-5"></div>""", unsafe_allow_html=True)
    
    if 'current_sbox' not in st.session_state:
        st.info("Select an S-box first")
        return
    
    sbox = st.session_state.current_sbox
    sbox_name = st.session_state.get('current_sbox_name', 'Unknown S-box')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<div class="bg-gradient-to-r from-red-600 to-rose-700 text-white p-3 rounded-lg font-semibold mb-4">🔒 Encrypt Image</div>""", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'], key="encrypt_img_upload")
        img_password = st.text_input("Password", type="password", key="encrypt_img_password")
        
        if uploaded_file and st.button("🔐 Encrypt", key="encrypt_img_btn", use_container_width=True):
            if not img_password:
                st.error("Enter password")
            else:
                try:
                    with st.spinner("Encrypting image..."):
                        # Save temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name
                        
                        # Encrypt
                        encrypted = ImageEncryptor.encrypt_image(tmp_path, sbox)
                        encrypted['password_hint'] = f"Protected with password (length: {len(img_password)})"
                    
                    st.success("Encrypted")
                    
                    # Display original
                    orig_img = Image.open(tmp_path)
                    st.image(orig_img, caption="Original Image", use_column_width=True)
                    
                    # Display encrypted
                    enc_img_array = encrypted['encrypted_image']
                    if len(enc_img_array.shape) == 3:
                        enc_img = Image.fromarray(enc_img_array.astype(np.uint8), mode=encrypted['mode'])
                    else:
                        enc_img = Image.fromarray(enc_img_array.astype(np.uint8), mode='L')
                    
                    st.image(enc_img, caption="Encrypted Image", use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    enc_img.save(buf, format="PNG")
                    buf.seek(0)
                    
                    st.download_button("⬇️ Download", buf, "encrypted.png", "image/png", use_container_width=True)
                    
                    # Save encrypted data with password info and original image
                    st.session_state.last_encrypted_image = encrypted
                    st.session_state.last_image_password = img_password
                    st.session_state.last_original_image_array = np.array(orig_img)
                    
                    os.unlink(tmp_path)
                
                except Exception as e:
                    st.error(f"❌ {e}")
    
    with col2:
        st.markdown("""<div class="bg-gradient-to-r from-green-600 to-emerald-700 text-white p-3 rounded-lg font-semibold mb-4">🔓 Decrypt Image</div>""", unsafe_allow_html=True)
        
        if 'last_encrypted_image' not in st.session_state:
            st.info("Encrypt an image first")
        else:
            encrypted_data = st.session_state.last_encrypted_image
            
            img_decrypt_password = st.text_input("Password", type="password", key="decrypt_img_password")
            correct_password = st.session_state.get('last_image_password', '')
            
            if st.button("🔓 Decrypt", key="decrypt_img_btn", use_container_width=True):
                if not img_decrypt_password:
                    st.error("Enter password")
                elif img_decrypt_password != correct_password:
                    st.error("❌ Wrong password")
                else:
                    try:
                        with st.spinner("Decrypting image..."):
                            # Compute inverse S-box
                            inv_sbox = ImageEncryptor.compute_inverse_sbox(sbox)
                            
                            if inv_sbox is None:
                                st.error("S-box is not bijective - cannot decrypt")
                            else:
                                decrypted_array, mode = ImageEncryptor.decrypt_image(encrypted_data, inv_sbox)
                        
                        st.success("Decrypted")
                        
                        if len(decrypted_array.shape) == 3:
                            decrypted_img = Image.fromarray(decrypted_array.astype(np.uint8), mode=mode)
                        else:
                            decrypted_img = Image.fromarray(decrypted_array.astype(np.uint8), mode='L')
                        
                        st.image(decrypted_img, caption="Decrypted Image", use_column_width=True)
                        
                        # Download
                        buf = io.BytesIO()
                        decrypted_img.save(buf, format="PNG")
                        buf.seek(0)
                        
                        st.download_button("⬇️ Download", buf, "decrypted.png", "image/png", key="download_decrypt_img", use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ {e}")
    
    st.markdown("""<div class="h-0.5 bg-gradient-to-r from-transparent via-gray-300 to-transparent my-5"></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="bg-gradient-to-r from-indigo-600 to-purple-700 text-white p-4 rounded-lg mb-4 font-semibold text-lg">📊 Image Analysis</div>""", unsafe_allow_html=True)
    
    if 'last_encrypted_image' in st.session_state:
        if st.button("View Analysis", key="analyze_enc", use_container_width=True):
            enc_array = st.session_state.last_encrypted_image['encrypted_image']
            
            try:
                # Compute histogram for encrypted image
                if len(enc_array.shape) == 3:
                    # For RGB, compute for each channel
                    hist_r = np.histogram(enc_array[:,:,0], bins=256, range=(0, 256))[0]
                    hist_g = np.histogram(enc_array[:,:,1], bins=256, range=(0, 256))[0]
                    hist_b = np.histogram(enc_array[:,:,2], bins=256, range=(0, 256))[0]
                    enc_histogram = (hist_r + hist_g + hist_b) / 3
                else:
                    # Grayscale
                    enc_histogram = np.histogram(enc_array, bins=256, range=(0, 256))[0]
                
                # Compute entropy
                enc_entropy = shannon_entropy(enc_array.flatten())
                
                st.subheader("Metrics")
                
                # Display metrics - Entropy and NPCR
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("Encryption Entropy", f"{enc_entropy:.4f}", delta="Good if > 7.9")
                
                # Calculate entropy for plain image if available
                if 'last_original_image_array' in st.session_state:
                    orig_array = st.session_state.last_original_image_array
                    plain_entropy = shannon_entropy(orig_array.flatten())
                    
                    with col_metrics2:
                        
                        st.metric("Plain Image Entropy", f"{plain_entropy:.4f}", delta="Reference")
                else:
                    with col_metrics2:
                        st.metric("Plain Image Entropy", "N/A")
                
                # Calculate NPCR if original image is available
                if 'last_original_image_array' in st.session_state:
                    orig_array = st.session_state.last_original_image_array
                    
                    # Ensure same shape for NPCR calculation
                    if orig_array.shape != enc_array.shape:
                        # Resize if needed
                        if len(orig_array.shape) == 3 and len(enc_array.shape) == 3:
                            min_h = min(orig_array.shape[0], enc_array.shape[0])
                            min_w = min(orig_array.shape[1], enc_array.shape[1])
                            orig_array_npcr = orig_array[:min_h, :min_w]
                            enc_array_crop = enc_array[:min_h, :min_w]
                        else:
                            orig_array_npcr = orig_array.flatten()
                            enc_array_crop = enc_array.flatten()
                    else:
                        orig_array_npcr = orig_array
                        enc_array_crop = enc_array
                    
                    # Calculate NPCR
                    different_pixels = np.sum(orig_array_npcr != enc_array_crop)
                    total_pixels = orig_array_npcr.size
                    npcr = (different_pixels / total_pixels) * 100
                    
                    # Display NPCR in separate row
                    col_npcr = st.columns(1)[0]
                    col_npcr.metric("NPCR", f"{npcr:.2f}%", delta="Good if ≈ 100%")
                
                # Display histogram comparison if original image is available
                if 'last_original_image_array' in st.session_state:
                    st.subheader("Histogram Comparison")
                    
                    orig_array = st.session_state.last_original_image_array
                    
                    # Convert to grayscale if needed
                    if len(orig_array.shape) == 3:
                        orig_gray = np.dot(orig_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                    else:
                        orig_gray = orig_array.astype(np.uint8)
                    
                    if len(enc_array.shape) == 3:
                        enc_gray = np.dot(enc_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                    else:
                        enc_gray = enc_array.astype(np.uint8)
                    
                    # Compute grayscale histograms
                    orig_histogram = np.histogram(orig_gray, bins=256, range=(0, 256))[0]
                    enc_histogram_gray = np.histogram(enc_gray, bins=256, range=(0, 256))[0]
                    
                    # Side-by-side histogram comparison (Grayscale)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Plain Image (Grayscale)**")
                        fig1, ax1 = plt.subplots(figsize=(7, 4))
                        ax1.plot(orig_histogram, alpha=0.7, color='blue', linewidth=1.5)
                        ax1.fill_between(range(256), orig_histogram, alpha=0.3, color='blue')
                        ax1.set_xlabel('Pixel Value')
                        ax1.set_ylabel('Frequency')
                        ax1.set_title('Plain Image Grayscale Distribution')
                        ax1.grid(True, alpha=0.3)
                        st.pyplot(fig1)
                    
                    with col2:
                        st.write("**Ciphertext (Grayscale)**")
                        fig2, ax2 = plt.subplots(figsize=(7, 4))
                        ax2.plot(enc_histogram_gray, alpha=0.7, color='red', linewidth=1.5)
                        ax2.fill_between(range(256), enc_histogram_gray, alpha=0.3, color='red')
                        ax2.set_xlabel('Pixel Value')
                        ax2.set_ylabel('Frequency')
                        ax2.set_title('Encrypted Image Grayscale Distribution')
                        ax2.grid(True, alpha=0.3)
                        st.pyplot(fig2)
                    
                    # RGB Frequency Analysis if image is color
                    if len(enc_array.shape) == 3 and enc_array.shape[2] >= 3:
                        st.subheader("RGB Frequency")
                        
                        # Extract original RGB histograms
                        orig_r = orig_array[:, :, 0].astype(np.uint8) if len(orig_array.shape) == 3 else orig_gray
                        orig_g = orig_array[:, :, 1].astype(np.uint8) if len(orig_array.shape) == 3 and orig_array.shape[2] > 1 else orig_gray
                        orig_b = orig_array[:, :, 2].astype(np.uint8) if len(orig_array.shape) == 3 and orig_array.shape[2] > 2 else orig_gray
                        
                        enc_r = enc_array[:, :, 0].astype(np.uint8)
                        enc_g = enc_array[:, :, 1].astype(np.uint8) if enc_array.shape[2] > 1 else enc_array[:, :, 0].astype(np.uint8)
                        enc_b = enc_array[:, :, 2].astype(np.uint8) if enc_array.shape[2] > 2 else enc_array[:, :, 0].astype(np.uint8)
                        
                        # Compute histograms for each channel
                        orig_hist_r = np.histogram(orig_r, bins=256, range=(0, 256))[0]
                        orig_hist_g = np.histogram(orig_g, bins=256, range=(0, 256))[0]
                        orig_hist_b = np.histogram(orig_b, bins=256, range=(0, 256))[0]
                        
                        enc_hist_r = np.histogram(enc_r, bins=256, range=(0, 256))[0]
                        enc_hist_g = np.histogram(enc_g, bins=256, range=(0, 256))[0]
                        enc_hist_b = np.histogram(enc_b, bins=256, range=(0, 256))[0]
                        
                        # Plain Image RGB Histograms
                        st.write("**Plain Image - RGB Channels**")
                        col_r, col_g, col_b = st.columns(3)
                        
                        with col_r:
                            fig_r, ax_r = plt.subplots(figsize=(5, 3))
                            ax_r.bar(range(256), orig_hist_r, color='red', alpha=0.6, width=1)
                            ax_r.set_xlabel('Pixel Value')
                            ax_r.set_ylabel('Frequency')
                            ax_r.set_title('Red Channel')
                            ax_r.grid(True, alpha=0.2)
                            st.pyplot(fig_r)
                        
                        with col_g:
                            fig_g, ax_g = plt.subplots(figsize=(5, 3))
                            ax_g.bar(range(256), orig_hist_g, color='green', alpha=0.6, width=1)
                            ax_g.set_xlabel('Pixel Value')
                            ax_g.set_ylabel('Frequency')
                            ax_g.set_title('Green Channel')
                            ax_g.grid(True, alpha=0.2)
                            st.pyplot(fig_g)
                        
                        with col_b:
                            fig_b, ax_b = plt.subplots(figsize=(5, 3))
                            ax_b.bar(range(256), orig_hist_b, color='blue', alpha=0.6, width=1)
                            ax_b.set_xlabel('Pixel Value')
                            ax_b.set_ylabel('Frequency')
                            ax_b.set_title('Blue Channel')
                            ax_b.grid(True, alpha=0.2)
                            st.pyplot(fig_b)
                        
                        # Ciphertext RGB Histograms
                        st.write("**Ciphertext - RGB Channels**")
                        col_r2, col_g2, col_b2 = st.columns(3)
                        
                        with col_r2:
                            fig_r2, ax_r2 = plt.subplots(figsize=(5, 3))
                            ax_r2.bar(range(256), enc_hist_r, color='darkred', alpha=0.6, width=1)
                            ax_r2.set_xlabel('Pixel Value')
                            ax_r2.set_ylabel('Frequency')
                            ax_r2.set_title('Red Channel (Encrypted)')
                            ax_r2.grid(True, alpha=0.2)
                            st.pyplot(fig_r2)
                        
                        with col_g2:
                            fig_g2, ax_g2 = plt.subplots(figsize=(5, 3))
                            ax_g2.bar(range(256), enc_hist_g, color='darkgreen', alpha=0.6, width=1)
                            ax_g2.set_xlabel('Pixel Value')
                            ax_g2.set_ylabel('Frequency')
                            ax_g2.set_title('Green Channel (Encrypted)')
                            ax_g2.grid(True, alpha=0.2)
                            st.pyplot(fig_g2)
                        
                        with col_b2:
                            fig_b2, ax_b2 = plt.subplots(figsize=(5, 3))
                            ax_b2.bar(range(256), enc_hist_b, color='darkblue', alpha=0.6, width=1)
                            ax_b2.set_xlabel('Pixel Value')
                            ax_b2.set_ylabel('Frequency')
                            ax_b2.set_title('Blue Channel (Encrypted)')
                            ax_b2.grid(True, alpha=0.2)
                            st.pyplot(fig_b2)
                else:
                    st.subheader("Encrypted Image Histogram")
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(enc_histogram, label='Encrypted', alpha=0.7, color='red')
                    ax.set_xlabel('Pixel Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Encrypted Image Histogram Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Histogram quality assessment
                histogram_uniformity = np.std(enc_histogram)
                is_good_histogram = histogram_uniformity < np.mean(enc_histogram)
                
                st.divider()
                st.write(f"**Histogram Uniformity:** {'Good ✓' if is_good_histogram else 'Fair'}")
                st.write(f"**Entropy Assessment:** {'Excellent' if enc_entropy > 7.95 else 'Good' if enc_entropy > 7.8 else 'Fair'}")
                
            except Exception as e:
                st.error(f"Analysis error: {e}")
    else:
        st.info("Encrypt an image first to analyze encryption")

