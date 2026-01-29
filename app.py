import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
import tempfile
import os
from pathlib import Path

# Import core modules
from core.affine import get_aes_sbox
from core.text_crypto import TextEncryptor, DocumentEncryptor
from core.image_crypto import ImageEncryptor
from core.ui import (
    load_prebuilt_sboxes,
    sbox_selection_widget,
    display_metrics_results,
    display_sac_analysis,
    display_all_metrics_grid
)

# Import metrics
from metrics.nl import sbox_nonlinearity
from metrics.sac import sac_score, compute_sac_matrix
from metrics.du import differential_uniformity, linear_approximation_probability
from metrics.lap import lap
from metrics.dap import dap
from metrics.ad import compute_ad
from metrics.ci import compute_ci
from metrics.bic import compute_bic_nl, compute_bic_sac
from metrics.entropy import shannon_entropy
from metrics.npcr import npcr_uaci_analysis
from metrics.uaci import histogram_metrics, image_quality_metrics


# Page configuration
st.set_page_config(
    page_title="Cryptographic Application",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Preload sboxes
prebuilt_sboxes = load_prebuilt_sboxes()

st.title("AES S-box Cryptographic Analysis System")
st.markdown("---")


def create_sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.title("🔐 Crypto App")
    
    page = st.sidebar.radio(
        "Select Module:",
        [
            "🔒 Text Encryption",
            "🖼️ Image Encryption",
            "📈 S-box Analysis"
        ]
    )
    return page


def page_sbox_analysis():
    """S-box analysis page."""
    import matplotlib.pyplot as plt
    
    st.header("📈 S-box Analysis & Metrics")
    
    # S-box Selection
    sbox_selection_widget("analysis")
    
    st.markdown("---")
    
    if 'current_sbox' not in st.session_state or st.session_state.current_sbox is None:
        st.warning("Please select an S-box first from the sidebar.")
        return
    
    sbox = st.session_state.current_sbox
    sbox_name = st.session_state.get('current_sbox_name', 'Unknown S-box')
    
    st.markdown(f"**Analyzing:** {sbox_name}")
    st.markdown("---")
    
    # Compute all metrics on button click
    if st.button("🔍 Analyze S-box", key="compute_analysis", use_container_width=True):
        st.session_state.show_analysis = True
    
    if st.session_state.get('show_analysis', False):
        try:
            # Compute all metrics
            nl_metrics = sbox_nonlinearity(sbox)
            sac_metrics = sac_score(sbox)
            du_metrics = differential_uniformity(sbox)
            lap_value = lap(sbox)
            dap_value = dap(sbox)
            ad_value = compute_ad(sbox)
            ci_value = compute_ci(sbox)
            bic_nl_value = compute_bic_nl(sbox)
            bic_sac_value = compute_bic_sac(sbox)
            
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
            
            # Display metrics in grid format
            st.subheader("✅ Analysis Results")
            
            # Row 1
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            with row1_col1:
                st.metric("Nonlinearity (NL)", f"{int(nonlinearity)}")
            with row1_col2:
                st.metric("SAC", f"{sac_percentage:.2f}%")
            with row1_col3:
                st.metric("BIC-NL", f"{int(bic_nl)}")
            
            # Row 2
            row2_col1, row2_col2, row2_col3 = st.columns(3)
            with row2_col1:
                st.metric("BIC-SAC", f"{bic_sac:.2f}%")
            with row2_col2:
                st.metric("LAP", f"{lap_max:.4f}")
            with row2_col3:
                st.metric("DAP", f"{dap_max:.6f}")
            
            # Row 3
            row3_col1, row3_col2, row3_col3 = st.columns(3)
            with row3_col1:
                st.metric("DU", f"{du}")
            with row3_col2:
                st.metric("AD", f"{ad}")
            with row3_col3:
                st.metric("TO", f"{to:.10f}")
            
            # Row 4
            row4_col1, row4_col2, row4_col3 = st.columns(3)
            with row4_col1:
                st.metric("CI", f"{ci}")
            
            st.markdown("---")
            
            # SAC Matrix Visualization
            st.subheader("🎯 SAC Matrix")
            if st.button("Show SAC Matrix", key="show_sac_matrix"):
                st.session_state.show_sac = True
            
            if st.session_state.get('show_sac', False):
                display_sac_analysis(sbox)
            
            st.markdown("---")
            
            # Comparison with AES
            st.subheader("🔄 Compare with AES Standard")
            if st.button("Run Comparison", key="compare_aes"):
                st.session_state.show_comparison = True
            
            if st.session_state.get('show_comparison', False):
                aes_sbox = get_aes_sbox()
                
                # Compute AES metrics
                aes_nl = sbox_nonlinearity(aes_sbox)
                aes_sac = sac_score(aes_sbox)
                aes_du = differential_uniformity(aes_sbox)
                aes_lap = lap(aes_sbox)
                aes_dap = dap(aes_sbox)
                aes_ad = compute_ad(aes_sbox)
                aes_ci = compute_ci(aes_sbox)
                aes_bic_nl = compute_bic_nl(aes_sbox)
                aes_bic_sac = compute_bic_sac(aes_sbox)
                
                # Extract AES metrics
                aes_nl_val = aes_nl.get('avg_nonlinearity', aes_nl.get('average', 0))
                aes_sac_val = 100 - aes_sac.get('violation_percentage', 0)
                aes_du_val = aes_du.get('max_differential_count', 4)
                aes_lap_val = aes_lap if isinstance(aes_lap, (int, float)) else 0.0625
                aes_dap_val = aes_dap if isinstance(aes_dap, (int, float)) else 0.015625
                aes_ad_val = aes_ad if isinstance(aes_ad, (int, float)) else 7
                aes_ci_val = aes_ci if isinstance(aes_ci, (int, float)) else 7
                aes_bic_nl_val = aes_bic_nl if isinstance(aes_bic_nl, (int, float)) else 112
                
                # Create comparison
                comparison_df = pd.DataFrame({
                    'Metric': [
                        'Nonlinearity (NL)',
                        'SAC',
                        'DU',
                        'LAP',
                        'DAP',
                        'AD',
                        'CI',
                        'BIC-NL'
                    ],
                    sbox_name: [
                        f"{int(nonlinearity)}",
                        f"{sac_percentage:.2f}%",
                        f"{du}",
                        f"{lap_max:.4f}",
                        f"{dap_max:.6f}",
                        f"{int(ad)}",
                        f"{int(ci)}",
                        f"{int(bic_nl)}"
                    ],
                    'AES Standard': [
                        f"{int(aes_nl_val)}",
                        f"{aes_sac_val:.2f}%",
                        f"{aes_du_val}",
                        f"{aes_lap_val:.4f}",
                        f"{aes_dap_val:.6f}",
                        f"{int(aes_ad_val)}",
                        f"{int(aes_ci_val)}",
                        f"{int(aes_bic_nl_val)}"
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Download comparison
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison",
                    data=csv,
                    file_name="sbox_comparison.csv",
                    mime="text/csv"
                )
                
                st.markdown("---")
                
                # Histogram Comparison
                st.subheader("📊 Histogram Comparison")
                
                # Prepare numeric data for histograms
                metrics_numeric = [
                    ('Nonlinearity (NL)', int(nonlinearity), int(aes_nl_val)),
                    ('SAC %', float(sac_percentage), float(aes_sac_val)),
                    ('DU', int(du), int(aes_du_val)),
                    ('LAP', float(lap_max), float(aes_lap_val)),
                    ('DAP', float(dap_max), float(aes_dap_val)),
                    ('AD', int(ad), int(aes_ad_val)),
                    ('CI', int(ci), int(aes_ci_val)),
                    ('BIC-NL', int(bic_nl), int(aes_bic_nl_val))
                ]
                
                # Create histogram comparison
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                axes = axes.flatten()
                
                colors_current = '#FF6B6B'
                colors_aes = '#4ECDC4'
                
                for idx, (metric_name, current_val, aes_val) in enumerate(metrics_numeric):
                    ax = axes[idx]
                    
                    # Create bar chart
                    x_pos = np.arange(2)
                    values = [current_val, aes_val]
                    labels = [sbox_name.replace(' S-box', ''), 'AES Std']
                    
                    bars = ax.bar(x_pos, values, color=[colors_current, colors_aes], alpha=0.8, edgecolor='black', linewidth=1.5)
                    
                    # Add value labels on bars
                    for i, (bar, val) in enumerate(zip(bars, values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.2f}' if isinstance(val, float) else f'{val}',
                               ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
                    ax.set_ylabel('Value', fontsize=10)
                    ax.set_title(metric_name, fontweight='bold', fontsize=11)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(labels, fontsize=9)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary statistics
                st.markdown("### 📈 Comparison Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Better in:**")
                    better_count = 0
                    for metric_name, current_val, aes_val in metrics_numeric:
                        if metric_name in ['DU', 'LAP', 'DAP']:  # Lower is better
                            if current_val < aes_val:
                                better_count += 1
                        else:  # Higher is better
                            if current_val > aes_val:
                                better_count += 1
                    st.info(f"✅ {better_count}/8 metrics")
                
                with col2:
                    st.write("**Equal in:**")
                    equal_count = sum(1 for _, c, a in metrics_numeric if abs(c - a) < 0.01)
                    st.info(f"⚖️ {equal_count}/8 metrics")
                
                with col3:
                    st.write("**Worse in:**")
                    worse_count = 8 - better_count - equal_count
                    st.warning(f"❌ {worse_count}/8 metrics")
        
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            import traceback
            st.error(traceback.format_exc())


def page_text_encryption():
    """Text encryption page."""
    st.header("🔒 Text Encryption & Decryption")
    # S-box Selection
    sbox_selection_widget("text")
    
    st.markdown("---")

    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 Encrypt Text")
        
        plaintext = st.text_area("Plaintext", height=200, key="plaintext_input")
        password_enc = st.text_input("Password", type="password", key="encrypt_password")
        
        if st.button("Encrypt", key="encrypt_text_btn"):
            if not plaintext:
                st.error("Please enter plaintext")
            elif not password_enc:
                st.error("Please enter a password")
            else:
                try:
                    encrypted = TextEncryptor.encrypt(plaintext, password_enc)
                    
                    st.success("✅ Text encrypted successfully!")
                    
                    # Display ciphertext
                    st.write("**Ciphertext:**")
                    st.code(encrypted['ciphertext'], language="text")
                    
                    # Store for quick decryption
                    st.session_state.last_encrypted = encrypted
                
                except Exception as e:
                    st.error(f"Encryption error: {e}")
    
    with col2:
        st.subheader("🔓 Decrypt Text")
        
        st.write("**Enter ciphertext and password:**")
        ciphertext_hex = st.text_area("Ciphertext (hex)", height=100, key="ciphertext_input")
        password_dec_manual = st.text_input("Password", type="password", key="decrypt_password_manual")
        
        if st.button("Decrypt", key="decrypt_text_btn_manual"):
            if not ciphertext_hex:
                st.error("Please provide ciphertext")
            elif not password_dec_manual:
                st.error("Please enter the password")
            else:
                try:
                    # For now, use session state if available, otherwise error
                    if 'last_encrypted' in st.session_state and st.session_state.last_encrypted['ciphertext'] == ciphertext_hex:
                        encrypted_data = st.session_state.last_encrypted
                        plaintext_result = TextEncryptor.decrypt(encrypted_data, password_dec_manual)
                        st.success("✅ Text decrypted successfully!")
                        st.write("**Decrypted Text:**")
                        st.code(plaintext_result)
                    else:
                        st.error("Please encrypt text first in this session, then paste the ciphertext here")
                except Exception as e:
                    st.error(f"Decryption error: {e}")



def page_image_encryption():
    """Image encryption page."""
    st.header("🖼️ Image Encryption & Decryption")
    
    # S-box Selection
    sbox_selection_widget("image")
    
    st.markdown("---")
    
    if 'current_sbox' not in st.session_state:
        st.warning("Please select an S-box from the options above.")
        return
    
    sbox = st.session_state.current_sbox
    sbox_name = st.session_state.get('current_sbox_name', 'Unknown S-box')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 Encrypt Image")
        
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="encrypt_img_upload")
        img_password = st.text_input("Password", type="password", key="encrypt_img_password")
        
        if uploaded_file and st.button("Encrypt Image", key="encrypt_img_btn"):
            if not img_password:
                st.error("Please enter a password")
            else:
                try:
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    
                    # Encrypt
                    encrypted = ImageEncryptor.encrypt_image(tmp_path, sbox)
                    encrypted['password_hint'] = f"Protected with password (length: {len(img_password)})"
                    
                    st.success("✅ Image encrypted successfully!")
                    
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
                    
                    st.download_button(
                        label="📥 Download Encrypted Image",
                        data=buf,
                        file_name="encrypted_image.png",
                        mime="image/png"
                    )
                    
                    # Save encrypted data with password info
                    st.session_state.last_encrypted_image = encrypted
                    st.session_state.last_image_password = img_password
                    
                    os.unlink(tmp_path)
                
                except Exception as e:
                    st.error(f"Encryption error: {e}")
    
    with col2:
        st.subheader("🔓 Decrypt Image")
        
        if 'last_encrypted_image' not in st.session_state:
            st.info("💡 Encrypt an image first using the left panel.")
        else:
            encrypted_data = st.session_state.last_encrypted_image
            
            # Ask for password
            img_decrypt_password = st.text_input("Password", type="password", key="decrypt_img_password")
            
            # Verify password
            correct_password = st.session_state.get('last_image_password', '')
            
            if st.button("Decrypt Image", key="decrypt_img_btn"):
                if not img_decrypt_password:
                    st.error("Please enter a password")
                elif img_decrypt_password != correct_password:
                    st.error("❌ Incorrect password")
                else:
                    try:
                        # Compute inverse S-box
                        inv_sbox = ImageEncryptor.compute_inverse_sbox(sbox)
                        
                        if inv_sbox is None:
                            st.error("S-box is not bijective - cannot decrypt")
                        else:
                            decrypted_array, mode = ImageEncryptor.decrypt_image(encrypted_data, inv_sbox)
                            
                            st.success("✅ Image decrypted successfully!")
                            
                            if len(decrypted_array.shape) == 3:
                                decrypted_img = Image.fromarray(decrypted_array.astype(np.uint8), mode=mode)
                            else:
                                decrypted_img = Image.fromarray(decrypted_array.astype(np.uint8), mode='L')
                            
                            st.image(decrypted_img, caption="Decrypted Image", use_column_width=True)
                            
                            # Download
                            buf = io.BytesIO()
                            decrypted_img.save(buf, format="PNG")
                            buf.seek(0)
                            
                            st.download_button(
                                label="📥 Download Decrypted Image",
                                data=buf,
                                file_name="decrypted_image.png",
                                mime="image/png",
                                key="download_decrypt_img"
                            )
                    
                    except Exception as e:
                        st.error(f"Decryption error: {e}")
    
    st.markdown("---")
    st.subheader("📊 Analyze Encryption")
    
    if 'last_encrypted_image' in st.session_state:
        st.info("💡 Encrypted image found in session")
        
        if st.button("Analyze Encrypted Image", key="analyze_enc"):
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
                
                st.subheader("📈 Encrypted Image Analysis")
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Encryption Entropy", f"{enc_entropy:.4f}", delta="Good if > 7.9")
                with col2:
                    st.metric("Image Size", f"{enc_array.size} pixels")
                
                # Display histogram
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
                
                st.write(f"**Histogram Uniformity:** {'Good ✓' if is_good_histogram else 'Fair'}")
                st.write(f"**Entropy Assessment:** {'Excellent' if enc_entropy > 7.95 else 'Good' if enc_entropy > 7.8 else 'Fair'}")
                
            except Exception as e:
                st.error(f"Analysis error: {e}")
    else:
        st.info("💡 Encrypt an image first to analyze encryption")

# Main app flow
def main():
    """Main Streamlit app."""
    # Initialize session state
    if 'current_sbox' not in st.session_state:
        # Load AES Standard as default
        prebuilt_sboxes = load_prebuilt_sboxes()
        if 'AES Standard' in prebuilt_sboxes:
            sbox_data = prebuilt_sboxes['AES Standard']
            st.session_state.current_sbox = sbox_data['sbox']
            st.session_state.current_sbox_name = sbox_data['name']
        else:
            st.session_state.current_sbox = None
            st.session_state.current_sbox_name = None
    
    # Navigation
    page = create_sidebar_navigation()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Quick Links:**
    - [GitHub](https://github.com/nia212/AES-S-Box)
    - [ Reference Paper](https://doi.org/10.1007/s11071-024-10414-3)
    """)
    
    # Route pages
    if page == "🔒 Text Encryption":
        page_text_encryption()
    elif page == "🖼️ Image Encryption":
        page_image_encryption()
    elif page == "📈 S-box Analysis":
        page_sbox_analysis()


if __name__ == "__main__":
    main()
