import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon

# -------------------------
# Helper to display LaTeX larger
# -------------------------
def big_latex(text):
    st.markdown(
        f"""
        <div style="font-size: 20px; margin-top:10px; margin-bottom:10px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# Plot PDF and CDF for one distribution
# -------------------------
def render_distribution_ui(title):
    st.subheader(title)

    # Choose distribution
    dist_name = st.selectbox(
        f"Choose distribution ({title})",
        ["Normal", "Uniform", "Exponential"],
        key=title + "_dist"
    )

    # Parameter inputs
    if dist_name == "Normal":
        mean = st.number_input("Mean (μ)", value=0.0, key=title + "_mean")
        sd = st.number_input("Std Dev (σ)", value=1.0, key=title + "_sd")
        dist = norm(loc=mean, scale=sd)
        x = np.linspace(mean - 4*sd, mean + 4*sd, 400)
        pdf_formula = r"$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}$"
        cdf_formula = r"$F(x)=\text{Gaussian CDF (no closed form)}$"

    elif dist_name == "Uniform":
        a = st.number_input("Lower bound (a)", value=0.0, key=title + "_a")
        b = st.number_input("Upper bound (b)", value=1.0, key=title + "_b")
        dist = uniform(loc=a, scale=b - a)
        x = np.linspace(a - 0.2*(b-a), b + 0.2*(b-a), 400)
        pdf_formula = r"$f(x)=\frac{1}{b-a}$ for a ≤ x ≤ b"
        cdf_formula = r"$F(x)=\frac{x-a}{b-a}$"

    elif dist_name == "Exponential":
        lam = st.number_input("Rate λ", value=1.0, key=title + "_lambda")
        dist = expon(scale=1/lam)
        x = np.linspace(0, 8/lam, 400)
        pdf_formula = r"$f(x)=\lambda e^{-\lambda x}$"
        cdf_formula = r"$F(x)=1-e^{-\lambda x}$"

    # Range to shade
    st.markdown("### Shade Area (Probability)")
    x_min = st.slider("Lower bound", float(x.min()), float(x.max()), float(x.min()+0.2), key=title + "_min")
    x_max = st.slider("Upper bound", float(x.min()), float(x.max()), float(x.min()+1.0), key=title + "_max")

    # PDF / CDF values
    pdf_vals = dist.pdf(x)
    cdf_vals = dist.cdf(x)
    prob = dist.cdf(x_max) - dist.cdf(x_min)

    st.markdown(f"**Probability = {prob:.4f}**")

    return x, pdf_vals, cdf_vals, dist, x_min, x_max, pdf_formula, cdf_formula


# --------------------------------------------
# ------------------- APP --------------------
# --------------------------------------------
def main():
    st.title("📊 Side-by-side Distribution Comparator")
    st.write("Compare distributions or parameter settings with **two independent columns**.")

    col1, col2 = st.columns(2)

    with col1:
        left_data = render_distribution_ui("Left Distribution")

    with col2:
        right_data = render_distribution_ui("Right Distribution")

    # Unpack
    (
        x1, pdf1, cdf1, dist1, xmin1, xmax1,
        pdf_formula1, cdf_formula1
    ) = left_data

    (
        x2, pdf2, cdf2, dist2, xmin2, xmax2,
        pdf_formula2, cdf_formula2
    ) = right_data

    # ------------------------------
    # Row 1 — PDF Comparison
    # ------------------------------
    st.markdown("## 📈 PDF Comparison Table")
    pdf_col1, pdf_col2 = st.columns(2)

    with pdf_col1:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(x1, pdf1, linewidth=2)
        fill_x = np.linspace(xmin1, xmax1, 400)
        ax1.fill_between(fill_x, dist1.pdf(fill_x), alpha=0.3)
        ax1.set_title("PDF")
        st.pyplot(fig1)
        big_latex(pdf_formula1)

    with pdf_col2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(x2, pdf2, linewidth=2)
        fill_x = np.linspace(xmin2, xmax2, 400)
        ax2.fill_between(fill_x, dist2.pdf(fill_x), alpha=0.3)
        ax2.set_title("PDF")
        st.pyplot(fig2)
        big_latex(pdf_formula2)

    # ------------------------------
    # Row 2 — CDF Comparison
    # ------------------------------
    st.markdown("## 📈 CDF Comparison Table")
    cdf_col1, cdf_col2 = st.columns(2)

    with cdf_col1:
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.plot(x1, cdf1, color="purple", linewidth=2)
        ax3.scatter([xmin1, xmax1], [dist1.cdf(xmin1), dist1.cdf(xmax1)], color="red")
        ax3.set_title("CDF")
        st.pyplot(fig3)
        big_latex(cdf_formula1)

    with cdf_col2:
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.plot(x2, cdf2, color="purple", linewidth=2)
        ax4.scatter([xmin2, xmax2], [dist2.cdf(xmin2), dist2.cdf(xmax2)], color="red")
        ax4.set_title("CDF")
        st.pyplot(fig4)
        big_latex(cdf_formula2)


if __name__ == "__main__":
    main()
