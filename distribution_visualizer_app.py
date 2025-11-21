import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon

def distribution_visualizer_app():
    st.title("📊 Distribution Explorer – PDF, CDF, and Probability Areas")
    st.write("Interactive tool for visualizing distributions, shaded probability areas, and Z-values.")

    descriptions = {
        "Normal (mean=0, std=1)": {
            "pdf": r"$f(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}$",
            "cdf": "No closed form; computed numerically via the Gaussian integral.",
            "details": "Symmetric bell curve. Mean=0, variance=1. Z-scores come from this distribution."
        },
        "Uniform(0,1)": {
            "pdf": r"$f(x)=1$ for 0 ≤ x ≤ 1$",
            "cdf": r"$F(x)=x$ for 0 ≤ x ≤ 1$",
            "details": "All values between 0 and 1 equally likely. Mean=0.5, variance=1/12."
        },
        "Exponential(λ=1)": {
            "pdf": r"$f(x)=e^{-x}$ for x≥0$",
            "cdf": r"$F(x)=1-e^{-x}$",
            "details": "Models waiting times. Memoryless. Mean=1, variance=1."
        }
    }

    dist_name = st.selectbox(
        "Choose a distribution",
        list(descriptions.keys())
    )

    st.subheader("Distribution Details")
    st.latex(descriptions[dist_name]["pdf"])
    st.write("**CDF:**", descriptions[dist_name]["cdf"])
    st.write("**About this distribution:**", descriptions[dist_name]["details"])

    # Distribution setup
    if dist_name == "Normal (mean=0, std=1)":
        dist = norm()
        x = np.linspace(-4, 4, 400)
    elif dist_name == "Uniform(0,1)":
        dist = uniform(0, 1)
        x = np.linspace(-0.2, 1.2, 400)
    elif dist_name == "Exponential(λ=1)":
        dist = expon()
        x = np.linspace(0, 8, 400)

    pdf_vals = dist.pdf(x)
    cdf_vals = dist.cdf(x)

    st.subheader("Select Probability Region")
    x_min = st.slider("Lower bound", float(x.min()), float(x.max()), float(x.min() + 0.5))
    x_max = st.slider("Upper bound", float(x.min()), float(x.max()), float(x.min() + 2))

    if x_min >= x_max:
        st.error("Lower bound must be less than upper bound.")
        return

    prob = dist.cdf(x_max) - dist.cdf(x_min)

    st.markdown(f"""
    ### 📐 Probability Between Two Points  
    \[
    P(x_{{min}} < X < x_{{max}}) = F(x_{{max}}) - F(x_{{min}})
    \]
    **Probability = {prob:.4f}**
    """)

    # PDF Plot
    fig_pdf, ax_pdf = plt.subplots(figsize=(7,4))
    ax_pdf.plot(x, pdf_vals, linewidth=2, label="PDF")
    x_fill = np.linspace(x_min, x_max, 400)
    ax_pdf.fill_between(x_fill, dist.pdf(x_fill), alpha=0.3, label="Shaded Probability Area")
    ax_pdf.set_title("PDF - Probability Density Function")
    ax_pdf.set_xlabel("x")
    ax_pdf.set_ylabel("Density")
    ax_pdf.legend()
    st.pyplot(fig_pdf)

    # CDF Plot
    fig_cdf, ax_cdf = plt.subplots(figsize=(7,4))
    ax_cdf.plot(x, cdf_vals, color="purple", linewidth=2, label="CDF")
    ax_cdf.scatter([x_min, x_max], [dist.cdf(x_min), dist.cdf(x_max)], color="red")
    ax_cdf.vlines([x_min, x_max], 0, [dist.cdf(x_min), dist.cdf(x_max)], colors="gray", linestyles="dashed")
    ax_cdf.set_title("CDF - Cumulative Distribution Function")
    ax_cdf.set_xlabel("x")
    ax_cdf.set_ylabel("F(x)")
    ax_cdf.legend()
    st.pyplot(fig_cdf)

    st.markdown("""
    ## 🧠 Interpretation Summary

    ### PDF  
    - Shows likelihood density  
    - Probability = area under the curve  
    - The shaded region = computed probability  

    ### CDF  
    - Gives **P(X ≤ x)**  
    - The probability between bounds is **CDF(x_max) – CDF(x_min)**  

    ### Z-values (Normal distribution only)  
    - For N(0,1), Z = x  
    """)


if __name__ == "__main__":
    distribution_visualizer_app()
