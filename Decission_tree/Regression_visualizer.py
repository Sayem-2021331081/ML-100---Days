import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Title & Sidebar Header
st.sidebar.title("Decision Tree Regressor")

# 1. Hyperparameters Input in Sidebar
criterion = st.sidebar.selectbox("Criterion", ("squared_error", "absolute_error", "friedman_mse", "poisson"))
splitter = st.sidebar.selectbox("Splitter", ("best", "random"))

max_depth = st.sidebar.number_input("Max Depth", min_value=0, value=0, step=1)
max_depth = None if max_depth == 0 else max_depth

min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=150, value=2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=150, value=1)

max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes", min_value=0, value=0, step=1)
max_leaf_nodes = None if max_leaf_nodes == 0 else max_leaf_nodes

min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrease", min_value=0.0, value=0.0, step=0.01)

# Run Algorithm Button
run_btn = st.sidebar.button("Run Algorithm")

# 2. Synthetic Dataset Generation
np.random.seed(42)
X = np.sort(5 * np.random.rand(150, 1) - 5, axis=0)
y = np.sin(X).ravel() + np.exp(-X**2).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 3. Execution on Button Click
if run_btn:
    regressor = DecisionTreeRegressor(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42,
    )

    regressor.fit(X, y)
    X_test = np.arange(-5.0, 5.0, 0.01)[:, np.newaxis]
    y_pred = regressor.predict(X_test)

    # Calculate R2 Score
    r2 = r2_score(y, regressor.predict(X))
    st.write(f"### R2 score: **{r2:.2f}**")

    # 4. Regression Curve Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color="yellow", edgecolor="black", label="data points")
    ax.plot(X_test, y_pred, color="blue", linewidth=1.5, label="prediction")
    ax.set_facecolor("#f2f2f2")
    ax.grid(True, color="white", linestyle="-", linewidth=1)
    st.pyplot(fig)

    # 5. Tree Visualization Structure
    st.write("### Tree Structure")
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(regressor, filled=True, ax=ax_tree, feature_names=["X"])
    st.pyplot(fig_tree)