import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Decision Tree Visualizer", layout="wide")

# Sidebar Controls
st.sidebar.title("Decision Tree Classifier")

criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])

max_depth_input = st.sidebar.number_input("Max Depth (0 for None)", min_value=0, max_value=50, value=3, step=1)
max_depth = None if max_depth_input == 0 else int(max_depth_input)

min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=375, value=2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=375, value=1)
max_features_input = st.sidebar.slider("Max Features", min_value=1, max_value=2, value=2)
max_features = None if max_features_input == 2 else max_features_input

max_leaf_nodes_input = st.sidebar.number_input("Max Leaf Nodes (0 for None)", min_value=0, max_value=100, value=0, step=1)
max_leaf_nodes = None if max_leaf_nodes_input == 0 else int(max_leaf_nodes_input)

min_impurity_decrease = st.sidebar.number_input("Min Impurity Decrease", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

run_btn = st.sidebar.button("Run Algorithm")

# Data Generation
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if run_btn or 'has_run' not in st.session_state:
    st.session_state['has_run'] = True

    # Model Train
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 1. Decision Boundary Plotting
    fig, ax = plt.subplots(figsize=(7, 5))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8, s=30)
    ax.set_xlabel("Col1", fontsize=12)
    ax.set_ylabel("Col2", fontsize=12)
    ax.set_facecolor('#f7f7f7')
    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Decision Boundary")
        st.pyplot(fig)
        st.markdown(f"### Accuracy: `{acc:.2f}`")

    # 2. Decision Tree Graph Generation
    with col2:
        st.subheader("Tree Structure")
        dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=["Col1", "Col2"],
            class_names=["0", "1"],
            filled=False,
            rounded=True,
            special_characters=True
        )
        st.graphviz_chart(dot_data)