import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset (for visualization)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names

# ----- Sidebar -----
st.sidebar.title("📘 Instructions")
st.sidebar.markdown("""
This app uses a Random Forest model trained on the Iris dataset.
- Input feature values
- Click 'Predict'
- See model prediction and visualizations
""")

# ----- App Title -----
st.title("🌸 Iris Flower Prediction App")
st.write("Built with Streamlit | Celebal Summer Internship")

# ----- User Input -----
st.header("🔢 Enter Input Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ----- Predict Button -----
if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.button("🚀 Predict"):
    prediction = model.predict(user_input)
    predicted_class = target_names[prediction[0]]
     
     # Stylish colored card using Markdown
    st.markdown(
        f"""
        <div style="background-color:#d4edda;padding:20px;border-radius:10px;border:1px solid #c3e6cb;">
            <h3 style="color:#155724;">🌼 Predicted Iris Species: <b>{predicted_class}</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show flower image
    image_paths = {
        "setosa": "images/setosa.jpg",
        "versicolor": "images/versicolor.jpg",
        "virginica": "images/virginica.jpg"
    }
    img_file = image_paths[predicted_class.lower()]
    st.image(img_file, caption=f"Iris Species: {predicted_class}", use_container_width=True)


    # Save to session history
    st.session_state.history.append({
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width,
        "Prediction": predicted_class
    })

     # Prepare download button
    pred_df = pd.DataFrame(user_input, columns=iris.feature_names)
    pred_df["Predicted Species"] = predicted_class
    st.download_button("⬇️ Download Result", pred_df.to_csv(index=False), "prediction.csv", "text/csv")


# Show prediction history
if st.session_state['history']:
    st.subheader("📚 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state['history']))
    
    # Reset Button
    if st.button("🔄 Reset History"):
        st.session_state['history'] = []
        st.success("Prediction history cleared!")
   

# ----- Data Overview -----
with st.expander("📊 See Dataset Sample"):
    st.write(X.head())

# ----- Visualizations -----
st.header("📈 Dataset Visualization")

# Correlation Heatmap
with st.expander("🔍 Correlation Heatmap"):
    fig, ax = plt.subplots()
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Pairplot with classes
with st.expander("🌸 Pairplot with Classes"):
    df = X.copy()
    df['target'] = y
    df['species'] = df['target'].apply(lambda i: target_names[i])
    fig2 = sns.pairplot(df, hue="species")
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("📍 Developed by **Debargha Karmakar** for Celebal Summer Internship")

