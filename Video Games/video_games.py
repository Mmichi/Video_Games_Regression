import os
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Ayarlar
KERAS_MODEL_PATH = "model/video_games_model.h5"
DATA_CSV = "vgsales.csv"

st.title("Video Game Sales Prediction — Demo (preprocessor.fit uygulamada)")

# Model yükle
if not os.path.exists(KERAS_MODEL_PATH):
    st.error(f"Keras modeli bulunamadı: {KERAS_MODEL_PATH}")
    st.stop()
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# Veri yükle (preprocessor'ı fit etmek için)
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV)
else:
    st.warning(f"{DATA_CSV} bulunamadı — demo için varsayılan seçenekler kullanılacak.")
    df = pd.DataFrame()

# UI
Rank = st.selectbox('Rank', df['Rank'].unique() if 'Rank' in df.columns else [1,2,3])
NA_Sales = st.number_input('NA Sales', min_value=0.0, step=0.1, value=0.0)
EU_Sales = st.number_input('EU Sales', min_value=0.0, step=0.1, value=0.0)
JP_Sales = st.number_input('JP Sales', min_value=0.0, step=0.1, value=0.0)
Other_Sales = st.number_input('Other Sales', min_value=0.0, step=0.1, value=0.0)
Genre = st.selectbox('Genre', df['Genre'].unique() if 'Genre' in df.columns else ['Action','Sports'])
Platform = st.selectbox('Platform', df['Platform'].unique() if 'Platform' in df.columns else ['PS4','PC'])
How_Old = st.number_input('How Old (release year)', min_value=1970, max_value=2040, step=1, value=2000)

input_df = pd.DataFrame([{
    'Rank': Rank,
    'NA_Sales': NA_Sales,
    'EU_Sales': EU_Sales,
    'JP_Sales': JP_Sales,
    'Other_Sales': Other_Sales,
    'Genre': Genre,
    'Platform': Platform,
    'How_Old': How_Old
}])

st.write("Input:", input_df)

# Preprocessor oluştur ve (demo amaçlı) fit et
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Genre','Platform'])
    ],
    remainder='drop'
)

# Fit preprocessor kullanılabilir veri var ise, yoksa fit hata verir — bu yüzden kontrol
if not df.empty and set(['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Genre','Platform']).issubset(df.columns):
    preprocessor.fit(df[['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Genre','Platform']])
else:
    st.warning("vgsales.csv yok veya sütunlar eksik; preprocessor demo amaçlı rastgele fit ediliyor (sonuçlar tutarsız olabilir).")
    # Basit fit: tek satırlık input ile fit (çok kötü ama demo için)
    preprocessor.fit(input_df[['Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Genre','Platform']])

# Transform ve predict
X_proc = preprocessor.transform(input_df)
pred = model.predict(X_proc)
pred_value = float(np.asarray(pred).ravel()[0])
st.success(f'Predicted Global Sales (demo): {pred_value:.3f}')