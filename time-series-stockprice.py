import streamlit as st
import yfinance as yf
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import japanize_matplotlib
import matplotlib.pyplot as plt
from PIL import Image


s = TSForecastingExperiment()


opm =['Naive Forecaster',
    'Grand Means Forecaster',
    'Seasonal Naive Forecaster',
    'Polynomial Trend Forecaster',
    'ARIMA',
    'Auto ARIMA',
    'Exponential Smoothing',
    'ETS',
    'Theta Forecaster',
    'STLF',
    'Croston',
    'BATS',
    'TBATS',
    'Linear w/ Cond. Deseasonalize & Detrending',
    'Elastic Net w/ Cond. Deseasonalize & Detrending',
    'Ridge w/ Cond. Deseasonalize & Detrending',
    'Lasso w/ Cond. Deseasonalize & Detrending',
    'Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending',
    'Bayesian Ridge w/ Cond. Deseasonalize & Detrending',
    'Huber w/ Cond. Deseasonalize & Detrending',
    'Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending',
    'K Neighbors w/ Cond. Deseasonalize & Detrending',
    'Decision Tree w/ Cond. Deseasonalize & Detrending',
    'Random Forest w/ Cond. Deseasonalize & Detrending',
    'Extra Trees w/ Cond. Deseasonalize & Detrending',
    'Gradient Boosting w/ Cond. Deseasonalize & Detrending',
    'AdaBoost w/ Cond. Deseasonalize & Detrending',
    'Light Gradient Boosting w/ Cond. Deseasonalize & Detrending']



#############################################################
def get_stock_series(code, col, start="2020-01-01"):
    """
    株価の時系列データを取得する関数
    """
    # Yahoo Financeから株価データを取得
    code= code + ".T"
    df = yf.download(code, start=start)
    df.columns =[col[0] for col in df.columns]
    print("OK!")
    df = df[col]
    # リサンプルして日次データに変換
    df = df.resample("D").mean()

    return df

def plt_pred_chart(dfseries, pred):
    """
    予測結果をプロットする関数
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dfseries.index, dfseries, label="実績値")
    plt.plot(pred.index, pred, label="予測値")
    plt.title("株価の予測")
    plt.xlabel("日付")
    plt.ylabel("株価")
    plt.legend()
    st.pyplot(plt)
    
    

#############################################################
# セッションステートに初期値を設定
if 'model' not in st.session_state:
    st.session_state['model'] = None
    
if 'bestmodel' not in st.session_state:
    st.session_state['bestmodel'] = None

#############################################################


st.title("株価の時系列予測")

st.text("時系列のアルゴリズムで株価を予測します。")
image = Image.open("./images/headertimeseries.png")
st.image(image)
st.caption("コードを入力して予測！")

col1, col2 = st.columns(2)
with col1:
    code = st.text_input("コードを入力", value="4449")
    fh = st.text_input("予測ホライズン", value=5)
    selectMthod =st.selectbox("予測方法",options=["モデルを指定する", "ベストモデル"], index=0)
with col2:
    selectcol = st.selectbox("予測列指定", options=["Close", "High", "Low", "Volume"], index=0)
    predspan = st.text_input("予測期間", value=20)
    selectModel =st.selectbox("モデルを指定",options=opm, index=0)
    chktune = st.checkbox("モデルチューニング実施", value=False)
    
col3, col4 = st.columns(2)
with col3:
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 70%;
            background-color: #0000ff;  /* 背景色 */
            color: white;  /* 文字色 */
            padding: 15px;  /* パディング */
            text-align: center;  /* テキストを中央揃え */
            text-decoration: none;  /* テキストの下線をなし */
            font-size: 16px;  /* フォントサイズ */
            border-radius: 4px;  /* 角を丸くする */
            cursor: pointer;  /* カーソルをポインタに */
        }
        </style>
        """,
        unsafe_allow_html=True
    )     
    btncreate = st.button("モデル作成")
with col4:
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 70%;
            background-color: #0000ff;  /* 背景色 */
            color: white;  /* 文字色 */
            padding: 15px;  /* パディング */
            text-align: center;  /* テキストを中央揃え */
            text-decoration: none;  /* テキストの下線をなし */
            font-size: 16px;  /* フォントサイズ */
            border-radius: 4px;  /* 角を丸くする */
            cursor: pointer;  /* カーソルをポインタに */
        }
        </style>
        """,
        unsafe_allow_html=True
    )     
    btnpred = st.button("予測実行")

#モデルの作成ボタン
if btncreate:
    dfseries = get_stock_series(code, selectcol)
    s.setup(dfseries, fh=int(fh), session_id=123, numeric_imputation_target="mean")
    modelID =s.models()[s.models()["Name"] == selectModel].index[0]
    
    if selectMthod == "モデルを指定する":
        model = s.create_model(modelID)
        st.session_state['model'] = model
        st.write(s.pull())
        s.plot_model(model, display_format="streamlit")
        print(st.session_state['model'])
        
        #チューニングチェックボックスの処理
        if chktune:
            model = s.tune_model(model)
            st.write(s.pull())
            s.plot_model(model, display_format="streamlit")
            print(model)
    
    else:   #ベストモデルを選択
        bestmodl = s.compare_models()
        st.session_state['bestmodel'] = bestmodl
        st.write("ベストモデル")
        st.write(s.pull())
        st.write("予測チャート")
        s.plot_model(bestmodl, display_format="streamlit")
        #ホールドアウトしてセッション変数に格納
        predholdout = s.predict_model(bestmodl)
        st.session_state['bestmodel'] = bestmodl
        if chktune:
            tuned_bestmodel = s.tune_model(bestmodl)
            st.session_state['bestmodel'] = tuned_bestmodel
            st.write("チューニング結果")
            st.write(s.pull())

#予測ボタン
if btnpred:
    model = None
    if st.session_state['bestmodel'] is not None:
        model = st.session_state['bestmodel']
    elif st.session_state['model'] is not None:
        model = st.session_state['model']
    else:
        st.subheader("モデルが作成されていません。")
        model = st.session_state['model']
    
    print("きてるで！") 
    pred = s.predict_model(model, fh=int(predspan))
    st.subheader("予測結果")
    st.write(pred)
    s.plot_model(model)
    