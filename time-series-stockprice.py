import streamlit as st
import yfinance as yf
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import japanize_matplotlib
import matplotlib.pyplot as plt
from PIL import Image

#自作ライブラリーのインポート
from stocklib import mystock_lib as mystc

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
    df = yf.download(code, start=start)
    df.columns =[col[0] for col in df.columns]
    print("OK!")
    df = df[col]
    # リサンプルして日次データに変換
    df = df.resample("D").mean()

    return df

def plt_pred_chart(actcdata, predcdata, code):
    plt.figure(figsize=(12, 6))
    plt.title(f'{code}の株価予想')
    plt.plot(actcdata.tail(60), label='実データ')
    plt.plot(predcdata, label='時系列予想データ')
    plt.xlabel('日付')
    plt.ylabel('株価')
    plt.legend()
    return plt
    
    

#############################################################
# セッションステートに初期値を設定
if 'stockdata' not in st.session_state:
    st.session_state['stockdata'] = None

if 'tunedbestmodel' not in st.session_state:
    st.session_state['tunedbestmodel'] = None
    
if 'bestmodel' not in st.session_state:
    st.session_state['bestmodel'] = None
    
if 'model' not in st.session_state:
    st.session_state['model'] = None
    
if 'tunedtmodel' not in st.session_state:
    st.session_state['tunedtmodel'] = None
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
    selectModel =st.selectbox("モデルを指定",options=opm, index=27)
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
    stus = st.status("進捗状況を確認出来ます!")
    ticker = mystc.checkTicker(code)
    dfseries = get_stock_series(ticker, selectcol)
    st.session_state['stockdata'] = dfseries
    stus.text("V 株価データを取得")
    s.setup(dfseries, fh=int(fh), session_id=123, numeric_imputation_target="mean")
    stus.text("V セットアップ")
    #モデルIDを取得
    modelID =s.models()[s.models()["Name"] == selectModel].index[0]
    
    if selectMthod == "モデルを指定する":
        stus.text("モデルの作成中...")
        model = s.create_model(modelID)
        stus.success("モデル作成完了!")
        st.write(s.pull())
        stus.text("グラフ描画")
        s.plot_model(model, display_format="streamlit")
        stus.success("グラフ描画完了!")
        predholdout = s.predict_model(model)
        st.session_state['model'] = model
        st.write(s.pull())

        #print(st.session_state['model'])
        
        #チューニングチェックボックスの処理
        if chktune:
            stus.text("モデルのチューニング中...")
            tunemodel = s.tune_model(model)
            stus.success("チューニング完了!")
            st.subheader("チューニング結果")
            st.write(s.pull())
            stus.text("チューニング後モデルの可視化")
            s.plot_model(tunemodel, display_format="streamlit")
            stus.success("完了!")
            predholdout = s.predict_model(tunemodel)
            st.session_state['tunedtmodel'] = tunemodel
            st.write(s.pull())
            #print(tunemodel)
    
    else:   #ベストモデルを選択
        stus.text("最適なモデル探索中（時間がかかります）...")
        bestmodl = s.compare_models()
        st.session_state['bestmodel'] = bestmodl
        stus.success("ベストモデル選定!")
        st.write("ベストモデル")
        st.write(s.pull())
        stus.text("チャートの描画")
        st.write("予測チャート")
        s.plot_model(bestmodl, display_format="streamlit")
        #ホールドアウトしてセッション変数に格納
        predholdout = s.predict_model(bestmodl)
        st.session_state['bestmodel'] = bestmodl
        st.write("選ばれたモデル")
        st.write(s.pull())
        #チューニングする場合
        if chktune:
            stus.text("モデルのチューニング中...")
            tuned_bestmodel = s.tune_model(bestmodl)
            stus.success("チューニング完了!")
            st.subheader("チューニング結果")
            st.write(s.pull())
            stus.text("チューニング後モデルの可視化")
            s.plot_model(tuned_bestmodel, display_format="streamlit")
            predholdout = s.predict_model(tuned_bestmodel)
            st.session_state['tunedbestmodel'] = tuned_bestmodel
            st.write("チューニング後モデル")
            st.write(s.pull())
            stus.success("完了!")
            
            
         

#予測ボタン
if btnpred:
    #モデルを作った時と同じデータでセットアップ
    try:
        data = st.session_state['stockdata']
        s.setup(data, fh=int(fh), session_id=321, numeric_imputation_target="mean")
    except:
        st.subheader("モデルが作成されていません。")
        st.text("はじめに「モデル作成」ボタンで予測モデルを作成します")
        st.stop()
    
    if st.session_state['tunedbestmodel'] is not None:
        usemodel = st.session_state['tunedbestmodel']
    elif st.session_state['bestmodel'] is not None:
        usemodel = st.session_state['bestmodel']
    elif st.session_state['tunedtmodel'] is not None:
        usemodel = st.session_state['tunedtmodel']
    elif st.session_state['model'] is not None:
        usemodel = st.session_state['model']
    else:
        st.subheader("モデルが作成されていません。")
        model = st.session_state['model']
    
    pred = s.predict_model(usemodel, fh=int(predspan))
    st.subheader("予測結果")
    #NaNを除外
    actcdata= data.dropna()
    #実データと被らない日から初めて土日を除く
    predcdata = pred[(data.index[-1] <= pred.index.to_timestamp()) & (pred.index.weekday < 5)]
    #
    title = f'{code}  {yf.Ticker(mystc.checkTicker(code)).info["shortName"]}'
    #チャートの描画
    plt = plt_pred_chart(actcdata, predcdata, title)
    st.pyplot(plt)
    st.write("予測データ")
    st.write(s.pull())
    st.dataframe(pred, width=1000, height=300)

    