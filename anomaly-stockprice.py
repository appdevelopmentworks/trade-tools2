import streamlit as st
import yfinance as yf
import pandas as pd
from pycaret.anomaly import AnomalyExperiment
from PIL import Image
import datetime

from stocklib import mystock_lib as mystc# 自作ライブラリのインポート

s = AnomalyExperiment()

opm =['Angle-base Outlier Detection',
      'Clustering-Based Local Outlier',
      'Connectivity-Based Local Outlier',
      'Isolation Forest',
      'Histogram-based Outlier Detection',
      'K-Nearest Neighbors Detector',
      'Local Outlier Factor',
      'One-class SVM detector',
      'Principal Component Analysis',
      'Minimum Covariance Determinant',
      'Subspace Outlier Detection',
      'Stochastic Outlier Selection	'
    ]


ops = ['Close','High','Low','Open','Volume']

def get_stock_data(tikker, select_col=ops, start_date="2015-01-01"):
    """
    Fetch stock data from Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, progress=False)
    df.columns = [col[0] for col in df.columns]
    stcdata = df[select_col]

    return stcdata
    

#######################################################################
st.title("株価の異常値検知")

st.text("今日の株価の異常値を検知します。")
image = Image.open("./images/headeranomaly.png")
st.image(image)
st.caption("リーマンショック以降のデータから株価の異常値を検知します。")

col1, col2 = st.columns(2)
with col1:
    ticker = mystc.checkTicker(st.text_input("株価コード", "^N225"))
with col2:
    selectModel = st.selectbox("モデル", options=opm, index=3)

chk_states ={}
chk_select = []
col3, col4, col5, col6, col7 = st.columns(5)
with col3:
    chk_states[ops[0]] = st.checkbox(ops[0], value=True)
with col4:
    chk_states[ops[1]] = st.checkbox(ops[1], value=True) 
with col5:
    chk_states[ops[2]] = st.checkbox(ops[2], value=True)
with col6:
    chk_states[ops[3]] = st.checkbox(ops[3], value=True)
with col7:
    chk_states[ops[4]] = st.checkbox(ops[4], value=False)


#選択されているチェックボックスを配列に格納
for op, state in chk_states.items():
    if state:
        chk_select.append(op)

#検出ボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #ff0000;  /* 背景色 */
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
btnpred = st.button("異常値検出")

#株価データを取得

dfstock = get_stock_data(ticker, chk_select)
st.dataframe(dfstock, width=1000, height=200)
#AnomalyExperimentのセットアップ
s.setup(dfstock, session_id=123)

if btnpred:
    #モデルIDを取得
    modelID =s.models()[s.models()["Name"] == selectModel].index[0]
    stus = st.status("モデルの学習状況")
    #モデルの作成
    model = s.create_model(modelID)
    res = s.assign_model(model)
    stus.text("グラフ描画中...")
    s.plot_model(model, plot='tsne', display_format='streamlit')
    stus.text("検出データフレーム作成中...")
    pred = s.predict_model(model, data=dfstock)
    st.write("異常値検出結果データセット")
    st.dataframe(pred.tail(20), width=1000, height=600)
    stus.success("異常値検出完了")
