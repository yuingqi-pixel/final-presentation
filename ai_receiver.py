# 安裝套件:
# pip install pandas scikit-learn matplotlib numpy

import numpy as np   #產生隨機數字
import pandas as pd  #建立表格資料
import socket        #接收IoT資料
import time          #控制時間

from sklearn.ensemble import RandomForestClassifier    #AI核心載入AI模型
from sklearn.model_selection import train_test_split   #訓練測試資料
from sklearn.metrics import accuracy_score             #計算準確率

# -----------------------------
# 1️⃣ 建立模擬訓練資料
# -----------------------------

np.random.seed(42)      #固定亂數，每次執行都一樣方便測試比較結果
N = 500                 #建500筆資料

df = pd.DataFrame({
    "amount": np.random.randint(100, 50000, N),
    "time_diff": np.random.randint(1, 3600, N),
    "location_diff": np.random.binomial(1, 0.2, N),
    "device_change": np.random.binomial(1, 0.15, N)
})

# 建立詐騙機率

base_prob = 0.005  #基本詐騙率0.5%

fraud_p = (
    base_prob
    + 0.08 * (df["amount"] > 30000).astype(float)
    + 0.15 * df["location_diff"]
    + 0.25 * df["device_change"]
    + 0.10 * (df["time_diff"] <= 30).astype(float)
)

fraud_p = np.clip(fraud_p, 0, 0.9)

df["fraud"] = np.random.binomial(1, fraud_p)   #決定哪筆是詐騙

print(f"詐騙比例: {df['fraud'].mean()*100:.2f}%")

# -----------------------------
# 2️⃣ 切訓練/測試資料
# -----------------------------

X = df[["amount", "time_diff", "location_diff", "device_change"]]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# -----------------------------
# 3️⃣ 訓練模型
# -----------------------------

model = RandomForestClassifier(
    n_estimators=120,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# 測試模型

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n=== 模型準確率 ===")
print(f"Accuracy: {acc:.3f}")


# -----------------------------
# 4️⃣ IoT資料接收（重點）
# -----------------------------

HOST = "0.0.0.0"
PORT = 5000

server = socket.socket()
server.bind((HOST, PORT))         #建立伺服器建立資料
server.listen(1)

print("\n等待 IoT 資料傳入...")

conn, addr = server.accept()

print("IoT裝置已連線:", addr)

# -----------------------------
# 5️⃣ 接收資料 + AI預測
# -----------------------------

while True:

    data = conn.recv(1024).decode()

    if not data:
        break

    print("\n收到資料:", data)

    try:

        amount, time_diff, location_diff, device_change = map(
            float,
            data.split(",")
        )

        new_tx = pd.DataFrame([{
            "amount": amount,
            "time_diff": time_diff,
            "location_diff": location_diff,
            "device_change": device_change
        }])

        prob = model.predict_proba(new_tx)[0][1]

        # 風險分類

        if prob >= 0.7:
            risk = "🔺 高風險"
        elif prob >= 0.3:
            risk = "🟡 中風險"
        else:
            risk = "🟢 低風險"

        print(f"詐騙機率：{prob*100:.2f}%")
        print(f"風險等級：{risk}")
        print(f"💰 金額: {amount}")
        print(f"⏱ 時間差: {time_diff}")
        print(f"📍 地點變化: {location_diff}")
        print(f"📱 裝置變更: {device_change}")
        print(f"🚨 詐騙機率: {prob*100:.2f}%")
        print(f"⚠️ 風險等級: {risk}")
        print("="*40)

    except:
        print("資料格式錯誤")
