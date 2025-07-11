import asyncio
from random import uniform
from datetime import datetime, timedelta
from collections import deque
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 模拟数据样本类
class PowerSample:
    def __init__(self, value: float, timestamp: datetime):
        self.value = value
        self.timestamp = timestamp


async def mock_power_stream():
    start_time = datetime.utcnow()
    i = 0
    while True:
        current_time = start_time + timedelta(minutes=i)
        
        base_power = math.sin(2 * math.pi * i / 30) * 10 + 50
        noise = np.random.normal(0, 0.5)
        power_value = base_power + noise

        yield PowerSample(power_value, current_time)
        i += 1

        await asyncio.sleep(1)  


# LSTM模型定义
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_len=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # 只用最后一个时间步的输出
        return out


model = LSTM()
model.load_state_dict(torch.load("lstm_model.pth"))  # 你先要准备好训练好的模型文件
model.eval()

buffer = deque(maxlen=30)  # 改成30条
timestamps = deque(maxlen=30)

scaler = MinMaxScaler()
scaler_fitted = False


async def forecast_next_10min(power_series, time_series):
    global scaler_fitted

    data = np.array(power_series).reshape(-1, 1)
    if not scaler_fitted:
        scaler.fit(data)
        scaler_fitted = True
    data_scaled = scaler.transform(data)

    input_seq = torch.tensor(data_scaled[-30:].reshape(1, 30, 1), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_seq).numpy().flatten()

    predicted_power = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
    future_times = [time_series[-1] + timedelta(minutes=i + 1) for i in range(10)]

    plt.figure(figsize=(10, 4))
    plt.plot(time_series, power_series, label="Actual")
    plt.plot(future_times, predicted_power, label="Forecast", marker="o")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.title("Real-Time LSTM Power Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast.png")
    plt.close()


async def main():
    async for sample in mock_power_stream():
        power = sample.value
        timestamp = sample.timestamp

        print(f"[{timestamp.isoformat()}] Mock power: {power:.2f} W")

        buffer.append(power)
        timestamps.append(timestamp)

        if len(buffer) == 30:
            await forecast_next_10min(list(buffer), list(timestamps))


if __name__ == "__main__":
    asyncio.run(main())
