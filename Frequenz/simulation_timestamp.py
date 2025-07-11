import asyncio
from random import uniform
from datetime import datetime

class PowerSample:
    def __init__(self, value: float, timestamp: datetime):
        self.value = value
        self.timestamp = timestamp

async def mock_power_stream():
    while True:
        power_value = uniform(10, 90)  # Simulate power in watts
        timestamp = datetime.utcnow()
        yield PowerSample(power_value, timestamp)
        await asyncio.sleep(1)

async def main():
    async for sample in mock_power_stream():
        print(f"[{sample.timestamp.isoformat()}] Mock power: {sample.value:.2f} W")

if __name__ == "__main__":
    asyncio.run(main())