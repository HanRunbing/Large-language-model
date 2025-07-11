import asyncio
from random import uniform

async def mock_power_stream():
    while True:
        # 模拟功率值，范围随意设定
        power_value = uniform(0, 100)  
        yield power_value
        await asyncio.sleep(1)  # 每秒产一次数据

async def main():
    async for power in mock_power_stream():
        print(f"Mock power: {power:.2f} W")
        # 这里可以加条件终止
        # 比如演示只跑10次
        # if some_condition:
        #     break

if __name__ == "__main__":
    asyncio.run(main())
