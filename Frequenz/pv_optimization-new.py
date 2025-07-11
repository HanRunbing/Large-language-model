import asyncio
from datetime import timedelta
from frequenz.sdk import microgrid
from frequenz.sdk.actor import ResamplerConfig

async def run():
    await microgrid.initialize(
        "microgrid.sandbox.api.frequenz.io", 62060,
        ResamplerConfig(resampling_period=timedelta(seconds=1)),
    )

    lm = microgrid.logical_meter()
    power_stream = lm.grid_power.new_receiver()

    async for power in power_stream:
        print(f"Grid power: {power.value} W")

asyncio.run(run())
