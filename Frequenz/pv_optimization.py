import asyncio
from datetime import timedelta
from frequenz.sdk import microgrid
from frequenz.sdk.actor import ResamplerConfig

async def run() -> None:
    await microgrid.initialize(
        host="microgrid.sandbox.api.frequenz.io",
        port=62060,
        resampler_config=ResamplerConfig(resampling_period=timedelta(seconds=1)),
    )

    grid_meter = microgrid.grid().power.new_receiver()

    async for power in grid_meter:
        print(power.value)

def main() -> None:
    asyncio.run(run())

if __name__ == "__main__":
    main()
