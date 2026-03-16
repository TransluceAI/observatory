"""Generate multiple choice question stems for each attribute class using an LLM."""

import asyncio

from latentqa.mc_util import generate_mc_stems


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    await generate_mc_stems(save_dir=args.save_dir)


if __name__ == "__main__":
    asyncio.run(main())
