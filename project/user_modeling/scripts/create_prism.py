import argparse

from user_modeling.prism import create_prism_gender_dataset


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--subject_client_url", type=str)
    args = parser.parse_args()

    await create_prism_gender_dataset(**vars(args))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
