import argparse

from user_modeling.self_describe import create_self_describe_dataset


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereotypes_path", type=str)
    parser.add_argument("--subject_hf_model_id", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--subject_client_url", type=str)
    args = parser.parse_args()

    await create_self_describe_dataset(**vars(args))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
