import argparse

from user_modeling.synthsys.subject_inputs import create_synthsys_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--eps", type=float, default=0.01)
    args = parser.parse_args()

    create_synthsys_dataset(**vars(args))


if __name__ == "__main__":
    main()
