from nplm_trainer import train
from eval import run_final_eval
from nplm_inference import generate
from view_metadata import view


MODELS = {
    "1": ("Context=3, 1 hidden layer", "model/nplm_ctx3_2hid0.pth"),
    "2": ("Context=5, 1 hidden layer", "model/nplm_ctx5_2hid0.pth"),
    "3": ("Context=3, 2 hidden layers", "model/nplm_ctx3_2hid1.pth"),
    "4": ("Context=5, 2 hidden layers", "model/nplm_ctx5_2hid1.pth"),
    "5": ("ALL models", None)
}


def select_model():
    print("\nSelect model:")
    for k, (desc, _) in MODELS.items():
        print(f"{k}. {desc}")

    choice = input("Enter choice: ").strip()
    return MODELS.get(choice, (None, None))


def main():
    while True:
        print("\n====== NPLM MENU ======")
        print("1. Evaluate model on input file")
        print("2. Generate text from seed file")
        print("3. Train models")
        print("4. View losses/metadata")
        print("5. Exit")

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            input_file = input("Enter file path: ").strip()
            print("1. Train / Validation split")
            print("2. Test only")

            mode = input("Enter choice (1 or 2): ").strip()

            if mode == "1":
                is_test = False
            elif mode == "2":
                is_test = True
            else:
                print("Invalid choice. Defaulting to Train/Validation.")
                is_test = False


            desc, model_file = select_model()

            if model_file is not None:
                run_final_eval(model_file, input_file, is_test)
            else:
                for _, model_file in MODELS.values():
                    if model_file is not None:
                        run_final_eval(model_file, input_file, is_test)

        elif choice == "2":
            seed_file = input("Enter file path: ").strip()
            k = int(input("Enter generation length k: ").strip())
            desc, model_file = select_model()

            if model_file is not None:
                generate(seed_file, model_file, k)
            else:
                for _, model_file in MODELS.values():
                    if model_file is not None:
                        generate(seed_file, model_file, k)

        elif choice == "3":
            print("\nTraining all models...")
            train()

        elif choice == "4":
            print("Losses/Meta data:")
            view()

        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
