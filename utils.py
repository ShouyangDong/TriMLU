def print_header(title):
    print("=" * 50)
    print(title)
    print("=" * 50)


def print_config(config, title="Configuration"):
    print(f"\n{title}:")
    for key, value in config.items():
        print(f"  {key}: {value}")
