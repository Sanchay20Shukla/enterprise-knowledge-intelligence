import argparse
from pathlib import Path
from .config import Settings
from .service import Service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['ingest', 'chat'])
    parser.add_argument('value')
    parser.add_argument('--strategy', default='recursive', choices=['recursive', 'token', 'semantic'])
    args = parser.parse_args()
    service = Service(Settings())
    try:
        if args.command == 'ingest':
            path = Path(args.value)
            files = sorted(path.iterdir()) if path.is_dir() else [path]
            for file in files:
                if file.is_file():
                    print(service.ingest(file.name, file.read_bytes(), args.strategy))
        else:
            print(service.chat(args.value, 'cli').model_dump_json(indent=2))
    finally:
        service.close()


if __name__ == '__main__':
    main()
