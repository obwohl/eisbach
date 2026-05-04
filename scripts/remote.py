import subprocess
import argparse
import json
import os

CONFIG_FILE = os.path.expanduser('~/.remote_config.json')

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def setup(args):
    config = load_config()
    config[args.name] = {
        'host': args.host,
        'port': args.port,
        'user': args.user,
        'key_file': args.key_file
    }
    if args.set_default or not config.get('default'):
        config['default'] = args.name
    save_config(config)
    print(f"Setup {args.name} complete.")

def get_server_details(name=None):
    config = load_config()
    name = name or config.get('default')
    if not name or name not in config:
        print("Server not found or default not set.")
        sys.exit(1)
    return config[name]

def run(args):
    server = get_server_details()
    cmd = [
        'ssh', '-p', str(server['port']),
        '-i', server['key_file'],
        '-o', 'StrictHostKeyChecking=no',
        f"{server['user']}@{server['host']}",
        args.command
    ]
    subprocess.run(cmd)

def sync(args):
    server = get_server_details()
    cmd = [
        'rsync', '-avz', '-e',
        f"ssh -p {server['port']} -i {server['key_file']} -o StrictHostKeyChecking=no",
        args.local,
        f"{server['user']}@{server['host']}:{args.remote}"
    ]
    subprocess.run(cmd)

def download(args):
    server = get_server_details()
    cmd = [
        'rsync', '-avz', '-e',
        f"ssh -p {server['port']} -i {server['key_file']} -o StrictHostKeyChecking=no",
        f"{server['user']}@{server['host']}:{args.remote}",
        args.local
    ]
    subprocess.run(cmd)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

setup_parser = subparsers.add_parser('setup')
setup_parser.add_argument('name')
setup_parser.add_argument('host')
setup_parser.add_argument('port')
setup_parser.add_argument('user')
setup_parser.add_argument('--key-file', required=True)
setup_parser.add_argument('--set-default', action='store_true')
setup_parser.set_defaults(func=setup)

run_parser = subparsers.add_parser('run')
run_parser.add_argument('command')
run_parser.set_defaults(func=run)

sync_parser = subparsers.add_parser('sync')
sync_parser.add_argument('local')
sync_parser.add_argument('remote')
sync_parser.set_defaults(func=sync)

down_parser = subparsers.add_parser('download')
down_parser.add_argument('remote')
down_parser.add_argument('local')
down_parser.set_defaults(func=download)

args = parser.parse_args()
if hasattr(args, 'func'):
    args.func(args)
else:
    parser.print_help()
