import os
import sys

# Try to import paramiko for SSH automation
try:
    import paramiko
except ImportError:
    print("Installing paramiko for SSH automation...")
    os.system(f"{sys.executable} -m pip install paramiko")
    import paramiko

# Server Credentials (from user)
HOST = "192.168.1.117"
USER = "to-00090"
PASS = "Siddarth@90"

def setup_remote():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print(f"Connecting to {USER}@{HOST}...")
    try:
        client.connect(HOST, username=USER, password=PASS)
        print("Connected successfully!")
        
        # 1. Update and install basic deps
        print("Updating package lists and checking for drivers...")
        stdin, stdout, stderr = client.exec_command("nvidia-smi")
        if stdout.channel.recv_exit_status() != 0:
            print("WARNING: nvidia-smi failed. Ensure NVIDIA drivers are installed on the remote.")
        else:
            print(stdout.read().decode())

        # 2. Clone repo (assuming public for now, or user has keys)
        print("Setting up project directory...")
        repo_url = "https://github.com/Gouravsiddoju/elephant-early-warning-pipeline.git"
        commands = [
            "sudo apt-get update -y",
            "sudo apt-get install -y git git-lfs python3-pip python3-venv",
            f"git clone {repo_url} || (cd elephant-early-warning-pipeline && git pull)",
            "cd elephant-early-warning-pipeline && git lfs install && git lfs pull",
            "python3 -m venv venv",
            "./venv/bin/pip install --upgrade pip",
            "./venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "./venv/bin/pip install -r elephant_early_warning_pipeline/early_warning_pipeline/requirements.txt"
        ]
        
        for cmd in commands:
            print(f"Executing: {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            # Handle possible sudo prompts (if necessary, though we use -y)
            status = stdout.channel.recv_exit_status()
            if status != 0:
                print(f"Error in command: {cmd}")
                print(stderr.read().decode())
        
        print("\n--- Setup Complete ---")
        client.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    setup_remote()
