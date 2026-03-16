import paramiko
import os

# Server Credentials
HOST = "192.168.1.117"
USER = "to-00090"
PASS = "Siddarth@90"

def check_server_health():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print(f"--- Diagnostic Report for {HOST} ---")
    try:
        client.connect(HOST, username=USER, password=PASS)
        
        # 1. Check GPU Load
        print("\n[Audit] GPU Status (nvidia-smi):")
        stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader")
        print(stdout.read().decode().strip() or "No GPU detected or drivers missing.")

        # 2. Check Disk Space
        print("\n[Audit] Disk Space (Home Directory):")
        stdin, stdout, stderr = client.exec_command("df -h ~")
        print(stdout.read().decode().strip())

        # 3. Check Running Processes (to avoid collisions)
        print("\n[Audit] Top Resource Consumers:")
        stdin, stdout, stderr = client.exec_command("ps -eo pcpu,pmem,comm --sort=-pcpu | head -n 5")
        print(stdout.read().decode().strip())

        # 4. Check for existing 'elephant' folders
        print("\n[Audit] Existing Workspaces:")
        stdin, stdout, stderr = client.exec_command("ls -d */ | grep elephant || echo 'No conflicting folders found.'")
        print(stdout.read().decode().strip())

        client.close()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    check_server_health()
