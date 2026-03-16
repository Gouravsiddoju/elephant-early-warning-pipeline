import paramiko
import sys

def check_gpu(host, username, password):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username, password=password)
        
        stdin, stdout, stderr = ssh.exec_command("nvidia-smi")
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if output:
            print("--- NVIDIA-SMI OUTPUT ---")
            print(output)
        if error:
            print("--- ERROR ---")
            print(error)
            
        ssh.close()
    except Exception as e:
        print(f"FAILED to connect or execute: {e}")

if __name__ == "__main__":
    check_gpu("192.168.1.117", "to-00090", "Siddarth@90")
