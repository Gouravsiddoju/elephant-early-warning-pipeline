import paramiko
import time

def run_remote_training(host, username, password, remote_dir):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username, password=password)
        
        # Determine the correct python command and install dependencies
        # Also ensure we use the specified GPU (GPU 2)
        commands = [
            f"cd {remote_dir}",
            "pip3 install -r requirements.txt", # Standard requirement for remote servers
            "export CUDA_VISIBLE_DEVICES=2",
            "python3 main.py"
        ]
        
        full_command = " && ".join(commands)
        print(f"Executing remote command: {full_command}")
        
        stdin, stdout, stderr = ssh.exec_command(f"bash -c '{full_command}'")
        
        # Read output in real-time
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                line = stdout.readline()
                if line:
                    try:
                        # Print normally, but fallback if Windows console chokes
                        print(f"REMOTE: {line.strip()}")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        # Force ASCII for the Windows console
                        safe_line = "".join(i for i in line.strip() if ord(i) < 128)
                        print(f"REMOTE: {safe_line}")
            time.sleep(1)
            
        # Capture any remaining output
        for line in stdout:
            try:
                print(f"REMOTE: {line.strip()}")
            except:
                safe_line = "".join(i for i in line.strip() if ord(i) < 128)
                print(f"REMOTE: {safe_line}")
        for line in stderr:
            try:
                print(f"REMOTE ERROR: {line.strip()}")
            except:
                safe_line = "".join(i for i in line.strip() if ord(i) < 128)
                print(f"REMOTE ERROR: {safe_line}")
            
        print(f"Remote process exited with status: {stdout.channel.recv_exit_status()}")
        ssh.close()
    except Exception as e:
        print(f"FAILED to run remote command: {e}")

if __name__ == "__main__":
    run_remote_training("192.168.1.117", "to-00090", "Siddarth@90", "~/elephant_pipeline_training")
