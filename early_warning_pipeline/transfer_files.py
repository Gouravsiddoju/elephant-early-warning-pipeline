import paramiko
import os
import scp

def transfer_files(host, username, password, local_paths, remote_dir):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username, password=password)
        
        # Create remote directory and subdirectories if not exists
        ssh.exec_command(f"mkdir -p {remote_dir}")
        ssh.exec_command(f"mkdir -p {remote_dir}/doi_10_5061_dryad_dr7sqv9v9__v20200116")
        
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            for local_path in local_paths:
                if os.path.isfile(local_path):
                    target = remote_dir
                    if "ElephantsData_ano.csv" in local_path:
                        target = f"{remote_dir}/doi_10_5061_dryad_dr7sqv9v9__v20200116"
                    
                    print(f"Transferring {local_path} to {target}...")
                    scp_client.put(local_path, remote_path=target)
                elif os.path.isdir(local_path):
                     print(f"Transferring directory {local_path}...")
                     scp_client.put(local_path, remote_path=remote_dir, recursive=True)
            
        print("Transfer complete.")
        ssh.close()
    except Exception as e:
        print(f"FAILED to transfer: {e}")

if __name__ == "__main__":
    local_files = [
        "main.py",
        "data_loader.py",
        "grid_builder.py",
        "feature_matrix.py",
        "model_trainer.py",
        "prediction_service.py",
        "requirements.txt",
        "predictor.py",
        "alert_engine.py",
        "gee_extractor.py",
        "human_features.py",
        "memory_features.py",
        "doi_10_5061_dryad_dr7sqv9v9__v20200116/ElephantsData_ano.csv",
        "botswana-260310-free.shp"
    ]
    # Filter to only existing files
    existing_files = [f for f in local_files if os.path.exists(f)]
    
    transfer_files("192.168.1.117", "to-00090", "Siddarth@90", existing_files, "~/elephant_pipeline_training")
