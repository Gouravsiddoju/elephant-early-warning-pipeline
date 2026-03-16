import paramiko
import scp
import os

def pull_artifacts(host, username, password, remote_dir, local_dir):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username, password=password)
        
        artifacts = [
            'elephant_lstm.pt',
            'scaler.pkl',
            'label_encoder.pkl',
            'feature_names.pkl',
            'evaluation_report.png',
            'alert_output.json',
            'alert_map.html',
            'grid_centroids.csv'
        ]
        
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            for artifact in artifacts:
                remote_path = f"{remote_dir}/{artifact}"
                print(f"Pulling {artifact}...")
                try:
                    scp_client.get(remote_path, local_path=local_dir)
                except Exception as artifact_error:
                    print(f"Could not pull {artifact}: {artifact_error}")
            
        print("Retrieval complete.")
        ssh.close()
    except Exception as e:
        print(f"FAILED to retrieve: {e}")

if __name__ == "__main__":
    pull_artifacts("192.168.1.117", "to-00090", "Siddarth@90", "~/elephant_pipeline_training", ".")
