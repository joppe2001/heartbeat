import subprocess
import time
import socket


class MLflowServer:
    def __init__(self, db_path="mads_exam.db", artifact_root="mlruns", host="127.0.0.1",
                 port=5001):  # Changed port to 5001
        self.db_path = db_path
        self.artifact_root = artifact_root
        self.host = host
        self.port = port
        self.process = None

    def _is_port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0


    def start(self):
        """Start MLflow server"""
        if self._is_port_in_use():
            raise RuntimeError(
                f"Port {self.port} is already in use. Please use a different port or stop the existing process.")

        print(f"Starting MLflow server on port {self.port}...")
        command = [
            "mlflow", "server",
            "--backend-store-uri", f"sqlite:///{self.db_path}",
            "--default-artifact-root", self.artifact_root,
            "--host", self.host,
            "--port", str(self.port)
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to start
        for _ in range(20):  # Wait up to 10 seconds
            if self._is_port_in_use():
                print(f"MLflow server running at http://{self.host}:{self.port}")
                return
            time.sleep(0.5)

        raise RuntimeError("MLflow server failed to start")

    def stop(self):
        """Stop MLflow server"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("MLflow server stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def setup_tracking(experiment_name: str):
    """Setup MLflow tracking"""
    import mlflow
    mlflow.set_tracking_uri(f"http://127.0.0.1:5001")  # Changed port to 5001
    mlflow.set_experiment(experiment_name)