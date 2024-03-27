import subprocess
import time
import os

BASE_URL = "https://weights.replicate.delivery/default/musicgen"


class WeightsDownloader:
    def __init__(self):
        pass

    def download_weights(self, weight_str, dest="models"):
        self.download_if_not_exists(weight_str, dest)

    def download_if_not_exists(self, weight_str, dest):
        if not os.path.exists(f"{dest}/{weight_str}"):
            self.download(weight_str, dest)

    def download(self, weight_str, dest):
        url = BASE_URL + "/" + weight_str + ".tar"
        print(f"Downloading {weight_str} to {dest}")
        start = time.time()
        subprocess.check_call(
            ["pget", "--log-level", "warn", "-xf", url, dest], close_fds=False
        )
        elapsed_time = time.time() - start
        if os.path.isfile(os.path.join(dest, os.path.basename(weight_str))):
            file_size_bytes = os.path.getsize(
                os.path.join(dest, os.path.basename(weight_str))
            )
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"Downloaded {weight_str} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        else:
            print(f"Downloaded {weight_str} in {elapsed_time:.2f}s")
