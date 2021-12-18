import subprocess


print(
    subprocess.run(
        ["fuser", "-v", '/dev/nvidia*"'], capture_output=True, text=True
    ).stdout
)
