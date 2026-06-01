import subprocess

threads = [1, 2, 4, 8]

for t in threads:
    print(f"\n=== Threads: {t} ===")

    cmd = [
        r"C:\et\executorch\build\Release\executor_runner.exe",
        "-model_path", r"C:\et\models\mv2_xnnpack_fp32.pte",
        "-num_executions", "50",
        "-cpu_threads", str(t),
        "-print_output", "none"
    ]

    subprocess.run(cmd)