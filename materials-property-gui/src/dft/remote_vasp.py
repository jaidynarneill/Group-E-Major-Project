import shlex


SSH_HOST = "m3.massive.org.au"
MODULEFILE_DIRECTORY = "/projects/lh36/sdwi0002/opt/modulefiles"
REMOTE_SCRATCH_DIRECTORY = "/fs04/scratch2/he41/temp_runs/jp_script/vasp"
VASP_MODULE = "vasp/6.6.1"


def _cluster_inspection_command():
    script = "\n".join((
        f"module use {shlex.quote(MODULEFILE_DIRECTORY)}",
        f"module load {shlex.quote(VASP_MODULE)}",
        "printf 'VASP_EXECUTABLE='; command -v vasp_std || true",
        "printf 'PYTHON_EXECUTABLE='; command -v python3 || true",
        "python3 -c 'import ase, numpy, scipy; print(\"PYTHON_PACKAGES=ase,numpy,scipy available\")' 2>&1 || true",
        f"find {shlex.quote(REMOTE_SCRATCH_DIRECTORY)} -maxdepth 4 -type f -name POTCAR -print 2>/dev/null | head -20",
    ))
    return "bash -lc " + shlex.quote(script)


def inspect_cluster(username, password, client_factory=None):
    """Authenticate without saving credentials and inspect the VASP runtime."""
    username = username.strip()
    if not username or not password:
        raise ValueError("Enter both your cluster username and password.")

    if client_factory is None:
        import paramiko

        client_factory = paramiko.SSHClient

    client = client_factory()
    sftp = None
    try:
        client.load_system_host_keys()
        import paramiko

        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=SSH_HOST,
            username=username,
            password=password,
            timeout=15,
            auth_timeout=20,
            banner_timeout=20,
            allow_agent=False,
            look_for_keys=False,
        )
        sftp = client.open_sftp()
        try:
            entries = sorted(sftp.listdir(REMOTE_SCRATCH_DIRECTORY))
        except OSError as error:
            entries = [f"Could not list scratch directory: {error}"]

        _, stdout, stderr = client.exec_command(_cluster_inspection_command(), timeout=60)
        output = stdout.read().decode("utf-8", errors="replace").strip()
        errors = stderr.read().decode("utf-8", errors="replace").strip()
        exit_status = stdout.channel.recv_exit_status()
        return {
            "host": SSH_HOST,
            "scratch_directory": REMOTE_SCRATCH_DIRECTORY,
            "scratch_entries": entries,
            "command_output": output,
            "command_errors": errors,
            "command_exit_status": exit_status,
        }
    finally:
        if sftp is not None:
            sftp.close()
        client.close()