import shlex
import re


SSH_HOST = "m3.massive.org.au"
MODULEFILE_DIRECTORY = "/projects/lh36/sdwi0002/opt/modulefiles"
REMOTE_SCRATCH_DIRECTORY = "/fs04/scratch2/lh36/jarn0012/vasp"
VASP_MODULE = "vasp/6.4.2"
VASP_PREREQUISITE_MODULES = (
    "hpcx/.2.14-redhat9.2-patch1",
    "hpcx-ompi",
    "hdf5/1.12.3",
    "wannier90/3.1.0-mpi",
)
MODULE_INIT_SCRIPT = "/etc/profile.d/modules.sh"


def module_shell_command(script):
    initialized_script = f"source {shlex.quote(MODULE_INIT_SCRIPT)}\n{script}"
    return "bash --noprofile --norc -c " + shlex.quote(initialized_script)


def clean_shell_startup_warnings(text):
    warning_pattern = re.compile(
        r"^/home/[^/]+/\.bash(?:rc|_profile): line \d+: .*: No such file or directory$"
    )
    return "\n".join(
        line for line in text.splitlines() if not warning_pattern.match(line)
    ).strip()


def _module_setup_lines():
    return (
        f"module use {shlex.quote(MODULEFILE_DIRECTORY)}",
        *(f"module load {shlex.quote(module)}" for module in VASP_PREREQUISITE_MODULES),
        f"module load {shlex.quote(VASP_MODULE)}",
    )


def _cluster_inspection_command():
    script = "\n".join((
        *_module_setup_lines(),
        "printf 'VASP_EXECUTABLE='; command -v vasp_std || true; printf '\\n'",
        "printf 'VASP_PP_PATH=%s\\n' \"${VASP_PP_PATH:-}\"",
        f"printf 'STAGING_DIRECTORY={shlex.quote(REMOTE_SCRATCH_DIRECTORY)}\\n'",
        "if [ -n \"${VASP_PP_PATH:-}\" ]; then find \"$VASP_PP_PATH\" -maxdepth 6 -type f -name POTCAR -print 2>/dev/null | head -20; fi",
    ))
    return module_shell_command(script)


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
        errors = clean_shell_startup_warnings(
            stderr.read().decode("utf-8", errors="replace")
        )
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