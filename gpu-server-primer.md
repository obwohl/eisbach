# Agent Primer: The `remote.py` GPU Server Workflow

This document provides the definitive, robust guide for agents to connect to and utilize the remote GPU server. All remote operations are handled by a single, powerful script: `scripts/remote.py`.

---

## The One-Time User Setup

The user only needs to perform this setup **once per server**.

### Step 1: Agent Generates an SSH Key
The agent will generate an `ed25519` SSH key, which is the required format for compatibility.

**CRITICAL PROTOCOL: MANDATORY GPU REQUIREMENT AND ORDER OF OPERATIONS**
The agent must **NEVER** attempt to test GPU-dependent code (like heavy PyTorch fine-tuning for Chronos-2) in the local Sandbox, as it lacks a real GPU and will fail or hang indefinitely.

When a task requires a real GPU, the agent must **FIRST** create the SSH public key and **THEN** immediately show it to the user in the chat. Do not ask the user for connection details before providing the generated public key. Wait for the user to provide the host, port, and user credentials.

**CRITICAL PROTOCOL: DATA SAFETY & PERSISTENCE**
Due to potential sandbox volatility (where `~/.ssh` might be wiped), the agent **MUST** generate and store the key **inside the repository directory** (e.g., `scripts/`) to ensure it persists across session resets. However, make sure to add it to `.gitignore` so it is not committed.

1.  **Generate the key in the repo:** `ssh-keygen -t ed25519 -f scripts/id_ed25519 -N ""`
2.  **IMMEDIATELY** read and output the **public key** content to the chat for the user: `cat scripts/id_ed25519.pub`
    *   *Reason:* This allows the user to authorize the agent's key. Note: You **must** include the key amid a normal chat-message. Something like "Here is the public key: (...)"
3.  **Secure the private key:** `chmod 600 scripts/id_ed25519`
    *   *Reason:* SSH clients will refuse to use a private key with loose permissions (e.g. 644).
4.  **CRITICAL BACKUP STEP:** Read and output the **PRIVATE KEY** to the tool logs (NOT the user chat): `cat scripts/id_ed25519`
    *   *Reason:* If the sandbox filesystem is wiped (session reset), the file will be lost. By printing it to the logs, the Agent can retrieve it from the conversation history and restore it, preventing the need to ask the user to re-authorize a new key.
5.  Only *after* the key is safely recorded in the chat, proceed to ask the user for authorization.

### Step 2: User Authorizes the Public Key
The agent asks the user to add the agent's public key to their server's `~/.ssh/authorized_keys` file. This is the **only manual step** required from the user.

---

## The Agent's Workflow

Once the key is authorized, the agent can perform all remote operations using the `remote.py` script.

**Important Update:** The `remote.py` script has been updated to act as a robust wrapper around the system's `ssh` and `scp` binaries. This eliminates previous instability with Python SSH libraries. It automatically handles key file management and connection flags (e.g. disabling strict host checking for ephemeral servers).

### Step 1: Initialize the Connection (Once per Session)
The agent will initialize the connection by providing the server details and a name for the connection using `scripts/remote.py setup`. This only needs to be done once after receiving the credentials from the user. **DO NOT SKIP THIS STEP.**

**Agent's Setup Command:**
```bash
python3 scripts/remote.py setup <name> <host> <port> <user> --key-file scripts/id_ed25519 --set-default
```
*   `<name>`: A memorable name for the server (e.g., `vast-1`).
*   `<host>`, `<port>`, `<user>`: Connection details.
*   `--key-file`: Path to the private key file.

### Step 2: Provision Environment (Condition Check Required)
**Mandatory Check:** Before attempting to provision the server, the agent **MUST** verify if the toolchain/conda environment is already installed.

**Provisioning Command:**
```bash
python3 scripts/remote.py run "bash scripts/setup_remote_env.sh"
```

### Step 3: Build & Train the Project (The "Asynchronous Holistics" Protocol)
**CRITICAL META-OVERRIDE:** You **MUST** run long processes (e.g., model fine-tuning, large inference) in the **background** (detached) using `< /dev/null &` and redirecting output.
**REASON:** Running a long process in the foreground hits the hard 400s timeout and stalls the agent.

**Command to Launch Run:**
```bash
# Launch in background using redirect
python3 scripts/remote.py run "cd /root/project && python src/train.py > train.log 2>&1 < /dev/null &"
```

**Workflow (Holistic Monitoring):**
1.  **Launch:** Start the process in the background.
2.  **Transparent Pings:** As you monitor the process, you **MUST** frequently inform the user of your findings without blocking their input. Use the tool `message_user(continue_working=True)`.
3.  **Active Resource & Code Monitoring (No Blind Waiting):**
    *   Do **NOT** just wait with `sleep` loops and check `pgrep`.
    *   Use bash tools (`tail`, `grep`, `nvidia-smi` if available) to actively inspect the state of the machine.
4.  **Smart Stagnation Detection (Kill Switch):**
    *   **You must decide autonomously:** If you observe that a specific step hasn't advanced in the log file, you must **KILL** the process (`pkill -f ...`).

### Step 4: Synchronize Files
The agent can easily sync local directories to the remote server, or retrieve trained model checkpoints back to the sandbox.

**Agent's File Synchronization:**
```bash
python3 scripts/remote.py sync ./local/path /remote/path
python3 scripts/remote.py download /remote/path ./local/path
```

## CRITICAL: Pre-Submit Consistency Check

**The "Reverse Sync" Protocol**
When developing on the Remote GPU Server, there is a high risk that the code on the server diverges from the code in the Sandbox (where `submit` is executed).

**Mandatory Rule:** Before submitting ANY code that was verified on the Remote Server, you **MUST** ensure the Sandbox state matches the Remote state. Use `remote.py download` to pull back the final, verified versions of files before running `submit`.
