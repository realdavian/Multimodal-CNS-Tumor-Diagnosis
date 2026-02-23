---
trigger: always_on
---

# Environment & Package Management

## **Project Architecture**

This project uses a **two-layer isolation pattern**:

* **Python Interpreter:** Provided by the shared `davian-py3110` conda environment.
* **Project Packages:** Isolated in a `.venv` inside the project root, managed by `uv`.

> **Why both?** `davian-py3110` is a shared conda env for all Python 3.11 projects on this machine.
> The `.venv` holds packages specific to *this* project so they don't pollute other projects.

---

## The Golden Rule

> **Never** run a Python script or `uv` command without activating **both** layers first.

The easiest way to do this is via the project's activation script:

```bash
source activate.sh
```

---

## Execution Protocols

### **1. Activation Requirement**

Always activate both layers before running any Python or `uv` command.

* **Recommended (uses the project script):**
  ```bash
  source activate.sh && <your_command_here>
  ```

* **Manual equivalent:**
  ```bash
  conda activate davian-py3110 && source .venv/bin/activate && <your_command_here>
  ```

### **2. Package Operations**

* **Strictly prohibited:** `pip install`
* **Standard practice:** Use `uv` for all dependency management within the activated environment.
* **Syncing:** After updating `pyproject.toml` or `requirements.txt`, run:
  ```bash
  uv pip sync
  ```

### **3. Verification Check**

To confirm both layers are active:

```bash
# Check conda env (look for * next to davian-py3110)
conda info --envs

# Check venv (should point to .venv inside the project)
which python
```

---

## Critical Constraints

* **Do not** use `pip install`. Use `uv` exclusively.
* **Do not** create additional venv folders outside the project root.
* **Do not** skip activation — the `.venv` packages will be missing without it.
* **Do not** use the system Python.
