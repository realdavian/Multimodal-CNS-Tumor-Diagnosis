---
trigger: always_on
---

# Environment & Package Management

## **Project Architecture**

This project utilizes a **hybrid setup**:

* **Environment Isolation:** Managed via `conda`.
* **Package Management:** Managed exclusively via `uv` (as a high-performance replacement for `pip`).

## **The Golden Rule**

> **Never** execute a Python script or a `uv` command without first ensuring the `davian-py3110` environment is active. `uv` is installed locally within this environment and will fail in the base shell.

---

## 📋 Execution Protocols

### **1. Activation Requirement**

Before running any terminal command involving Python, `uv`, or project dependencies, you **must** prefix the command with the activation sequence or verify activation.

* **Target Environment:** `davian-py3110`
* **Preferred Command Pattern:** ```bash
conda activate davian-py3110 && <your_command_here>
```


```



### **2. Package Operations**

* **Strict Prohibit:** Do not use `pip install`.
* **Standard Practice:** Use `uv` for all dependency management.
* **Syncing:** If a `pyproject.toml` or `requirements.txt` is updated, use `uv pip sync` or `uv add` within the activated environment.

### **3. Verification Check**

If you are unsure if the environment is active, run:

```bash
conda info --envs

```

Ensure the asterisk `*` is next to `davian-py3110` before proceeding.

---

## 🚫 Critical Constraints

* **Do not** attempt to install packages using the system Python.
* **Do not** create new `venv` folders; we strictly stick to the `davian-py3110` conda environment.
* **Do not** skip the `conda activate` step even if the terminal appears to be in the correct directory.