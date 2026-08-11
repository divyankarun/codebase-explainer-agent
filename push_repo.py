import subprocess
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

def run(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    return res

run("git branch -M main")
run("git add .")
run('git commit -m "Initial commit: Codebase Explainer Agent codebase"')

remote_url = f"https://x-access-token:{token}@github.com/divyankarun/codebase-explainer-agent.git"
run(f"git remote add origin {remote_url}")

push_res = run("git push -u origin main")

# Clean up remote URL so token isn't stored in .git/config
run("git remote set-url origin https://github.com/divyankarun/codebase-explainer-agent.git")

if push_res.returncode == 0:
    print("\nSUCCESS: Code successfully pushed to https://github.com/divyankarun/codebase-explainer-agent")
else:
    print("\nFAILED to push")
