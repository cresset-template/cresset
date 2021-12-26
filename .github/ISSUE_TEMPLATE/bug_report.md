---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: veritas9872

---

**Pre-Issue Checklist**
Before submitting an issue, please check that you have tried the following.
Please raise an issue after making a reasonable attempt to solve the problem.
Reports of genuine bugs and well-formed proposals are more than welcome.

1. Read the `Known Issues` section of the `README`.
2. Google your error message.
3. Check that you NVIDIA driver and Docker/Docker Compose installations are up-to-date and properly configured. The NVIDIA driver is especially easy to get wrong.
4. Reboot your computer/server. If this is not possible, restart Docker.
5. Remove all pre-existing pytorch_source:* images, run `docker system prune` to remove all docker caches, then run the *-clean commands for clean builds.
6. Try on another computer/server.

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:


**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Host information (please complete the following information):**
 - Host OS: [e.g., Ubuntu 20.04 LTS]
 - NVIDIA Driver version: 
 - Image CUDA version:
 - Docker/Compose Version:

**Additional context**
Add any other context about the problem here.
