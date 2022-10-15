# Run the following commands **inside the container**
# after the image has been built.
# They do not work during the build for unknown reasons.

# For GLIBC version mismatch between the system and Anaconda.
# Remove later when Anaconda fully supports Python 3.10.
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# Solve NCurses version mismatch bug for `htop`, etc.
rm -f /opt/conda/lib/libncursesw.so.6
