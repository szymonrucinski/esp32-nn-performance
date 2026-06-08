FROM espressif/idf:release-v5.3

# Workspace lives here when mounted from the host
WORKDIR /workspace

# Drop into a shell by default
CMD ["/bin/bash"]
