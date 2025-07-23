#!/usr/bin/env bash

# Ensure .ssh directory exists
mkdir -p ~/.ssh

# Write the private key from the secret
echo "$SSH_PRIVATE_KEY" > ~/.ssh/id_ed25519

# Restrict permissions
chmod 600 ~/.ssh/id_ed25519

# Add GitHub to known_hosts (avoids host prompt)
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Start ssh-agent and add the key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
