#!/bin/bash

# Node.js Installation Script
# This script automates the installation of Node.js on Debian-based systems

# Print colored output
print_message() {
  echo -e "\e[1;34m$1\e[0m"
}

print_success() {
  echo -e "\e[1;32m$1\e[0m"
}

print_error() {
  echo -e "\e[1;31m$1\e[0m"
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  print_error "Please run this script with sudo or as root."
  exit 1
fi

# Step 0: Check if curl is installed
print_message "Checking if curl is installed..."
if ! command -v curl &> /dev/null; then
  print_message "curl not found. Installing curl..."
  apt-get update
  apt-get install -y curl
  
  if [ $? -ne 0 ]; then
    print_error "Failed to install curl. Exiting."
    exit 1
  else
    print_success "curl installed successfully."
  fi
else
  print_success "curl is already installed."
fi

# Step 1: Download the NodeSource setup script
print_message "Downloading NodeSource setup script..."
curl -fsSL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh

if [ $? -ne 0 ]; then
  print_error "Failed to download NodeSource setup script. Exiting."
  exit 1
else
  print_success "NodeSource setup script downloaded successfully."
fi

# Step 2: Run the setup script
print_message "Running NodeSource setup script..."
bash nodesource_setup.sh

if [ $? -ne 0 ]; then
  print_error "Failed to run NodeSource setup script. Exiting."
  exit 1
else
  print_success "NodeSource setup script executed successfully."
fi

# Step 3: Install Node.js
print_message "Installing Node.js..."
apt-get install -y nodejs

if [ $? -ne 0 ]; then
  print_error "Failed to install Node.js. Exiting."
  exit 1
else
  print_success "Node.js installed successfully."
fi

# Step 4: Verify the installation
print_message "Verifying Node.js installation..."
NODE_VERSION=$(node -v)

if [ $? -ne 0 ]; then
  print_error "Node.js installation verification failed. Please check manually."
  exit 1
else
  print_success "Node.js $NODE_VERSION has been successfully installed."
fi

# Cleanup
print_message "Cleaning up..."
rm nodesource_setup.sh

print_success "Installation complete. You can now use Node.js on your system."