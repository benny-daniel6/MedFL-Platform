Step 1: Provision the Virtual Machine (VM)
First, create a small, free-tier virtual machine that will act as our server.

Bash

# Run these commands on your local machine

# 1. Set your GCP project ID

gcloud config set project YOUR_PROJECT_ID

# 2. Create the e2-micro VM instance (part of GCP's "Always Free" tier)

gcloud compute instances create fl-aggregator-server \
 --machine-type=e2-micro \
 --image-family=debian-11 \
 --image-project=debian-cloud \
 --zone=us-central1-a \
 --tags=flower-server
Step 2: Configure the Firewall
Open port 8080 on the VM to allow Flower clients to connect to the server.

Bash

# Run this on your local machine

gcloud compute firewall-rules create allow-flower-tcp-8080 \
 --network=default \
 --allow=tcp:8080 \
 --source-ranges=0.0.0.0/0 \
 --target-tags=flower-server
Step 3: Set Up the Server Environment
Now, connect to your new VM and set up the project environment.

Bash

# 1. SSH into your new VM

gcloud compute ssh fl-aggregator-server --zone=us-central1-a

# --- The following commands are run inside the VM ---

# 2. Update package lists and install required tools

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# 3. Clone your project from GitHub

git clone https://github.com/benny-daniel6/MedFL-Platform.git
cd YOUR_REPO_NAME

# 4. Create and activate a Python virtual environment

python3 -m venv venv
source venv/bin/activate

# 5. Install your project's dependencies

pip install -r requirements.txt
Step 4: Create a systemd Service
To ensure the server runs continuously and restarts automatically, we'll create a systemd service. This is the professional way to manage background processes on Linux.

Create a service file using the nano text editor:

Bash

sudo nano /etc/systemd/system/flower.service
Paste the following configuration into the text editor. You must replace YOUR_USERNAME and YOUR_REPO_NAME with your actual details.

Ini, TOML

[Unit]
Description=Flower Federated Learning Server
After=network.target

[Service]

# Replace 'your_vm_username' with the username you see in the VM's terminal prompt

User=your_vm_username
Group=your_vm_username

# Replace with the actual path to your project

WorkingDirectory=/home/your_vm_username/YOUR_REPO_NAME

# Command to start the server, using the Python from your virtual environment

ExecStart=/home/your_vm_username/YOUR_REPO_NAME/venv/bin/python run_server.py --rounds 20

Restart=always

[Install]
WantedBy=multi-user.target
Save and exit nano: Press Ctrl+X, then Y, then Enter.

Step 5: Start and Enable the Server
Finally, start your new service and enable it to launch on boot.

Bash

# --- Run these commands inside the VM ---

# 1. Reload the systemd manager to recognize the new service

sudo systemctl daemon-reload

# 2. Start the Flower service

sudo systemctl start flower.service

# 3. Check the status to ensure it's running without errors

sudo systemctl status flower.service

# (Press 'q' to exit the status view)

# 4. Enable the service to start automatically when the VM reboots

sudo systemctl enable flower.service
Your Flower server is now running permanently on the GCP virtual machine. You can find your VM's public IP address in the GCP Console and use it to connect your clients from anywhere.
