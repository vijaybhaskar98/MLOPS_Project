# Infrastructure Setup

Guide to using GCP service

## Prerequisite
1. Create a new Gmail account for $300 free credit on GCP
2. Create an account on GCP and setup a billing
---
## Step 1: Login into the GCP console and create a project

Name: `mlops-demo`
![Project Creation](/images/01.png)


## Step 2: Create an instance using Compute Engine

Configuration:
* Name: `demo-vm`
* Region: `us-east1`
* Zone: `us-east-b`
* Series: `N2`
* Machine Type: `n2-standard-2`
* Boot disk: 
    * Image: `Ubuntu 22.04 LTS`
* Firewall: Check both
    * Allow HTTP traffic
    * Allow HTTPS traffic

![Project Creation](/images/02.png)

![Project Creation](/images/03.png)
