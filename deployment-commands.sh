# Make the Python service publicly accessible or grant the Rust service account access
gcloud run services set-iam-policy python-analysis-server-179718527697 policy.yaml

# Example policy.yaml content:
# bindings:
# - members:
#   - allUsers  # For public access
#   role: roles/run.invoker