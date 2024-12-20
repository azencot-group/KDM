import neptune

global run
run = neptune.init_run(
    project="azencot-group/func-diff",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDRkNWRmNS1mMmYyLTQ0MTctODhjZC0yNjgwM2M1MDM3YTUifQ==",
)  # your credentials