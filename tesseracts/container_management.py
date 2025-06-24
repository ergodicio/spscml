import boto3
import uuid


def create_or_update_tesseract_service(tesseract_name, needs_gpu=True):
    # Create a Boto3 session
    session = boto3.Session()

    # Create an ECS client
    ecs_client = session.client("ecs")

    # get cluster
    for _cluster_arn in ecs_client.list_clusters()["clusterArns"]:
        if "continuum-cluster" in _cluster_arn:
            cluster = _cluster_arn

    # check if service exists
    service_found = False
    services = ecs_client.list_services(cluster=cluster, maxResults=100)
    for service in services["serviceArns"]:
        if tesseract_name in service:
            this_service = service
            service_found = True

    log_group_name = f"/hackathon/{tesseract_name}"

    new_container_def = {
        "name": tesseract_name,
        "image": f"106231741818.dkr.ecr.us-east-1.amazonaws.com/hackathon:{tesseract_name}",
        "portMappings": [{"containerPort": 8000, "hostPort": 8000}],
        "environment": [
            {"name": "MLFLOW_TRACKING_URI", "value": "http://hackathon_mlfs.continuum"},
            {"name": "RANDOM_VAR", "value": str(uuid.uuid4())},
        ],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group_name,
                "awslogs-region": "us-east-1",
                "awslogs-stream-prefix": "ecs",
            },
        },
        "resourceRequirements": [{"type": "GPU", "value": "1"}] if needs_gpu else [],
        "entryPoint": ["tesseract-runtime"],
        "command": ["serve", "--host", "0.0.0.0"],
    }

    family_name = tesseract_name + "-family"
    new_task_definition = register_task_definition(
        ecs_client, new_container_def, family_name
    )

    if service_found:
        print("Updating existing service: {}".format(this_service))
        ecs_client.update_service(
            cluster=cluster,
            service=this_service,
            taskDefinition=new_task_definition["taskDefinition"]["taskDefinitionArn"],
            desiredCount=1,
        )

    else:
        print("Creating new service for tesseract: {}".format(tesseract_name))
        sd = boto3.client("servicediscovery")
        logs = session.client("logs")
        logs.create_log_group(logGroupName=log_group_name)

        response = sd.create_service(
            Name=tesseract_name,
            NamespaceId="ns-w5wmjwymvhbhoeq3",
            DnsConfig={
                "NamespaceId": "ns-w5wmjwymvhbhoeq3",
                "DnsRecords": [{"Type": "A", "TTL": 60}],
            },
            HealthCheckCustomConfig={"FailureThreshold": 1},
        )

        ecs_client.create_service(
            cluster=cluster,
            serviceName=tesseract_name,
            taskDefinition=new_task_definition["taskDefinition"]["taskDefinitionArn"],
            desiredCount=1,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": [
                        "subnet-0fa0f789e06ae28f9",
                        "subnet-00baf77a953fe5698",
                    ],
                    "securityGroups": ["sg-0f35111be1df8c52e"],
                    "assignPublicIp": "DISABLED",
                }
            },
            capacityProviderStrategy=[
                {"capacityProvider": "MyAsgCapacityProvider", "weight": 1, "base": 1}
            ],
            serviceRegistries=[
                {
                    "registryArn": response["Service"]["Arn"],
                    "containerName": tesseract_name,
                }
            ],
        )

    return "Service management completed for tesseract: {}".format(tesseract_name)


def register_task_definition(ecs_client, new_container_def, family_name):
    new_task_definition = ecs_client.register_task_definition(
        family=family_name,
        cpu="2040",
        memory="7660",
        containerDefinitions=[new_container_def],
        networkMode="awsvpc",
        requiresCompatibilities=["EC2"],
        taskRoleArn="arn:aws:iam::106231741818:role/continuum-hackathonmlfstaskdefTaskRole6B9D66CE-sMELLebLsQrC",
        # placementConstraints=[{"type": "distinctInstance", "expression": ""}],
    )

    return new_task_definition


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage ECS containers for the hackathon."
    )
    parser.add_argument(
        "--tesseract",
        type=str,
        required=True,
        help="The name of the tesseract to update",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Deploy without GPU resources",
    )

    args = parser.parse_args()
    tesseract_name = args.tesseract
    needs_gpu = not args.no_gpu

    print(create_or_update_tesseract_service(tesseract_name, needs_gpu))
