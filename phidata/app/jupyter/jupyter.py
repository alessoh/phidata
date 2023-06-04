from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from phidata.app.aws_app import AwsApp, AwsAppArgs
from phidata.app.base_app import WorkspaceVolumeType, AppVolumeType
from phidata.app.docker_app import DockerApp, DockerAppArgs
from phidata.app.k8s_app import (
    K8sApp,
    K8sAppArgs,
    ImagePullPolicy,
    RestartPolicy,
    ServiceType,
)
from phidata.utils.log import logger


class JupyterArgs(AwsAppArgs, DockerAppArgs, K8sAppArgs):
    # -*- Jupyter Configuration
    # Absolute path to JUPYTER_CONFIG_FILE,
    # Also used to set the JUPYTER_CONFIG_FILE env var,
    # This value is appended to the command using `--config`,
    jupyter_config_file: Optional[str] = None
    # Absolute path to the notebook directory,
    # Defaults to the workspace_root if mount_workspace = True else "/",
    notebook_dir: Optional[str] = None


class Jupyter(AwsApp, DockerApp, K8sApp):
    def __init__(
        self,
        name: str = "jupyter",
        version: str = "1",
        enabled: bool = True,
        # -*- Jupyter Configuration,
        # Absolute path to JUPYTER_CONFIG_FILE,
        # Also used to set the JUPYTER_CONFIG_FILE env var,
        jupyter_config_file: str = "/resources/jupyter_lab_config.py",
        # Absolute path to the notebook directory,
        # Defaults to the workspace_root if mount_workspace = True else "/mnt",
        notebook_dir: Optional[str] = None,
        # -*- Image Configuration,
        # Image can be provided as a DockerImage object or as image_name:image_tag
        image: Optional[Any] = None,
        image_name: str = "phidata/jupyter",
        image_tag: str = "3.6.3",
        entrypoint: Optional[Union[str, List[str]]] = None,
        command: Union[str, List[str]] = "jupyter lab",
        # -*- Debug Mode
        debug_mode: bool = False,
        # -*- Python Configuration,
        # Install python dependencies using a requirements.txt file,
        install_requirements: bool = False,
        # Path to the requirements.txt file relative to the workspace_root,
        requirements_file: str = "requirements.txt",
        # Set the PYTHONPATH env var,
        set_python_path: bool = False,
        # Manually provide the PYTHONPATH,
        python_path: Optional[str] = None,
        # Add paths to the PYTHONPATH env var,
        # If python_path is provided, this value is ignored,
        add_python_paths: Optional[List[str]] = None,
        # -*- Container Environment,
        # Add env variables to container,
        env: Optional[Dict[str, Any]] = None,
        # Read env variables from a file in yaml format,
        env_file: Optional[Path] = None,
        # Add secret variables to container,
        secrets: Optional[Dict[str, Any]] = None,
        # Read secret variables from a file in yaml format,
        secrets_file: Optional[Path] = None,
        # Read secret variables from AWS Secrets,
        aws_secrets: Optional[Any] = None,
        # -*- Container Ports,
        # Open a container port if open_container_port=True,
        open_container_port: bool = True,
        # Port number on the container,
        container_port: int = 8888,
        # Port name (only used by the K8sContainer),
        container_port_name: str = "http",
        # Host port to map to the container port,
        container_host_port: int = 8888,
        # -*- Workspace Volume,
        # Mount the workspace directory on the container,
        mount_workspace: bool = False,
        workspace_volume_name: Optional[str] = None,
        workspace_volume_type: Optional[WorkspaceVolumeType] = None,
        # Path to mount the workspace volume inside the container,
        workspace_dir_container_path: str = "/mnt/workspaces",
        # Add the workspace name to the container path,
        add_workspace_name_to_container_path: bool = True,
        # -*- If workspace_volume_type=WorkspaceVolumeType.HostPath,
        # Mount workspace_dir to workspace_dir_container_path,
        # If None, use the workspace_root,
        workspace_dir: Optional[str] = None,
        # -*- Resources Volume,
        # Mount a resources directory on the container,
        mount_resources: bool = False,
        # Resources directory relative to the workspace_root,
        resources_dir: str = "workspace/jupyter/resources",
        # Path to mount the resources_dir,
        resources_dir_container_path: str = "/mnt/resources",
        # -*- App Volume,
        # Create a volume for mounting app data like notebooks, models, etc.,
        create_app_volume: bool = True,
        app_volume_name: Optional[str] = None,
        app_volume_type: AppVolumeType = AppVolumeType.EmptyDir,
        # Path to mount the app volume inside the container,
        app_volume_container_path: str = "/mnt/app",
        # -*- If volume_type=AppVolumeType.HostPath,
        app_volume_host_path: Optional[str] = None,
        # -*- If volume_type=AppVolumeType.AwsEbs,
        # EbsVolume: used to derive the volume_id, region, and az,
        app_ebs_volume: Optional[Any] = None,
        app_ebs_volume_region: Optional[str] = None,
        app_ebs_volume_az: Optional[str] = None,
        # Provide Ebs Volume-id manually,
        app_ebs_volume_id: Optional[str] = None,
        # -*- If volume_type=AppVolumeType.PersistentVolume,
        # AccessModes is a list of ways the volume can be mounted.,
        # More info: https://kubernetes.io/docs/concepts/storage/persistent-volumes#access-modes,
        # Type: phidata.infra.k8s.enums.pv.PVAccessMode,
        app_pv_access_modes: Optional[List[Any]] = None,
        app_pv_requests_storage: Optional[str] = None,
        # A list of mount options, e.g. ["ro", "soft"]. Not validated - mount will simply fail if one is invalid.,
        # More info: https://kubernetes.io/docs/concepts/storage/persistent-volumes/#mount-options,
        app_pv_mount_options: Optional[List[str]] = None,
        # What happens to a persistent volume when released from its claim.,
        #   The default policy is Retain.,
        # Literal["Delete", "Recycle", "Retain"],
        app_pv_reclaim_policy: Optional[str] = None,
        app_pv_storage_class: str = "",
        app_pv_labels: Optional[Dict[str, str]] = None,
        # -*- If volume_type=AppVolumeType.AwsEfs,
        app_efs_volume_id: Optional[str] = None,
        # Add NodeSelectors to Pods, so they are scheduled in the same region and zone as the ebs_volume,
        schedule_pods_in_ebs_topology: bool = True,
        # -*- Container Configuration,
        container_name: Optional[str] = None,
        # Run container in the background and return a Container object.,
        container_detach: bool = True,
        # Enable auto-removal of the container on daemon side when the container’s process exits.,
        container_auto_remove: bool = True,
        # Remove the container when it has finished running. Default: True.,
        container_remove: bool = True,
        # Username or UID to run commands as inside the container.,
        container_user: Optional[Union[str, int]] = None,
        # Keep STDIN open even if not attached.,
        container_stdin_open: bool = True,
        # Return logs from STDOUT when container_detach=False.,
        container_stdout: Optional[bool] = True,
        # Return logs from STDERR when container_detach=False.,
        container_stderr: Optional[bool] = True,
        container_tty: bool = True,
        # Specify a test to perform to check that the container is healthy.,
        container_healthcheck: Optional[Dict[str, Any]] = None,
        # Optional hostname for the container.,
        container_hostname: Optional[str] = None,
        # Platform in the format os[/arch[/variant]].,
        container_platform: Optional[str] = None,
        # Path to the working directory.,
        container_working_dir: Optional[str] = None,
        # Add labels to the container,
        container_labels: Optional[Dict[str, str]] = None,
        # Restart the container when it exits. Configured as a dictionary with keys:,
        # Name: One of on-failure, or always.,
        # MaximumRetryCount: Number of times to restart the container on failure.,
        # For example: {"Name": "on-failure", "MaximumRetryCount": 5},
        container_restart_policy: Optional[Dict[str, Any]] = None,
        # Add volumes to DockerContainer,
        # container_volumes is a dictionary which adds the volumes to mount,
        # inside the container. The key is either the host path or a volume name,,
        # and the value is a dictionary with 2 keys:,
        #   bind - The path to mount the volume inside the container,
        #   mode - Either rw to mount the volume read/write, or ro to mount it read-only.,
        # For example:,
        # {,
        #   '/home/user1/': {'bind': '/mnt/vol2', 'mode': 'rw'},,
        #   '/var/www': {'bind': '/mnt/vol1', 'mode': 'ro'},
        # },
        container_volumes: Optional[Dict[str, dict]] = None,
        # Add ports to DockerContainer,
        # The keys of the dictionary are the ports to bind inside the container,,
        # either as an integer or a string in the form port/protocol, where the protocol is either tcp, udp.,
        # The values of the dictionary are the corresponding ports to open on the host, which can be either:,
        #   - The port number, as an integer.,
        #       For example, {'2222/tcp': 3333} will expose port 2222 inside the container as port 3333 on the host.,
        #   - None, to assign a random host port. For example, {'2222/tcp': None}.,
        #   - A tuple of (address, port) if you want to specify the host interface.,
        #       For example, {'1111/tcp': ('127.0.0.1', 1111)}.,
        #   - A list of integers, if you want to bind multiple host ports to a single container port.,
        #       For example, {'1111/tcp': [1234, 4567]}.,
        container_ports: Optional[Dict[str, Any]] = None,
        # -*- Pod Configuration,
        pod_name: Optional[str] = None,
        pod_annotations: Optional[Dict[str, str]] = None,
        pod_node_selector: Optional[Dict[str, str]] = None,
        # -*- Secret Configuration,
        secret_name: Optional[str] = None,
        # -*- Configmap Configuration,
        configmap_name: Optional[str] = None,
        # -*- Deployment Configuration,
        replicas: int = 1,
        deploy_name: Optional[str] = None,
        # Type: ImagePullPolicy,
        image_pull_policy: Optional[Any] = None,
        # Type: RestartPolicy,
        deploy_restart_policy: Optional[Any] = None,
        deploy_labels: Optional[Dict[str, Any]] = None,
        termination_grace_period_seconds: Optional[int] = None,
        # Key to spread the pods across a topology,
        topology_spread_key: Optional[str] = None,
        # The degree to which pods may be unevenly distributed,
        topology_spread_max_skew: Optional[int] = None,
        # How to deal with a pod if it doesn't satisfy the spread constraint.,
        topology_spread_when_unsatisfiable: Optional[str] = None,
        # -*- Service Configuration,
        create_service: bool = False,
        service_name: Optional[str] = None,
        # Type: ServiceType,
        service_type: Optional[Any] = None,
        # The port exposed by the service.,
        service_port: int = 8000,
        # The node_port exposed by the service if service_type = ServiceType.NODE_PORT,
        service_node_port: Optional[int] = None,
        # The target_port is the port to access on the pods targeted by the service.,
        # It can be the port number or port name on the pod.,
        service_target_port: Optional[Union[str, int]] = None,
        # Extra ports exposed by the webserver service. Type: List[CreatePort],
        service_ports: Optional[List[Any]] = None,
        service_labels: Optional[Dict[str, Any]] = None,
        service_annotations: Optional[Dict[str, str]] = None,
        # If ServiceType == ServiceType.LoadBalancer,
        service_health_check_node_port: Optional[int] = None,
        service_internal_traffic_policy: Optional[str] = None,
        service_load_balancer_class: Optional[str] = None,
        service_load_balancer_ip: Optional[str] = None,
        service_load_balancer_source_ranges: Optional[List[str]] = None,
        service_allocate_load_balancer_node_ports: Optional[bool] = None,
        # -*- Ingress Configuration,
        create_ingress: bool = False,
        ingress_name: Optional[str] = None,
        ingress_annotations: Optional[Dict[str, str]] = None,
        # -*- RBAC Configuration,
        use_rbac: bool = False,
        # Create a Namespace with name ns_name & default values,
        ns_name: Optional[str] = None,
        # or Provide the full Namespace definition,
        # Type: CreateNamespace,
        namespace: Optional[Any] = None,
        # Create a ServiceAccount with name sa_name & default values,
        sa_name: Optional[str] = None,
        # or Provide the full ServiceAccount definition,
        # Type: CreateServiceAccount,
        service_account: Optional[Any] = None,
        # Create a ClusterRole with name cr_name & default values,
        cr_name: Optional[str] = None,
        # or Provide the full ClusterRole definition,
        # Type: CreateClusterRole,
        cluster_role: Optional[Any] = None,
        # Create a ClusterRoleBinding with name crb_name & default values,
        crb_name: Optional[str] = None,
        # or Provide the full ClusterRoleBinding definition,
        # Type: CreateClusterRoleBinding,
        cluster_role_binding: Optional[Any] = None,
        # -*- AWS Configuration,
        aws_subnets: Optional[List[str]] = None,
        aws_security_groups: Optional[List[str]] = None,
        # -*- ECS Configuration,
        ecs_cluster: Optional[Any] = None,
        ecs_launch_type: str = "FARGATE",
        ecs_task_cpu: str = "512",
        ecs_task_memory: str = "1024",
        ecs_service_count: int = 1,
        assign_public_ip: bool = True,
        ecs_enable_exec: bool = True,
        # -*- LoadBalancer Configuration,
        enable_load_balancer: bool = True,
        load_balancer: Optional[Any] = None,
        # HTTP or HTTPS,
        load_balancer_protocol: str = "HTTP",
        # Default 80 for HTTP and 443 for HTTPS,
        load_balancer_port: Optional[int] = None,
        load_balancer_certificate_arn: Optional[str] = None,
        #  -*- Resource Control,
        skip_create: bool = False,
        skip_read: bool = False,
        skip_update: bool = False,
        recreate_on_update: bool = False,
        skip_delete: bool = False,
        wait_for_creation: bool = True,
        wait_for_update: bool = True,
        wait_for_deletion: bool = True,
        waiter_delay: int = 30,
        waiter_max_attempts: int = 50,
        # Skip creation if resource with the same name is active,
        use_cache: bool = True,
        #  -*- Other args,
        print_env_on_load: bool = False,
        # Extra kwargs used to capture additional args,
        **extra_kwargs,
    ):
        super().__init__()

        if jupyter_config_file is not None:
            self.container_env = {"JUPYTER_CONFIG_FILE": jupyter_config_file}

        try:
            self.args: JupyterArgs = JupyterArgs(
                name=name,
                version=version,
                enabled=enabled,
                jupyter_config_file=jupyter_config_file,
                notebook_dir=notebook_dir,
                image=image,
                image_name=image_name,
                image_tag=image_tag,
                entrypoint=entrypoint,
                command=command,
                debug_mode=debug_mode,
                install_requirements=install_requirements,
                requirements_file=requirements_file,
                set_python_path=set_python_path,
                python_path=python_path,
                add_python_paths=add_python_paths,
                env=env,
                env_file=env_file,
                secrets=secrets,
                secrets_file=secrets_file,
                aws_secrets=aws_secrets,
                open_container_port=open_container_port,
                container_port=container_port,
                container_port_name=container_port_name,
                container_host_port=container_host_port,
                mount_workspace=mount_workspace,
                workspace_volume_name=workspace_volume_name,
                workspace_volume_type=workspace_volume_type,
                workspace_dir_container_path=workspace_dir_container_path,
                add_workspace_name_to_container_path=add_workspace_name_to_container_path,
                workspace_dir=workspace_dir,
                mount_resources=mount_resources,
                resources_dir=resources_dir,
                resources_dir_container_path=resources_dir_container_path,
                create_app_volume=create_app_volume,
                app_volume_name=app_volume_name,
                app_volume_type=app_volume_type,
                app_volume_container_path=app_volume_container_path,
                app_volume_host_path=app_volume_host_path,
                app_ebs_volume=app_ebs_volume,
                app_ebs_volume_region=app_ebs_volume_region,
                app_ebs_volume_az=app_ebs_volume_az,
                app_ebs_volume_id=app_ebs_volume_id,
                app_pv_access_modes=app_pv_access_modes,
                app_pv_requests_storage=app_pv_requests_storage,
                app_pv_mount_options=app_pv_mount_options,
                app_pv_reclaim_policy=app_pv_reclaim_policy,
                app_pv_storage_class=app_pv_storage_class,
                app_pv_labels=app_pv_labels,
                app_efs_volume_id=app_efs_volume_id,
                schedule_pods_in_ebs_topology=schedule_pods_in_ebs_topology,
                container_name=container_name,
                container_detach=container_detach,
                container_auto_remove=container_auto_remove,
                container_remove=container_remove,
                container_user=container_user,
                container_stdin_open=container_stdin_open,
                container_stdout=container_stdout,
                container_stderr=container_stderr,
                container_tty=container_tty,
                container_healthcheck=container_healthcheck,
                container_hostname=container_hostname,
                container_platform=container_platform,
                container_working_dir=container_working_dir,
                container_labels=container_labels,
                container_restart_policy=container_restart_policy,
                container_volumes=container_volumes,
                container_ports=container_ports,
                pod_name=pod_name,
                pod_annotations=pod_annotations,
                pod_node_selector=pod_node_selector,
                secret_name=secret_name,
                configmap_name=configmap_name,
                replicas=replicas,
                deploy_name=deploy_name,
                image_pull_policy=image_pull_policy,
                deploy_restart_policy=deploy_restart_policy,
                deploy_labels=deploy_labels,
                termination_grace_period_seconds=termination_grace_period_seconds,
                topology_spread_key=topology_spread_key,
                topology_spread_max_skew=topology_spread_max_skew,
                topology_spread_when_unsatisfiable=topology_spread_when_unsatisfiable,
                create_service=create_service,
                service_name=service_name,
                service_type=service_type,
                service_port=service_port,
                service_node_port=service_node_port,
                service_target_port=service_target_port,
                service_ports=service_ports,
                service_labels=service_labels,
                service_annotations=service_annotations,
                service_health_check_node_port=service_health_check_node_port,
                service_internal_traffic_policy=service_internal_traffic_policy,
                service_load_balancer_class=service_load_balancer_class,
                service_load_balancer_ip=service_load_balancer_ip,
                service_load_balancer_source_ranges=service_load_balancer_source_ranges,
                service_allocate_load_balancer_node_ports=service_allocate_load_balancer_node_ports,
                create_ingress=create_ingress,
                ingress_name=ingress_name,
                ingress_annotations=ingress_annotations,
                use_rbac=use_rbac,
                ns_name=ns_name,
                namespace=namespace,
                sa_name=sa_name,
                service_account=service_account,
                cr_name=cr_name,
                cluster_role=cluster_role,
                crb_name=crb_name,
                cluster_role_binding=cluster_role_binding,
                aws_subnets=aws_subnets,
                aws_security_groups=aws_security_groups,
                ecs_cluster=ecs_cluster,
                ecs_launch_type=ecs_launch_type,
                ecs_task_cpu=ecs_task_cpu,
                ecs_task_memory=ecs_task_memory,
                ecs_service_count=ecs_service_count,
                assign_public_ip=assign_public_ip,
                ecs_enable_exec=ecs_enable_exec,
                enable_load_balancer=enable_load_balancer,
                load_balancer=load_balancer,
                load_balancer_protocol=load_balancer_protocol,
                load_balancer_port=load_balancer_port,
                load_balancer_certificate_arn=load_balancer_certificate_arn,
                skip_create=skip_create,
                skip_read=skip_read,
                skip_update=skip_update,
                recreate_on_update=recreate_on_update,
                skip_delete=skip_delete,
                wait_for_creation=wait_for_creation,
                wait_for_update=wait_for_update,
                wait_for_deletion=wait_for_deletion,
                waiter_delay=waiter_delay,
                waiter_max_attempts=waiter_max_attempts,
                use_cache=use_cache,
                print_env_on_load=print_env_on_load,
                extra_kwargs=extra_kwargs,
            )
        except Exception as e:
            logger.error(f"Args for {self.name} are not valid: {e}")
            raise

    def get_container_command_docker(self) -> Optional[List[str]]:
        container_cmd: List[str]
        if isinstance(self.args.command, str):
            container_cmd = self.args.command.split(" ")
        elif isinstance(self.args.command, list):
            container_cmd = self.args.command
        else:
            container_cmd = ["jupyter", "lab"]

        if self.args.jupyter_config_file is not None:
            container_cmd.append(f"--config={str(self.args.jupyter_config_file)}")

        if self.args.notebook_dir is None:
            if self.args.mount_workspace:
                container_paths = self.get_container_paths()
                if (
                    container_paths is not None
                    and container_paths.workspace_root is not None
                ):
                    container_cmd.append(
                        f"--notebook-dir={str(container_paths.workspace_root)}"
                    )
            else:
                container_cmd.append("--notebook-dir=/")
        else:
            container_cmd.append(f"--notebook-dir={str(self.args.notebook_dir)}")
        return container_cmd

    def get_container_args_k8s(self) -> Optional[List[str]]:
        container_args: List[str]
        if isinstance(self.args.command, str):
            container_args = self.args.command.split(" ")
        elif isinstance(self.args.command, list):
            container_args = self.args.command
        else:
            container_args = ["jupyter", "lab"]

        if self.args.jupyter_config_file is not None:
            container_args.append(f"--config={str(self.args.jupyter_config_file)}")

        if self.args.notebook_dir is None:
            if self.args.mount_workspace:
                if (
                    self.container_paths is not None
                    and self.container_paths.workspace_root is not None
                ):
                    container_args.append(
                        f"--notebook-dir={str(self.container_paths.workspace_root)}"
                    )
            else:
                container_args.append("--notebook-dir=/")
        else:
            container_args.append(f"--notebook-dir={str(self.args.notebook_dir)}")
        return container_args
