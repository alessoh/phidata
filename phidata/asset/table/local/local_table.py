from pathlib import Path
from typing import Optional, Any, Union, List, Dict
from typing_extensions import Literal

from phidata.asset.local import LocalAsset, LocalAssetArgs
from phidata.check.df.dataframe_check import DataFrameCheck
from phidata.utils.enums import ExtendedEnum
from phidata.utils.log import logger


class LocalTableFormat(ExtendedEnum):
    CSV = "csv"
    IPC = "ipc"
    ARROW = "arrow"
    FEATHER = "feather"
    ORC = "orc"
    PARQUET = "parquet"


class LocalTableArgs(LocalAssetArgs):
    # Table Name
    name: str
    # Database for the table
    database: str = "default"
    # Table Format
    table_format: LocalTableFormat

    # Checks to run before reading from disk
    read_checks: Optional[List[DataFrameCheck]] = None
    # Checks to run before writing to disk
    write_checks: Optional[List[DataFrameCheck]] = None

    # -*- Table Path
    # If current_dir=True, store the table in the current directory
    current_dir: bool = False
    # Top level directory for all tables, under the "storage" directory
    top_level_dir: Optional[str] = "tables"
    # Provide absolute path to the table directory
    table_dir: Optional[str] = None
    # A template string used to generate basenames of written data files.
    # The token ‘{i}’ will be replaced with an automatically incremented integer.
    # If not specified, it defaults to “part-{i}.” + format.default_extname
    basename_template: Optional[str] = None

    # List of partition columns
    partitions: Optional[List[str]] = None

    # Maximum number of partitions any batch may be written into.
    max_partitions: Optional[int] = None
    # If greater than 0 then this will limit the maximum number of files that can be left open.
    # If an attempt is made to open too many files then the least recently used file will be closed.
    # If this setting is set too low you may end up fragmenting your data into many small files.
    max_open_files: Optional[int] = None
    # Maximum number of rows per file. If greater than 0 then this will limit how many rows are placed in any single
    # file. Otherwise there will be no limit and one file will be created in each output directory unless files need
    # to be closed to respect max_open_files
    max_rows_per_file: Optional[int] = None
    # Minimum number of rows per group. When the value is greater than 0, the dataset writer will batch incoming data
    # and only write the row groups to the disk when sufficient rows have accumulated.
    min_rows_per_group: Optional[int] = None
    # Maximum number of rows per group. If the value is greater than 0, then the dataset writer may split up large
    # incoming batches into multiple row groups. If this value is set, then min_rows_per_group should also be set.
    # Otherwise it could end up with very small row groups.
    max_rows_per_group: Optional[int] = None
    # Controls how the dataset will handle data that already exists in the destination.
    # The default behavior (‘error’) is to raise an error if any data exists in the destination.
    # ‘overwrite_or_ignore’ will ignore any existing data and will overwrite files with the same name
    # as an output file. Other existing files will be ignored. This behavior, in combination with a unique
    # basename_template for each write, will allow for an append workflow.
    # ‘delete_matching’ is useful when you are writing a partitioned dataset.
    # The first time each partition directory is encountered the entire directory will be deleted.
    # This allows you to overwrite old partitions completely.
    write_mode: Literal[
        "delete_matching", "overwrite_or_ignore", "error"
    ] = "delete_matching"


class LocalTable(LocalAsset):
    """Base Class for Local tables"""

    def __init__(
        self,
        # Table Name: required
        name: str,
        # S3 Table Format: required
        table_format: LocalTableFormat,
        # Database for the table
        database: str = "default",
        # DataModel for this table
        data_model: Optional[Any] = None,
        # Checks to run before reading from disk
        read_checks: Optional[List[DataFrameCheck]] = None,
        # Checks to run before writing to disk
        write_checks: Optional[List[DataFrameCheck]] = None,
        # -*- Table Path
        # If current_dir=True, store the table in the current directory
        current_dir: bool = False,
        # Top level directory for all tables, under the "storage" directory
        top_level_dir: Optional[str] = "tables",
        # Provide absolute path to the table directory
        table_dir: Optional[str] = None,
        # A template string used to generate basenames of written data files.
        # The token ‘{i}’ will be replaced with an automatically incremented integer.
        # If not specified, it defaults to “part-{i}.” + format.default_extname
        basename_template: Optional[str] = None,
        # List of partition columns
        partitions: Optional[List[str]] = None,
        # Maximum number of partitions any batch may be written into.
        max_partitions: Optional[int] = None,
        # If greater than 0 then this will limit the maximum number of files that can be left open.
        # If an attempt is made to open too many files then the least recently used file will be closed.
        # If this setting is set too low you may end up fragmenting your data into many small files.
        max_open_files: Optional[int] = None,
        # Maximum number of rows per file. If greater than 0 then this will limit how many rows are placed in any single
        # file. Otherwise there will be no limit and one file will be created in each output directory unless files need
        # to be closed to respect max_open_files
        max_rows_per_file: Optional[int] = None,
        # Minimum number of rows per group. When the value is greater than 0, the dataset writer will batch incoming data
        # and only write the row groups to the disk when sufficient rows have accumulated.
        min_rows_per_group: Optional[int] = None,
        # Maximum number of rows per group. If the value is greater than 0, then the dataset writer may split up large
        # incoming batches into multiple row groups. If this value is set, then min_rows_per_group should also be set.
        # Otherwise it could end up with very small row groups.
        max_rows_per_group: Optional[int] = None,
        # Controls how the dataset will handle data that already exists in the destination.
        # The default behavior (‘error’) is to raise an error if any data exists in the destination.
        # ‘overwrite_or_ignore’ will ignore any existing data and will overwrite files with the same name
        # as an output file. Other existing files will be ignored. This behavior, in combination with a unique
        # basename_template for each write, will allow for an append workflow.
        # ‘delete_matching’ is useful when you are writing a partitioned dataset.
        # The first time each partition directory is encountered the entire directory will be deleted.
        # This allows you to overwrite old partitions completely.
        write_mode: Literal[
            "delete_matching", "overwrite_or_ignore", "error"
        ] = "delete_matching",
        version: Optional[str] = None,
        enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.args: Optional[LocalTableArgs] = None
        if name is None:
            raise ValueError("name is required")
        if table_format is None:
            raise ValueError("table_format is required")

        try:
            self.args = LocalTableArgs(
                name=name,
                table_format=table_format,
                database=database,
                data_model=data_model,
                read_checks=read_checks,
                write_checks=write_checks,
                current_dir=current_dir,
                top_level_dir=top_level_dir,
                table_dir=table_dir,
                basename_template=basename_template,
                partitions=partitions,
                max_partitions=max_partitions,
                max_open_files=max_open_files,
                max_rows_per_file=max_rows_per_file,
                min_rows_per_group=min_rows_per_group,
                max_rows_per_group=max_rows_per_group,
                write_mode=write_mode,
                version=version,
                enabled=enabled,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Args for {self.name} are not valid")
            raise

    @property
    def database(self) -> Optional[str]:
        return self.args.database if self.args else None

    @database.setter
    def database(self, database: str) -> None:
        if self.args and database:
            self.args.database = database

    @property
    def table_format(self) -> Optional[LocalTableFormat]:
        return self.args.table_format if self.args else None

    @table_format.setter
    def table_format(self, table_format: LocalTableFormat) -> None:
        if self.args and table_format:
            self.args.table_format = table_format

    @property
    def read_checks(self) -> Optional[List[DataFrameCheck]]:
        return self.args.read_checks if self.args else None

    @read_checks.setter
    def read_checks(self, read_checks: List[DataFrameCheck]) -> None:
        if self.args and read_checks:
            self.args.read_checks = read_checks

    @property
    def write_checks(self) -> Optional[List[DataFrameCheck]]:
        return self.args.write_checks if self.args else None

    @write_checks.setter
    def write_checks(self, write_checks: List[DataFrameCheck]) -> None:
        if self.args and write_checks:
            self.args.write_checks = write_checks

    @property
    def current_dir(self) -> Optional[bool]:
        return self.args.current_dir if self.args else None

    @current_dir.setter
    def current_dir(self, current_dir: bool) -> None:
        if self.args and current_dir:
            self.args.current_dir = current_dir

    @property
    def top_level_dir(self) -> Optional[str]:
        return self.args.top_level_dir if self.args else None

    @top_level_dir.setter
    def top_level_dir(self, top_level_dir: str) -> None:
        if self.args and top_level_dir:
            self.args.top_level_dir = top_level_dir

    @property
    def table_dir(self) -> Optional[str]:
        return self.args.table_dir if self.args else None

    @table_dir.setter
    def table_dir(self, table_dir: str) -> None:
        if self.args and table_dir:
            self.args.table_dir = table_dir

    @property
    def basename_template(self) -> Optional[str]:
        return self.args.basename_template if self.args else None

    @basename_template.setter
    def basename_template(self, basename_template: str) -> None:
        if self.args and basename_template:
            self.args.basename_template = basename_template

    @property
    def partitions(self) -> Optional[List[str]]:
        return self.args.partitions if self.args else None

    @partitions.setter
    def partitions(self, partitions: List[str]) -> None:
        if self.args and partitions:
            self.args.partitions = partitions

    @property
    def max_partitions(self) -> Optional[int]:
        return self.args.max_partitions if self.args else None

    @max_partitions.setter
    def max_partitions(self, max_partitions: int) -> None:
        if self.args and max_partitions:
            self.args.max_partitions = max_partitions

    @property
    def max_open_files(self) -> Optional[int]:
        return self.args.max_open_files if self.args else None

    @max_open_files.setter
    def max_open_files(self, max_open_files: int) -> None:
        if self.args and max_open_files:
            self.args.max_open_files = max_open_files

    @property
    def max_rows_per_file(self) -> Optional[int]:
        return self.args.max_rows_per_file if self.args else None

    @max_rows_per_file.setter
    def max_rows_per_file(self, max_rows_per_file: int) -> None:
        if self.args and max_rows_per_file:
            self.args.max_rows_per_file = max_rows_per_file

    @property
    def min_rows_per_group(self) -> Optional[int]:
        return self.args.min_rows_per_group if self.args else None

    @min_rows_per_group.setter
    def min_rows_per_group(self, min_rows_per_group: int) -> None:
        if self.args and min_rows_per_group:
            self.args.min_rows_per_group = min_rows_per_group

    @property
    def max_rows_per_group(self) -> Optional[int]:
        return self.args.max_rows_per_group if self.args else None

    @max_rows_per_group.setter
    def max_rows_per_group(self, max_rows_per_group: int) -> None:
        if self.args and max_rows_per_group:
            self.args.max_rows_per_group = max_rows_per_group

    @property
    def write_mode(
        self,
    ) -> Optional[Literal["delete_matching", "overwrite_or_ignore", "error"]]:
        return self.args.write_mode if self.args else None

    @write_mode.setter
    def write_mode(
        self, write_mode: Literal["delete_matching", "overwrite_or_ignore", "error"]
    ) -> None:
        if self.args and write_mode:
            self.args.write_mode = write_mode

    @property
    def table_location(self) -> str:
        if self.table_dir is not None:
            return self.table_dir

        logger.debug("-*- Building local table location")
        base_dir: Optional[Path] = None

        # Use current_dir as base path if set
        if self.current_dir:
            base_dir = Path(__file__).resolve()

        # Or use storage_dir_path as the base path
        if base_dir is None:
            # storage_dir_path is loaded from the current environment variable
            base_dir = self.storage_dir_path

        # Add the file_dir if provided
        if self.top_level_dir is not None:
            if base_dir is None:
                # base_dir is None meaning no storage_dir_path
                base_dir = Path(".").resolve().joinpath(self.top_level_dir)
            else:
                base_dir = base_dir.joinpath(self.top_level_dir)

        # Add the file_name
        if self.name is not None:
            if base_dir is None:
                base_dir = Path(".").resolve().joinpath(self.name)
            else:
                base_dir = base_dir.joinpath(self.name)

        self.table_dir = str(base_dir)
        logger.debug(f"-*- Table location: {self.table_dir}")
        return self.table_dir

    ######################################################
    ## Validate data asset
    ######################################################

    def is_valid(self) -> bool:
        return True

    ######################################################
    ## Build data asset
    ######################################################

    def build(self) -> bool:
        logger.debug(f"@build not defined for {self.name}")
        return False

    ######################################################
    ## Create DataAsset
    ######################################################

    def _create(self) -> bool:
        logger.error(f"@_create not defined for {self.name}")
        return False

    def post_create(self) -> bool:
        return True

    def write_df(self, df: Optional[Any] = None, **write_options) -> bool:
        """
        Write DataFrame to disk.
        """

        # LocalTable not yet initialized
        if self.args is None:
            return False

        # Check name is available
        if self.name is None:
            logger.error("Table name invalid")
            return False

        # Check table_location is available
        table_location = self.table_location
        if table_location is None:
            logger.error("Table location invalid")
            return False

        # Check S3FileSystem is available
        fs = self._get_fs()
        if fs is None:
            logger.error("Could not create LocalFileSystem")
            return False

        # Validate polars is installed
        try:
            import polars as pl
        except ImportError as ie:
            logger.error(f"Polars not installed: {ie}")
            return False

        # Validate pyarrow is installed
        try:
            import pyarrow as pa
            import pyarrow.dataset
        except ImportError as ie:
            logger.error(f"PyArrow not installed: {ie}")
            return False

        # Validate df
        if df is None or not isinstance(df, pl.DataFrame):
            logger.error("DataFrame invalid")
            return False

        logger.debug("Format: {}".format(self.args.table_format.value))
        try:
            # Run write checks
            if self.write_checks is not None:
                for check in self.write_checks:
                    if not check.check(df):
                        return False

            # Create arrow table
            table: pa.Table = df.to_arrow()
            if table is None:
                logger.error("Could not create Arrow table")
                return False

            # Create a dict of args which are not null
            not_null_args: Dict[str, Any] = {}
            if self.args.basename_template is not None:
                not_null_args["basename_template"] = self.args.basename_template
            if self.args.partitions is not None:
                not_null_args["partitioning"] = self.args.partitions
                not_null_args["partitioning_flavor"] = "hive"
                # cast partition keys to string
                # ref: https://bneijt.nl/blog/write-polars-dataframe-as-parquet-dataset/
                table = table.cast(
                    pyarrow.schema(
                        [
                            f.with_type(pyarrow.string())
                            if f.name in self.args.partitions
                            else f
                            for f in table.schema
                        ]
                    )
                )
            if self.args.max_partitions is not None:
                not_null_args["max_partitions"] = self.args.max_partitions
            if self.args.max_open_files is not None:
                not_null_args["max_open_files"] = self.args.max_open_files
            if self.args.max_rows_per_file is not None:
                not_null_args["max_rows_per_file"] = self.args.max_rows_per_file
            if self.args.min_rows_per_group is not None:
                not_null_args["min_rows_per_group"] = self.args.min_rows_per_group
            if self.args.max_rows_per_group is not None:
                not_null_args["max_rows_per_group"] = self.args.max_rows_per_group

            # Build file_options: FileFormat specific write options
            # created using the FileFormat.make_write_options() function.
            if write_options:
                file_options = pyarrow.dataset.FileFormat.make_write_options(
                    **write_options
                )
                not_null_args["file_options"] = file_options

            # Write table to s3
            pyarrow.dataset.write_dataset(
                table,
                table_location,
                format=self.args.table_format.value,
                filesystem=fs,
                existing_data_behavior=self.args.write_mode,
                **not_null_args,
            )
            logger.info(f"Table {self.name} written to {table_location}")
            return True
        except Exception:
            logger.error("Could not write table: {}".format(self.name))
            raise

    def write_pandas_df(self, df: Optional[Any] = None) -> bool:
        logger.debug(f"@write_pandas_df not defined for {self.name}")
        return False

    ######################################################
    ## Read DataAsset
    ######################################################

    def read_df(self) -> Optional[Any]:
        """
        Read DataFrame from disk.
        """

        # LocalTable not yet initialized
        if self.args is None:
            return None

        # Check name is available
        if self.name is None:
            logger.error("Table name invalid")
            return None

        # Check table_location is available
        table_location = self.table_location
        if table_location is None:
            logger.error("Table location invalid")
            return False

        # Check S3FileSystem is available
        fs = self._get_fs()
        if fs is None:
            logger.error("Could not create LocalFileSystem")
            return False

        # Validate polars is installed
        try:
            import polars as pl
        except ImportError as ie:
            logger.error(f"Polars not installed: {ie}")
            return None

        # Validate pyarrow is installed
        try:
            import pyarrow as pa
            import pyarrow.dataset
        except ImportError as ie:
            logger.error(f"PyArrow not installed: {ie}")
            return None

        logger.debug("Format: {}".format(self.args.table_format.value))
        try:
            # Create a dict of args which are not null
            not_null_args: Dict[str, Any] = {}
            if self.args.partitions is not None:
                not_null_args["partitioning"] = "hive"

            # Read dataset from s3
            # https://arrow.apache.org/docs/python/generated/pyarrow.dataset.dataset.html#pyarrow.dataset.dataset
            # https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html#pyarrow.dataset.Dataset
            dataset: pyarrow.dataset.Dataset = pyarrow.dataset.dataset(
                table_location,
                format=self.args.table_format.value,
                filesystem=fs,
                **not_null_args,
            )

            # Convert dataset to polars DataFrame
            # https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.from_arrow.html
            df: Union[pl.DataFrame, pl.Series] = pl.from_arrow(dataset.to_table())

            # Run read checks
            if self.read_checks is not None:
                for check in self.read_checks:
                    if not check.check(df):
                        return None

            return df
        except Exception:
            logger.error("Could not read table: {}".format(self.name))
            raise

    def read_pandas_df(self) -> Optional[Any]:
        logger.debug(f"@read_pandas_df not defined for {self.name}")
        return False

    def _read(self) -> Any:
        logger.error(f"@_read not defined for {self.name}")
        return False

    ######################################################
    ## Update DataAsset
    ######################################################

    def _update(self) -> Any:
        logger.error(f"@_update not defined for {self.name}")
        return False

    def post_update(self) -> bool:
        return True

    ######################################################
    ## Delete DataAsset
    ######################################################

    def _delete(self) -> Any:
        logger.error(f"@_delete not defined for {self.name}")
        return False

    def post_delete(self) -> bool:
        return True