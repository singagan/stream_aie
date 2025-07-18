import logging
from typing import Any

from zigzag.parser.upgraded_validator import UpgradedValidator

logger = logging.getLogger(__name__)


class MappingValidator:
    """Class to validate user-given mappings from yaml file"""

    TILING_REGEX = r"^[A-Z]+, ([0-9]+|all|\*)$"
    SPATIAL_MAPPING_REGEX = r"^[A-Z]+, [0-9]+$"
    SPATIAL_MAPPING_HINT_REGEX = r"^[A-Z]+$"

    # Schema for a single operation, UpgradeValidator extrapolates to list of operations
    SCHEMA_SINGLE: Any = {
        "name": {"type": "string", "required": True},
        "core_allocation": {
            "type": "list",
            "schema": {"type": "integer"},
            "default": [0],
        },
        "inter_core_tiling": {
            "type": "list",
            "schema": {"type": "string", "regex": TILING_REGEX},
            "default": [],
        },
        "layer_dimension_names": {
            "type": "list",
            "schema": {"type": "string", "nullable": True},
            "default": [],
        },
        "intra_core_tiling": {
            "type": "list",
            "schema": {"type": "string", "regex": TILING_REGEX},
            "default": [],
        },
        "kernel": {
            "type": "dict",
            "schema": {
                "name": {"type": "string", "required": True},
                "utilization": {"type": "float", "required": True},
            },
            "required": True,
        },
        "spatial_mapping": {
            "type": "dict",
            "schema": {
                "D1": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_REGEX},
                    "required": False,
                },
                "D2": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_REGEX},
                    "required": False,
                },
                "D3": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_REGEX},
                    "required": False,
                },
                "D4": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_REGEX},
                    "required": False,
                },
            },
            "required": False,
            "nullable": True,
        },
        "memory_operand_links": {
            "type": "dict",
            "schema": {
                "O": {"type": "string", "required": True},
                "W": {"type": "string", "required": True},
                "I": {"type": "string", "required": True},
            },
            "default": {"O": "O", "I": "I1", "W": "I2"},
        },
        "spatial_mapping_hint": {
            "type": "dict",
            "schema": {
                "D1": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_HINT_REGEX},
                    "required": False,
                },
                "D2": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_HINT_REGEX},
                    "required": False,
                },
                "D3": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_HINT_REGEX},
                    "required": False,
                },
                "D4": {
                    "type": "list",
                    "schema": {"type": "string", "regex": SPATIAL_MAPPING_HINT_REGEX},
                    "required": False,
                },
            },
            "required": False,
        },
        "temporal_ordering": {
            "type": "list",
            "schema": {
                "type": "list",
                "items": [{"type": "string"}, {"oneof": [{"type": "integer"}, {"type": "string", "allowed": ["*"]}]}],
                "minlength": 2,
                "maxlength": 2,
            },
        },
    }

    def __init__(self, data: Any):
        """Initialize Validator object, assign schema and store normalize user-given data"""
        self.validator = UpgradedValidator(is_array=True)
        self.schema = MappingValidator.SCHEMA_SINGLE  # type: ignore
        self.data: list[dict[str, Any]] = self.validator.normalize_list(data, schema=self.schema)  # type: ignore
        self.is_valid = True

    @property
    def normalized_data(self):
        """! Return normalized, user-provided data."""
        # Can only be called after __init__, where data is automatically normalized
        return self.data

    def invalidate(self, extra_msg: str):
        self.is_valid = False
        logger.critical("User-defined mapping is invalid. %s", extra_msg)

    def validate(self) -> bool:
        """! Validate the user-provided accelerator data. Log a critical warning when invalid data is encountered and
        return true iff valid.
        """
        # Validate according to schema
        validate_success = self.validator.validate(self.data, schema=self.schema)  # type: ignore
        errors = self.validator.errors
        if not validate_success:
            self.invalidate(f"The following restrictions apply: {errors}")

        # Extra checks
        if "default" not in map(lambda x: x["name"], self.data):
            self.invalidate("No default mapping defined.")

        for mapping_data in self.data:
            self.validate_single_mapping(mapping_data)

        return self.is_valid

    def validate_single_mapping(self, layer_data: dict[str, Any]) -> None:
        """
        # TODO check that the inter-core splits do not exceed the number of cores
        """
        return
