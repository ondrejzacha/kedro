from random import random

import pandas as pd
import pytest

from kedro.io import DataCatalog, LambdaDataSet, MemoryDataSet
from kedro.pipeline import node, pipeline


def source():
    return "stuff"


def identity(arg):
    return arg


def first_arg(*args):
    return args[0]


def sink(arg):  # pylint: disable=unused-argument
    pass


def fan_in(*args):
    return args


def exception_fn(*args):
    raise Exception("test exception")  # pylint: disable=broad-exception-raised


def return_none(arg):
    arg = None
    return arg


def return_not_serialisable(arg):  # pylint: disable=unused-argument
    return lambda x: x


def multi_input_list_output(arg1, arg2, arg3=None):  # pylint: disable=unused-argument
    return [arg1, arg2]


@pytest.fixture
def conflicting_feed_dict(pandas_df_feed_dict):
    ds1 = MemoryDataSet({"data": 0})
    ds3 = pandas_df_feed_dict["ds3"]
    return {"ds1": ds1, "ds3": ds3}


@pytest.fixture
def pandas_df_feed_dict():
    pandas_df = pd.DataFrame({"Name": ["Alex", "Bob"], "Age": [15, 25]})
    return {"ds3": pandas_df}


@pytest.fixture
def catalog():
    return DataCatalog()


@pytest.fixture
def memory_catalog():
    ds1 = MemoryDataSet({"data": 42})
    ds2 = MemoryDataSet([1, 2, 3, 4, 5])
    return DataCatalog({"ds1": ds1, "ds2": ds2})


@pytest.fixture
def persistent_dataset_catalog():
    def _load():
        return 0

    # pylint: disable=unused-argument
    def _save(arg):
        pass

    persistent_dataset = LambdaDataSet(load=_load, save=_save)
    return DataCatalog(
        {
            "ds0_A": persistent_dataset,
            "ds0_B": persistent_dataset,
            "ds2_A": persistent_dataset,
            "ds2_B": persistent_dataset,
            "dsX": persistent_dataset,
            "params:p": MemoryDataSet(1),
        }
    )


@pytest.fixture
def fan_out_fan_in():
    return pipeline(
        [
            node(identity, "A", "B"),
            node(identity, "B", "C"),
            node(identity, "B", "D"),
            node(identity, "B", "E"),
            node(fan_in, ["C", "D", "E"], "Z"),
        ]
    )


@pytest.fixture
def branchless_no_input_pipeline():
    """The pipeline runs in the order A->B->C->D->E."""
    return pipeline(
        [
            node(identity, "D", "E", name="node1"),
            node(identity, "C", "D", name="node2"),
            node(identity, "A", "B", name="node3"),
            node(identity, "B", "C", name="node4"),
            node(random, None, "A", name="node5"),
        ]
    )


@pytest.fixture
def branchless_pipeline():
    return pipeline(
        [
            node(identity, "ds1", "ds2", name="node1"),
            node(identity, "ds2", "ds3", name="node2"),
        ]
    )


@pytest.fixture
def saving_result_pipeline():
    return pipeline([node(identity, "ds", "dsX")])


@pytest.fixture
def saving_none_pipeline():
    return pipeline(
        [node(random, None, "A"), node(return_none, "A", "B"), node(identity, "B", "C")]
    )


@pytest.fixture
def unfinished_outputs_pipeline():
    return pipeline(
        [
            node(identity, {"arg": "ds4"}, "ds8", name="node1"),
            node(sink, "ds7", None, name="node2"),
            node(multi_input_list_output, ["ds3", "ds4"], ["ds6", "ds7"], name="node3"),
            node(identity, "ds2", "ds5", name="node4"),
            node(identity, "ds1", "ds4", name="node5"),
        ]
    )  # Outputs: ['ds8', 'ds5', 'ds6'] == ['ds1', 'ds2', 'ds3']


@pytest.fixture(
    params=[(), ("dsX",), ("params:p",)],
    ids=[
        "no_extras",
        "extra_persistent_ds",
        "extra_param",
    ],
)
def two_branches_crossed_pipeline(request):
    """A ``Pipeline`` with an X-shape (two branches with one common node).
    Non-persistent datasets (other than parameters) are prefixed with an underscore.
    """
    extra_inputs = list(request.param)

    return pipeline(
        [
            node(first_arg, ["ds0_A"] + extra_inputs, "_ds1_A", name="node1_A"),
            node(first_arg, ["ds0_B"] + extra_inputs, "_ds1_B", name="node1_B"),
            node(
                multi_input_list_output,
                ["_ds1_A", "_ds1_B"] + extra_inputs,
                ["ds2_A", "ds2_B"],
                name="node2",
            ),
            node(first_arg, ["ds2_A"] + extra_inputs, "_ds3_A", name="node3_A"),
            node(first_arg, ["ds2_B"] + extra_inputs, "_ds3_B", name="node3_B"),
            node(first_arg, ["_ds3_A"] + extra_inputs, "_ds4_A", name="node4_A"),
            node(first_arg, ["_ds3_B"] + extra_inputs, "_ds4_B", name="node4_B"),
        ]
    )
