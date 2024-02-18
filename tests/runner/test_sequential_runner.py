from __future__ import annotations

import re
from typing import Any

import pandas as pd
import pytest

from kedro.framework.hooks import _create_hook_manager
from kedro.io import (
    AbstractDataset,
    DataCatalog,
    DatasetError,
    LambdaDataset,
    MemoryDataset,
)
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.runner import SequentialRunner
from kedro.runner.runner import (
    _find_all_required_nodes,
    find_initial_node_group,
    find_nodes_to_resume_from,
)
from tests.runner.conftest import exception_fn, identity, sink, source


class TestValidSequentialRunner:
    def test_run_with_plugin_manager(self, fan_out_fan_in, catalog):
        catalog.add_feed_dict({"A": 42})
        result = SequentialRunner().run(
            fan_out_fan_in, catalog, hook_manager=_create_hook_manager()
        )
        assert "Z" in result
        assert result["Z"] == (42, 42, 42)

    def test_run_without_plugin_manager(self, fan_out_fan_in, catalog):
        catalog.add_feed_dict({"A": 42})
        result = SequentialRunner().run(fan_out_fan_in, catalog)
        assert "Z" in result
        assert result["Z"] == (42, 42, 42)

    def test_log_not_using_async(self, fan_out_fan_in, catalog, caplog):
        catalog.add_feed_dict({"A": 42})
        SequentialRunner().run(fan_out_fan_in, catalog)
        assert "Using synchronous mode for loading and saving data." in caplog.text


@pytest.mark.parametrize("is_async", [False, True])
class TestSeqentialRunnerBranchlessPipeline:
    def test_no_input_seq(self, is_async, branchless_no_input_pipeline, catalog):
        outputs = SequentialRunner(is_async=is_async).run(
            branchless_no_input_pipeline, catalog
        )
        assert "E" in outputs
        assert len(outputs) == 1

    def test_no_datasets(self, is_async, branchless_pipeline):
        catalog = DataCatalog({}, {"ds1": 42})
        outputs = SequentialRunner(is_async=is_async).run(branchless_pipeline, catalog)
        assert "ds3" in outputs
        assert outputs["ds3"] == 42

    def test_no_feed(self, is_async, memory_catalog, branchless_pipeline):
        outputs = SequentialRunner(is_async=is_async).run(
            branchless_pipeline, memory_catalog
        )
        assert "ds3" in outputs
        assert outputs["ds3"]["data"] == 42

    def test_node_returning_none(self, is_async, saving_none_pipeline, catalog):
        pattern = "Saving 'None' to a 'Dataset' is not allowed"
        with pytest.raises(DatasetError, match=pattern):
            SequentialRunner(is_async=is_async).run(saving_none_pipeline, catalog)

    def test_result_saved_not_returned(self, is_async, saving_result_pipeline):
        """The pipeline runs ds->dsX but save does not save the output."""

        def _load():
            return 0

        def _save(arg):
            assert arg == 0

        catalog = DataCatalog(
            {
                "ds": LambdaDataset(load=_load, save=_save),
                "dsX": LambdaDataset(load=_load, save=_save),
            }
        )
        output = SequentialRunner(is_async=is_async).run(
            saving_result_pipeline, catalog
        )

        assert output == {}


@pytest.mark.parametrize("is_async", [False, True])
class TestSequentialRunnerBranchedPipeline:
    def test_input_seq(
        self,
        is_async,
        memory_catalog,
        unfinished_outputs_pipeline,
        pandas_df_feed_dict,
    ):
        memory_catalog.add_feed_dict(pandas_df_feed_dict, replace=True)
        outputs = SequentialRunner(is_async=is_async).run(
            unfinished_outputs_pipeline, memory_catalog
        )
        assert set(outputs.keys()) == {"ds8", "ds5", "ds6"}
        # the pipeline runs ds2->ds5
        assert outputs["ds5"] == [1, 2, 3, 4, 5]
        assert isinstance(outputs["ds8"], dict)
        # the pipeline runs ds1->ds4->ds8
        assert outputs["ds8"]["data"] == 42
        # the pipline runs ds3
        assert isinstance(outputs["ds6"], pd.DataFrame)

    def test_conflict_feed_catalog(
        self,
        is_async,
        memory_catalog,
        unfinished_outputs_pipeline,
        conflicting_feed_dict,
    ):
        """ds1 and ds3 will be replaced with new inputs."""
        memory_catalog.add_feed_dict(conflicting_feed_dict, replace=True)
        outputs = SequentialRunner(is_async=is_async).run(
            unfinished_outputs_pipeline, memory_catalog
        )
        assert isinstance(outputs["ds8"], dict)
        assert outputs["ds8"]["data"] == 0
        assert isinstance(outputs["ds6"], pd.DataFrame)

    def test_unsatisfied_inputs(self, is_async, unfinished_outputs_pipeline, catalog):
        """ds1, ds2 and ds3 were not specified."""
        with pytest.raises(ValueError, match=r"not found in the DataCatalog"):
            SequentialRunner(is_async=is_async).run(
                unfinished_outputs_pipeline, catalog
            )


class LoggingDataset(AbstractDataset):
    def __init__(self, log, name, value=None):
        self.log = log
        self.name = name
        self.value = value

    def _load(self) -> Any:
        self.log.append(("load", self.name))
        return self.value

    def _save(self, data: Any) -> None:
        self.value = data

    def _release(self) -> None:
        self.log.append(("release", self.name))
        self.value = None

    def _describe(self) -> dict[str, Any]:
        return {}


@pytest.mark.parametrize("is_async", [False, True])
class TestSequentialRunnerRelease:
    def test_dont_release_inputs_and_outputs(self, is_async):
        log = []
        test_pipeline = modular_pipeline(
            [node(identity, "in", "middle"), node(identity, "middle", "out")]
        )
        catalog = DataCatalog(
            {
                "in": LoggingDataset(log, "in", "stuff"),
                "middle": LoggingDataset(log, "middle"),
                "out": LoggingDataset(log, "out"),
            }
        )
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)

        # we don't want to see release in or out in here
        assert log == [("load", "in"), ("load", "middle"), ("release", "middle")]

    def test_release_at_earliest_opportunity(self, is_async):
        log = []
        test_pipeline = modular_pipeline(
            [
                node(source, None, "first"),
                node(identity, "first", "second"),
                node(sink, "second", None),
            ]
        )
        catalog = DataCatalog(
            {
                "first": LoggingDataset(log, "first"),
                "second": LoggingDataset(log, "second"),
            }
        )
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)

        # we want to see "release first" before "load second"
        assert log == [
            ("load", "first"),
            ("release", "first"),
            ("load", "second"),
            ("release", "second"),
        ]

    def test_count_multiple_loads(self, is_async):
        log = []
        test_pipeline = modular_pipeline(
            [
                node(source, None, "dataset"),
                node(sink, "dataset", None, name="bob"),
                node(sink, "dataset", None, name="fred"),
            ]
        )
        catalog = DataCatalog({"dataset": LoggingDataset(log, "dataset")})
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)

        # we want to the release after both the loads
        assert log == [("load", "dataset"), ("load", "dataset"), ("release", "dataset")]

    def test_release_transcoded(self, is_async):
        log = []
        test_pipeline = modular_pipeline(
            [node(source, None, "ds@save"), node(sink, "ds@load", None)]
        )
        catalog = DataCatalog(
            {
                "ds@save": LoggingDataset(log, "save"),
                "ds@load": LoggingDataset(log, "load"),
            }
        )

        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)

        # we want to see both datasets being released
        assert log == [("release", "save"), ("load", "load"), ("release", "load")]

    @pytest.mark.parametrize(
        "test_pipeline",
        [
            modular_pipeline([node(identity, "ds1", "ds2", confirms="ds1")]),
            modular_pipeline(
                [
                    node(identity, "ds1", "ds2"),
                    node(identity, "ds2", None, confirms="ds1"),
                ]
            ),
        ],
    )
    def test_confirms(self, mocker, test_pipeline, is_async):
        fake_dataset_instance = mocker.Mock()
        catalog = DataCatalog(datasets={"ds1": fake_dataset_instance})
        SequentialRunner(is_async=is_async).run(test_pipeline, catalog)
        fake_dataset_instance.confirm.assert_called_once_with()


class TestSuggestResumeScenario:
    @pytest.mark.parametrize(
        "failing_node_names,expected_pattern",
        [
            (["node1_A"], r"No nodes ran."),
            (["node2"], r"(node1_A,node1_B|node1_B,node1_A)"),
            (["node3_A"], r"(node3_A,node3_B|node3_B,node3_A)"),
            (["node4_A"], r"(node3_A,node3_B|node3_B,node3_A)"),
            (["node3_A", "node4_A"], r"(node3_A,node3_B|node3_B,node3_A)"),
            (["node2", "node4_A"], r"(node1_A,node1_B|node1_B,node1_A)"),
        ],
    )
    def test_suggest_resume_scenario(
        self,
        caplog,
        two_branches_crossed_pipeline,
        persistent_dataset_catalog,
        failing_node_names,
        expected_pattern,
    ):
        nodes = {n.name: n for n in two_branches_crossed_pipeline.nodes}
        for name in failing_node_names:
            two_branches_crossed_pipeline -= modular_pipeline([nodes[name]])
            two_branches_crossed_pipeline += modular_pipeline(
                [nodes[name]._copy(func=exception_fn)]
            )
        with pytest.raises(Exception):
            SequentialRunner().run(
                two_branches_crossed_pipeline,
                persistent_dataset_catalog,
                hook_manager=_create_hook_manager(),
            )
        assert re.search(
            expected_pattern, caplog.text
        ), f"{expected_pattern=}, {caplog.text=}"

    @pytest.mark.parametrize(
        "failing_node_names,expected_pattern",
        [
            (["node1_A"], r"No nodes ran."),
            (["node2"], r'"node1_A,node1_B"'),
            (["node3_A"], r'"node3_A,node3_B"'),
            (["node4_A"], r'"node3_A,node3_B"'),
            (["node3_A", "node4_A"], r'"node3_A,node3_B"'),
            (["node2", "node4_A"], r'"node1_A,node1_B"'),
        ],
    )
    def test_stricter_suggest_resume_scenario(
        self,
        caplog,
        two_branches_crossed_pipeline_variable_inputs,
        persistent_dataset_catalog,
        failing_node_names,
        expected_pattern,
    ):
        """
        Stricter version of previous test.
        Covers pipelines where inputs are shared across nodes.
        """
        test_pipeline = two_branches_crossed_pipeline_variable_inputs

        nodes = {n.name: n for n in test_pipeline.nodes}
        for name in failing_node_names:
            test_pipeline -= modular_pipeline([nodes[name]])
            test_pipeline += modular_pipeline([nodes[name]._copy(func=exception_fn)])

        with pytest.raises(Exception, match="test exception"):
            SequentialRunner().run(
                test_pipeline,
                persistent_dataset_catalog,
                hook_manager=_create_hook_manager(),
            )
        assert re.search(
            expected_pattern, caplog.text
        ), f"{expected_pattern=}, {caplog.text=}"


class TestMemoryDatasetBehaviour:
    def test_run_includes_memory_datasets(self, pipeline_with_memory_datasets):
        # Create a catalog with MemoryDataset entries and inputs for the pipeline
        catalog = DataCatalog(
            {
                "Input1": LambdaDataset(load=lambda: "data1", save=lambda data: None),
                "Input2": LambdaDataset(load=lambda: "data2", save=lambda data: None),
                "MemOutput1": MemoryDataset(),
                "MemOutput2": MemoryDataset(),
            }
        )

        # Add a regular dataset to the catalog
        catalog.add("RegularOutput", LambdaDataset(None, None, lambda: True))

        # Run the pipeline
        output = SequentialRunner().run(pipeline_with_memory_datasets, catalog)

        # Check that MemoryDataset outputs are included in the run results
        assert "MemOutput1" in output
        assert "MemOutput2" in output
        assert (
            "RegularOutput" not in output
        )  # This output is registered in DataCatalog and so should not be in free outputs


class TestResumeLogicBehaviour:
    @pytest.mark.parametrize(
        "remaining_node_names,expected_result",
        [
            (["node3", "node4", "node2"], {"node2", "node1"}),
            (["node3", "node4"], {"node2", "node1"}),
            (["node4"], {"node2", "node1"}),
            ([], set()),
        ],
    )
    def test_simple_pipeline(
        self,
        pipeline_asymmetric,
        persistent_dataset_catalog,
        remaining_node_names,
        expected_result,
    ):
        """
        ...
        """
        remaining_nodes = pipeline_asymmetric.only_nodes(*remaining_node_names).nodes
        result_nodes = find_nodes_to_resume_from(
            pipeline_asymmetric, remaining_nodes, persistent_dataset_catalog
        )
        result_node_names = {n.name for n in result_nodes}
        assert expected_result == result_node_names

    @pytest.mark.parametrize(
        "remaining_node_names",
        [
            ["node3", "node4", "node2"],
            ["node3", "node4"],
            ["node4"],
            [],
        ],
    )
    def test_pipeline_with_persistent_datasets(
        self,
        pipeline_asymmetric_persistent_datasets,
        persistent_dataset_catalog,
        remaining_node_names,
    ):
        """
        ...
        """
        remaining_nodes = pipeline_asymmetric_persistent_datasets.only_nodes(
            *remaining_node_names
        ).nodes
        result_nodes = find_nodes_to_resume_from(
            pipeline_asymmetric_persistent_datasets,
            remaining_nodes,
            persistent_dataset_catalog,
        )
        result_node_names = {n.name for n in result_nodes}
        # Superset due to "compression" via initial node group
        assert set(remaining_node_names).issuperset(result_node_names)

    def test_empty_pipeline(self):
        """
        ...
        """
        test_pipeline = modular_pipeline([])
        result_nodes = find_nodes_to_resume_from(test_pipeline, [], DataCatalog())
        assert result_nodes == set()

    @pytest.mark.parametrize(
        "remaining_node_names",
        [
            ["node3", "node4", "node2"],
            ["node3", "node4"],
            ["node4"],
            [],
        ],
    )
    def test_initial_node_group_always_within(
        self,
        pipeline_asymmetric,
        remaining_node_names,
    ):
        """
        ...
        """
        remaining_nodes = pipeline_asymmetric.only_nodes(*remaining_node_names).nodes
        result_nodes = find_initial_node_group(
            pipeline_asymmetric,
            remaining_nodes,
        )
        # Superset due to "compression" via initial node group
        assert set(remaining_nodes).issuperset(result_nodes)

    def test_triangular_pipeline(
        self,
        pipeline_triangular,
        persistent_dataset_catalog,
    ):
        """
        Test that for nodes in result pipeline, no start nodes
        """
        remaining_nodes = pipeline_triangular.only_nodes("node3").nodes
        result_nodes = _find_all_required_nodes(
            pipeline_triangular, remaining_nodes, persistent_dataset_catalog
        )
        result_node_names = {n.name for n in result_nodes}
        assert {"node1", "node2", "node3"} == result_node_names

    @pytest.mark.parametrize(
        "remaining_node_names,expected_result",
        [
            (
                [
                    "node1_A",
                    "node1_B",
                    "node2",
                    "node3_A",
                    "node3_B",
                    "node4_A",
                    "node4_B",
                ],
                {"node1_A", "node1_B"},
            ),
            (
                ["node2", "node3_A", "node3_B", "node4_A", "node4_B"],
                {"node1_A", "node1_B"},
            ),
            (["node3_A", "node3_B", "node4_A", "node4_B"], {"node3_A", "node3_B"}),
            (["node4_A", "node4_B"], {"node3_A", "node3_B"}),
            (["node4_A"], {"node3_A"}),
            (["node3_A", "node4_A"], {"node3_A"}),
        ],
    )
    def test_crossed_pipeline(
        self,
        two_branches_crossed_pipeline_variable_inputs,
        persistent_dataset_catalog,
        remaining_node_names,
        expected_result,
    ):
        """ """
        test_pipeline = two_branches_crossed_pipeline_variable_inputs

        remaining_nodes = test_pipeline.only_nodes(*remaining_node_names).nodes
        result_nodes = find_nodes_to_resume_from(
            test_pipeline, remaining_nodes, persistent_dataset_catalog
        )
        result_node_names = {n.name for n in result_nodes}
        assert expected_result == result_node_names
